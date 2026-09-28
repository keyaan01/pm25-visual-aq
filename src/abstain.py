"""C3 — inference-time abstention (paper Sec 3.9).

THE IDEA
--------
An honest system must sometimes say **"I can't answer this."** Two separate reasons to refuse, checked
in this order:

1. **OOD / unusable gate (checked FIRST).** Is this even a valid daytime street photo? Night shots and
   featureless textures are *out of distribution* — the model has no business guessing on them. The
   **deployed** gate is a pair of **interpretable, station-invariant** checks — **too dark** (night) or
   **too flat** (featureless) — because a valid daytime street photo from *any* station is bright and
   detailed. We *also* compute a shrinkage **Mahalanobis** distance in the backbone feature space and
   **report it for comparison only**: it over-refuses valid photos from unseen stations (it confuses
   "new station" with "unusable"), so it is NOT the deployed gate. Indoor scenes are *near*-OOD that the
   handcrafted checks do NOT catch (a lit indoor photo is bright and detailed) — a documented limitation
   the reported Mahalanobis would catch. We do **NOT** use interval width here: conformal validity is
   only guaranteed on in-distribution data, so a width-based OOD test would be meaningless off-distribution.

2. **Uncertainty gate (checked second).** Among valid photos, refuse the ones the model is least sure
   about — measured by the **log-space** interval width `q95_log - q05_log`. (Log space, not raw AQI:
   the log target makes raw AQI width grow with magnitude, so a raw-width rule would just refuse every
   high-AQI photo.)

Both thresholds are chosen on **good images only** (no OOD data, no labels): the OOD threshold fits a 5%
false-refusal budget on held-out, deployment-representative (unseen-station) good photos; the width
threshold is the target-coverage log-width on the calibration split.

WHAT WE MEASURE
---------------
- Selective prediction: **risk-coverage** curves (selective MAE / miscoverage / category-error), the
  area under them (**AURC**, and excess-AURC vs an oracle), and metrics at 90/80/70% coverage.
- OOD detection: **AUROC, AUPR, FPR@95TPR**, near-OOD (indoor) vs far-OOD (night/textures), and a
  refuse-rate table.

Pure + unit-testable; the network/data live in the notebook. `physics.five_channel` routes every OOD
image through the identical pipeline used for real photos.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score


def _trapz(y, x):
    """Trapezoidal integral, version-agnostic (np.trapz was removed in numpy 2.0 -> np.trapezoid)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))

# ---------------------------------------------------------------------------
# OOD score — REPORTED ONLY (comparison, NOT deployed): shrinkage Mahalanobis distance in
# backbone-feature space. It over-refuses valid unseen-station photos, so the deployed gate is the
# handcrafted station-invariant score below; we keep this to quantify that over-refusal honestly.
# ---------------------------------------------------------------------------
def fit_mahalanobis(feats_cal_good) -> dict:
    """Fit a Ledoit-Wolf shrinkage Gaussian to the calibration good-image features.

    Shrinkage keeps the covariance well-conditioned even when the feature dimension (~1280) is large
    relative to the number of calibration images. Returns {"mean": (D,), "precision": (D, D)} — small
    enough to save beside the model (mean + precision), and used by `mahalanobis_score`.
    """
    X = np.asarray(feats_cal_good, dtype=np.float64)
    lw = LedoitWolf().fit(X)
    return {"mean": lw.location_.astype(np.float64), "precision": lw.precision_.astype(np.float64)}


def mahalanobis_score(maha: dict, feats) -> np.ndarray:
    """Mahalanobis distance of each feature row from the fitted good-image Gaussian (higher = more OOD)."""
    X = np.asarray(feats, dtype=np.float64)
    d = X - maha["mean"][None, :]
    m2 = np.einsum("ni,ij,nj->n", d, maha["precision"], d)   # squared Mahalanobis
    return np.sqrt(np.maximum(m2, 0.0))


def save_mahalanobis(out_dir: str, maha: dict):
    """Save the Mahalanobis params as .npy beside the bundle (precision can be a few MB — not JSON)."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "maha_mean.npy"), maha["mean"])
    np.save(os.path.join(out_dir, "maha_precision.npy"), maha["precision"])


def load_mahalanobis(out_dir: str):
    """Load the Mahalanobis params if present, else None (demo can fall back to handcrafted checks)."""
    mp = os.path.join(out_dir, "maha_mean.npy")
    pp = os.path.join(out_dir, "maha_precision.npy")
    if not (os.path.exists(mp) and os.path.exists(pp)):
        return None
    return {"mean": np.load(mp), "precision": np.load(pp)}


def knn_ood_score(feats, ref_feats, k: int = 5) -> np.ndarray:
    """Secondary OOD score: distance to the k-th nearest calibration good-image feature."""
    from sklearn.neighbors import NearestNeighbors
    ref = np.asarray(ref_feats, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    dist, _ = nn.kneighbors(np.asarray(feats, dtype=np.float64))
    return dist[:, -1]


# ---------------------------------------------------------------------------
# OOD score — the DEPLOYED gate: interpretable, station-invariant checks on the RGB image in [0,1]
# ---------------------------------------------------------------------------
def brightness(rgb01) -> float:
    """Mean brightness in [0,1]. Very low = night / underexposed (a classic unusable photo)."""
    return float(np.asarray(rgb01, dtype=float).mean())


def detail_variance(rgb01) -> float:
    """Spatial detail = variance of the grayscale image. Near-zero = a flat, featureless texture."""
    g = np.asarray(rgb01, dtype=float).mean(axis=2)
    return float(g.var())


def sky_fraction(rgb01, top_frac: float = 0.34) -> float:
    """Fraction of the UPPER frame that looks like sky (bright + low-saturation). Low = no sky visible
    (indoor / occluded). NOTE: kept for the *reported* near-OOD (indoor) analysis only — it is NOT part
    of the deployed gate, because many valid urban street photos have little visible sky, so a sky
    requirement would wrongly refuse them."""
    a = np.asarray(rgb01, dtype=float)
    h = max(1, int(a.shape[0] * top_frac))
    top = a[:h]
    bright = top.mean(axis=2)
    sat = top.max(axis=2) - top.min(axis=2)
    return float(((bright > 0.5) & (sat < 0.2)).mean())


def handcrafted_components(rgb01):
    """The two interpretable OOD terms, each in [0,1]: (dark, flat)."""
    dark = max(0.0, (0.25 - brightness(rgb01)) / 0.25)          # 1 when pitch black, 0 by brightness 0.25
    flat = max(0.0, (0.003 - detail_variance(rgb01)) / 0.003)   # 1 when perfectly flat / featureless
    return float(dark), float(flat)


def handcrafted_ood_score(rgb01) -> float:
    """A single interpretable 'unusableness' score in [0,1]: **too dark** (night) OR **too flat**
    (featureless texture) raises it. Both are STATION-INVARIANT — a valid daytime street photo from any
    station is bright and detailed — so this does NOT over-refuse unseen-station photos the way a
    feature-space Mahalanobis distance does. We deliberately do NOT include a sky-fraction term in the
    gate: many valid urban street photos have little visible sky, so requiring sky would wrongly refuse
    them (indoor scenes are 'near-OOD' this gate does NOT catch — a documented limitation)."""
    dark, flat = handcrafted_components(rgb01)
    return float(max(dark, flat))


def handcrafted_reason(rgb01) -> str:
    """Human-readable reason naming which handcrafted term dominates (for the abstain badge)."""
    dark, flat = handcrafted_components(rgb01)
    return "too dark (night / underexposed)" if dark >= flat else "too flat / featureless"


# ---------------------------------------------------------------------------
# uncertainty score: LOG-space interval width
# ---------------------------------------------------------------------------
def log_interval_width(q_log) -> np.ndarray:
    """q95_log - q05_log from (N, 3) log-space quantiles. The uncertainty score for the second gate."""
    q = np.asarray(q_log, dtype=float)
    return q[:, -1] - q[:, 0]


# ---------------------------------------------------------------------------
# thresholds — fit on CALIBRATION only
# ---------------------------------------------------------------------------
def ood_threshold(cal_good_scores, false_refusal_budget: float = 0.05) -> float:
    """tau_ood = the smallest threshold that refuses at most `budget` of the good images.

    Setting it on good images alone gives an explicit false-refusal budget (default 5%) with NO OOD
    data — so the threshold is honest and not tuned to the test-time OOD sets. We use an exact **top-k**
    rule (refuse the k = floor(budget * n) highest-scoring good images) rather than a raw percentile,
    because the handcrafted score is **spiked at exactly 0** for most good photos: a plain percentile
    can collapse to 0 and then refuse every mildly-dim photo. With top-k, `scores > tau` is at most k,
    and when few scores are positive tau falls at 0 so only the genuinely dark/flat photos refuse — the
    budget is never exceeded either way."""
    s = np.sort(np.asarray(cal_good_scores, dtype=float))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.floor(false_refusal_budget * n))            # refuse at most k of the good images
    if k <= 0:
        return float(s[-1])                                # budget < 1 image -> refuse none (tau = max)
    return float(s[n - k - 1])                             # scores strictly above the k-th-largest refuse


def width_threshold(cal_widths, target_coverage: float = 0.90) -> float:
    """tau_width = accept the `target_coverage` fraction of calibration photos with the SMALLEST width.

    i.e. the `target_coverage` percentile of calibration log-widths. At serve time we answer when the
    width is below this and abstain above it, targeting ~`target_coverage` coverage on good inputs."""
    w = np.asarray(cal_widths, dtype=float)
    return float(np.percentile(w, 100.0 * target_coverage))


# ---------------------------------------------------------------------------
# the combined gate (OOD first, then uncertainty)
# ---------------------------------------------------------------------------
def decide(ood_score: float, log_width: float, tau_ood: float, tau_width: float,
           ood_reason: str | None = None) -> dict:
    """Return {'answer': bool, 'reason': str}. OOD is checked BEFORE uncertainty.

    `ood_reason` lets the caller name the specific trigger (e.g. "too dark (night)") for the badge; if
    omitted we use an honest generic message. The deployed gate catches night / featureless photos —
    NOT indoor scenes (a documented limitation), so we no longer claim indoor here."""
    if ood_score > tau_ood:
        reason = ood_reason or "out-of-distribution / unusable photo (too dark or too featureless)"
        return {"answer": False, "reason": reason}
    if log_width > tau_width:
        return {"answer": False, "reason": "model too uncertain on this photo (interval too wide)"}
    return {"answer": True, "reason": "confident, in-distribution"}


# ---------------------------------------------------------------------------
# selective-prediction evaluation (risk-coverage, AURC)
# ---------------------------------------------------------------------------
def risk_coverage_curve(uncertainty, errors):
    """Accept the most-confident (lowest-uncertainty) points first; return (coverages, risks).

    risk at coverage c = mean error over the accepted fraction. A good confidence signal makes risk
    rise monotonically as coverage grows (you're forced to accept harder inputs)."""
    u = np.asarray(uncertainty, dtype=float)
    e = np.asarray(errors, dtype=float)
    order = np.argsort(u, kind="mergesort")                    # most confident first
    e_sorted = e[order]
    n = len(e_sorted)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(e_sorted) / np.arange(1, n + 1)
    return coverages, risks


def aurc(uncertainty, errors) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(uncertainty, errors)
    return _trapz(risk, cov)


def excess_aurc(uncertainty, errors) -> float:
    """AURC minus the ORACLE AURC (sorting by the true error). >= 0; smaller = better confidence signal."""
    return float(aurc(uncertainty, errors) - aurc(np.asarray(errors, dtype=float), errors))


def augrc(uncertainty, errors) -> float:
    """Area under the Generalised Risk-Coverage curve (risk weighted by coverage) — a robustness check."""
    cov, risk = risk_coverage_curve(uncertainty, errors)
    return _trapz(risk * cov, cov)


def selective_metrics_at_coverage(uncertainty, y, point, intervals, target_coverage):
    """MAE / RMSE / EPA-category-acc / interval-coverage on the accepted (most-confident) fraction."""
    from .aqi import aqi_category_index
    u = np.asarray(uncertainty, dtype=float)
    order = np.argsort(u, kind="mergesort")
    k = max(1, int(round(target_coverage * len(u))))
    keep = order[:k]
    y, point = np.asarray(y)[keep], np.asarray(point)[keep]
    iv = np.asarray(intervals)[keep]
    cat_t = np.array([aqi_category_index(v) for v in y])
    cat_p = np.array([aqi_category_index(v) for v in point])
    return {
        "coverage_target": float(target_coverage),
        "n": int(k),
        "MAE": float(np.mean(np.abs(point - y))),
        "RMSE": float(np.sqrt(np.mean((point - y) ** 2))),
        "category_accuracy": float((cat_t == cat_p).mean()),
        "interval_coverage": float(((y >= iv[:, 0]) & (y <= iv[:, -1])).mean()),
    }


# ---------------------------------------------------------------------------
# OOD-detection metrics
# ---------------------------------------------------------------------------
def ood_metrics(id_scores, ood_scores) -> dict:
    """AUROC / AUPR / FPR@95TPR treating OOD as the positive class (higher score = more OOD)."""
    id_s = np.asarray(id_scores, dtype=float)
    ood_s = np.asarray(ood_scores, dtype=float)
    y = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    s = np.concatenate([id_s, ood_s])
    auroc = float(roc_auc_score(y, s))
    aupr = float(average_precision_score(y, s))
    # FPR when TPR (OOD recall) is >= 0.95: threshold = 5th percentile of OOD scores
    thr = np.percentile(ood_s, 5.0)
    fpr95 = float((id_s >= thr).mean())
    return {"AUROC": auroc, "AUPR": aupr, "FPR@95TPR": fpr95,
            "n_id": int(len(id_s)), "n_ood": int(len(ood_s))}


def refuse_rate(scores, tau) -> float:
    """Fraction of a set that would be refused at threshold `tau` (scores above tau are refused)."""
    return float((np.asarray(scores, dtype=float) > tau).mean())
