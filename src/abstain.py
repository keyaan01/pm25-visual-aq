"""C3 — inference-time abstention (paper Sec 3.9).

THE IDEA
--------
An honest system must sometimes say **"I can't answer this."** Two separate reasons to refuse, checked
in this order:

1. **OOD / unusable gate (checked FIRST).** Is this even a valid daytime street photo? Night shots,
   indoor scenes, and featureless textures are *out of distribution* — the model has no business
   guessing on them. We score "how far from the training photos" in the network's own **feature space**
   (a shrinkage **Mahalanobis** distance on the backbone features), with interpretable
   brightness / detail / sky-fraction checks as a human-readable backup. We do **NOT** use interval
   width here: conformal validity is only guaranteed on in-distribution data, so a width-based OOD test
   would be meaningless off-distribution.

2. **Uncertainty gate (checked second).** Among valid photos, refuse the ones the model is least sure
   about — measured by the **log-space** interval width `q95_log - q05_log`. (Log space, not raw AQI:
   the log target makes raw AQI width grow with magnitude, so a raw-width rule would just refuse every
   high-AQI photo.)

Both thresholds are chosen on the **calibration** split only (no OOD data needed to set the OOD
threshold — we fit a 5% false-refusal budget on good images).

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
# OOD score 1 (primary): shrinkage Mahalanobis distance in backbone-feature space
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
# OOD score 2 (interpretable backup): handcrafted checks on the RGB image in [0,1]
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
    (indoor / occluded), which the outdoor-haze model relies on."""
    a = np.asarray(rgb01, dtype=float)
    h = max(1, int(a.shape[0] * top_frac))
    top = a[:h]
    bright = top.mean(axis=2)
    sat = top.max(axis=2) - top.min(axis=2)
    return float(((bright > 0.5) & (sat < 0.2)).mean())


def handcrafted_ood_score(rgb01) -> float:
    """A single interpretable 'unusableness' score in [0,1]: dark OR featureless OR no-sky raises it.
    A transparent baseline/backup to the Mahalanobis score (never the sole gate)."""
    dark = max(0.0, (0.25 - brightness(rgb01)) / 0.25)          # 1 when pitch black, 0 by 0.25
    flat = max(0.0, (0.003 - detail_variance(rgb01)) / 0.003)   # 1 when perfectly flat
    nosky = 1.0 - min(1.0, sky_fraction(rgb01) / 0.15)          # 1 when no sky at all
    return float(max(dark, flat, nosky))


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
    """tau_ood = the (1 - budget) percentile of the calibration good-image OOD scores.

    Setting it on good images alone gives an explicit false-refusal budget (default 5%) with NO OOD
    data — so the threshold is honest and not tuned to the test-time OOD sets."""
    s = np.asarray(cal_good_scores, dtype=float)
    return float(np.percentile(s, 100.0 * (1.0 - false_refusal_budget)))


def width_threshold(cal_widths, target_coverage: float = 0.90) -> float:
    """tau_width = accept the `target_coverage` fraction of calibration photos with the SMALLEST width.

    i.e. the `target_coverage` percentile of calibration log-widths. At serve time we answer when the
    width is below this and abstain above it, targeting ~`target_coverage` coverage on good inputs."""
    w = np.asarray(cal_widths, dtype=float)
    return float(np.percentile(w, 100.0 * target_coverage))


# ---------------------------------------------------------------------------
# the combined gate (OOD first, then uncertainty)
# ---------------------------------------------------------------------------
def decide(ood_score: float, log_width: float, tau_ood: float, tau_width: float) -> dict:
    """Return {'answer': bool, 'reason': str}. OOD is checked BEFORE uncertainty."""
    if ood_score > tau_ood:
        return {"answer": False, "reason": "unusable / out-of-distribution photo (night, indoor, or no sky)"}
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
