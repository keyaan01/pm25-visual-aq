"""Leakage-safe K-fold cross-validation (grouped by station) — rigour for the honest R².

WHY GROUPED, NOT PLAIN K-FOLD
-----------------------------
Ordinary k-fold would LEAK: photos from the same monitoring station (which share a near-identical
daily-average label) would land in both train and test, inflating the score — exactly the flaw this
project exposes. So we use **sklearn GroupKFold on `station_id`**: every station is tested in exactly
one fold and never appears in its own fold's training data. Within each fold's training stations we
carve a station-grouped **calibration** set (needed for conformal), giving the usual
0.65 / 0.15 / 0.20 train / cal / test per fold.

WHAT IT BUYS
------------
A single split gives one R² number; five station-disjoint folds give **R² (and MAE, coverage, …) as a
mean ± 95% CI**, so we can say "0.22 ± CI" rather than hoping one split wasn't lucky. Training and
evaluation reuse `leakage.train_and_evaluate`, so each fold is apples-to-apples with the headline
single-split run.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

from . import leakage as L
from . import splits as S


def make_cv_folds(df, n_splits=5, cal_frac=0.1875, seed=42, station_col="station_id"):
    """Return `n_splits` split-frames (copies of `df` with a train/cal/test 'split' column).

    GroupKFold(station) picks each fold's TEST stations (disjoint from train); the remaining stations
    are split station-grouped into train + cal at `cal_frac` (0.1875 of the 0.8 non-test share -> 0.15
    overall). Each fold is therefore leakage-safe by construction (0 straddling stations).
    """
    assert df[station_col].notna().all(), f"{station_col} has NaN -- would misassign folds"
    stations = df[station_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for k, (train_idx, test_idx) in enumerate(gkf.split(df, groups=stations)):
        lab = np.array(["train"] * len(df), dtype=object)
        lab[test_idx] = "test"
        train_df = df.iloc[train_idx]
        # carve a station-grouped calibration set out of this fold's training stations
        grp = S._group_split(train_df[station_col], (1 - cal_frac, cal_frac, 0.0), seed=seed + k)
        for j, g in zip(train_idx, grp):
            lab[j] = "cal" if g == "cal" else "train"
        sp = df.copy()
        sp["split"] = lab
        sp.attrs["split_strategy"] = f"cv_fold_{k}"
        folds.append(sp)
    return folds


def summarize_cv(reports, keys=("R2", "r2_pearson", "MAE", "RMSE", "Spearman", "coverage")):
    """Mean, std and a 95% CI (Student-t) across folds for each metric."""
    n = len(reports)
    tcrit = float(stats.t.ppf(0.975, max(1, n - 1)))
    out = {}
    for kx in keys:
        vals = np.array([r[kx] for r in reports if kx in r], dtype=float)
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        half = tcrit * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[kx] = {"mean": m, "std": sd, "ci_lo": m - half, "ci_hi": m + half, "n": int(len(vals))}
    return out


def run_cv(ds, df, cache, cfg, device, out_root, n_splits=5, seed=None, num_workers=2,
           max_epochs=None, verbose=False):
    """Train + evaluate every station-disjoint fold. Returns (per_fold_reports, summary_dict).

    Each fold reuses `leakage.train_and_evaluate` with a pre-built fold split, so the setup is
    identical to the single-split honest run — only the fold assignment changes.
    """
    seed = seed if seed is not None else cfg["seed"]
    folds = make_cv_folds(df, n_splits=n_splits, seed=seed, station_col=cfg["data"]["station_col"])
    reports = []
    for k, sp in enumerate(folds):
        r = L.train_and_evaluate(ds, df, cache, cfg, f"fold_{k}", device, out_root, seed=seed,
                                 sp=sp, max_epochs=max_epochs, num_workers=num_workers, verbose=verbose)
        rep = r["report"]
        rep["fold"] = k
        reports.append(rep)
    return reports, summarize_cv(reports)
