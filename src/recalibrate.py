"""Isotonic recalibration of the point estimate (accuracy fix, research-backed).

WHY
---
The model systematically **under-predicts high AQI** (the point estimate caps well below the
true value for the rare, extreme-pollution photos). That systematic bias is what crushes the
coefficient-of-determination R² even though the model *ranks* pollution well.

Isotonic regression learns a **monotonic** mapping g: prediction -> true value on a held-out
set. Applied to new predictions it removes systematic bias (e.g. "when the model says ~260 the
truth averages ~340, so map 260 -> 340") while preserving the ordering. It cannot invent
resolution the model lacks, but it removes the bias term, which directly improves R² and MAE.

RIGOUR
------
We fit the mapping on the **calibration** split only and apply it to test — never fit on test.
Monotonic by construction, so it can't scramble the model's ranking. This is a standard,
low-risk post-hoc recalibration; no retraining required.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(cal_point: np.ndarray, cal_true: np.ndarray) -> IsotonicRegression:
    """Fit a monotonic prediction->truth map on the calibration set."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    iso.fit(np.asarray(cal_point, dtype=float), np.asarray(cal_true, dtype=float))
    return iso


def apply_isotonic(iso: IsotonicRegression, point: np.ndarray) -> np.ndarray:
    """Apply a fitted isotonic map to new point predictions (clipped at 0)."""
    return np.maximum(iso.predict(np.asarray(point, dtype=float)), 0.0)
