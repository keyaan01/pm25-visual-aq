"""Conformalized Quantile Regression (paper Sec 3.8).

The trained model gives us a low/median/high range, but nothing yet *guarantees* the true
value falls inside it 90% of the time — the model could be over-confident (too narrow) or
under-confident (too wide). Conformal prediction fixes this with one held-out calibration
set and a sorting operation, and the guarantee holds for **any** model, however good or
bad (a weak model just gets honestly-wide intervals), as long as calibration and test data
come from the same distribution.

Recipe (Romano, Patterson & Candès, 2019):
1. On the calibration set, record how far outside the predicted range each truth fell:
       E_i = max( qhat_lo(x_i) - y_i,  y_i - qhat_hi(x_i) )
   (negative when the truth was comfortably inside; positive by the size of the miss.)
2. Take Q = the finite-sample (1-alpha) quantile of those E_i.
3. Widen every future interval by Q at both ends (clip the lower end at 0 — AQI can't be
   negative). That single number is the whole calibration step.
"""
from __future__ import annotations

import numpy as np


def conformal_Q(preds_cal: np.ndarray, y_cal: np.ndarray, coverage: float = 0.90) -> float:
    """Compute the conformal widening `Q` from calibration predictions/targets (AQI units).

    preds_cal : (N, 3) array of [q05, q50, q95] in AQI.
    y_cal     : (N,) true AQI values.
    """
    lo, hi = preds_cal[:, 0], preds_cal[:, 2]
    E = np.maximum(lo - y_cal, y_cal - hi)                 # nonconformity scores
    n = len(E)
    # finite-sample correction: the ceil((n+1)(coverage))/n empirical quantile
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(E, level, method="higher"))


def apply_conformal(preds: np.ndarray, Q: float) -> np.ndarray:
    """Return a copy of `preds` with the interval widened by `Q` (lower end clipped at 0)."""
    out = preds.astype(float).copy()
    out[:, 0] = np.maximum(out[:, 0] - Q, 0.0)
    out[:, -1] = out[:, -1] + Q
    return out


def coverage(preds: np.ndarray, y: np.ndarray) -> float:
    """Fraction of points whose true value lies within [q_lo, q_hi]."""
    inside = (y >= preds[:, 0]) & (y <= preds[:, -1])
    return float(inside.mean())


def mean_width(preds: np.ndarray) -> float:
    """Average interval width (q_hi - q_lo), in AQI points."""
    return float((preds[:, -1] - preds[:, 0]).mean())
