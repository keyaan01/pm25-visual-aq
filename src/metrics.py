"""Evaluation metrics (paper Sec 3.11), all in AQI points.

We report:
* point-accuracy of the median: MAE, RMSE, R2, and Spearman rank correlation;
* interval quality: coverage (how often the truth is inside) and mean width;
* error broken down by true-AQI magnitude (the visual signal is weak at low pollution,
  and aggregate numbers hide that);
* EPA-category accuracy (to compare with the classification literature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .aqi import AQI_CATEGORIES, aqi_category_index
from .calibrate import coverage, mean_width


def point_metrics(y_true: np.ndarray, y_median: np.ndarray) -> dict:
    """MAE, RMSE, R2, Spearman rank correlation for the median prediction."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_median)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_median))),
        "R2": float(r2_score(y_true, y_median)),
        "Spearman": float(spearmanr(y_true, y_median).statistic),
    }


def interval_metrics(preds: np.ndarray, y_true: np.ndarray, target: float = 0.90) -> dict:
    """Coverage vs target and mean interval width."""
    return {
        "coverage": coverage(preds, y_true),
        "target_coverage": target,
        "mean_width": mean_width(preds),
    }


def full_report(preds: np.ndarray, y_true: np.ndarray, target: float = 0.90) -> dict:
    """All headline numbers for a set of calibrated predictions.

    preds : (N, 3) calibrated [q05, q50, q95] in AQI; y_true : (N,) AQI.
    """
    out = point_metrics(y_true, preds[:, 1])
    out.update(interval_metrics(preds, y_true, target))
    out["category_accuracy"] = category_accuracy(y_true, preds[:, 1])
    return out


def report(point_pred: np.ndarray, interval_preds: np.ndarray, y_true: np.ndarray,
           target: float = 0.90) -> dict:
    """Headline numbers using the dedicated point head for accuracy + quantiles for intervals.

    point_pred    : (N,) smeared point estimate in AQI (drives MAE/RMSE/R²/category accuracy).
    interval_preds: (N, 3) conformally-calibrated [q05, q50, q95] in AQI (drives coverage/width).
    """
    out = point_metrics(y_true, point_pred)
    out.update(interval_metrics(interval_preds, y_true, target))
    out["category_accuracy"] = category_accuracy(y_true, point_pred)
    return out


def category_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of predicted EPA category (Good, Moderate, …) from the median."""
    t = np.array([aqi_category_index(v) for v in y_true])
    p = np.array([aqi_category_index(v) for v in y_pred])
    return float((t == p).mean())


def error_by_magnitude(y_true: np.ndarray, y_pred: np.ndarray,
                       edges=(0, 50, 100, 150, 200, 300, 10_000)) -> pd.DataFrame:
    """MAE within each true-AQI band — reveals where the model is weak (usually low AQI)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        rows.append({"band": f"{lo}-{hi if hi < 10_000 else '+'}",
                     "n": int(m.sum()),
                     "MAE": float(mean_absolute_error(y_true[m], y_pred[m]))})
    return pd.DataFrame(rows)
