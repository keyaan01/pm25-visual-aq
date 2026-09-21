"""Pinball (quantile) loss (paper Sec 3.7, Eq. 2).

Ordinary regression minimises squared error, which targets the *mean*. To make a head
learn a specific *percentile* instead, we penalise being too low and too high by
*different* amounts:

    L_tau(y, yhat) = max( tau*(y - yhat),  (tau - 1)*(y - yhat) )

For tau = 0.95, under-prediction (guessing too low) is penalised 19x more than
over-prediction, so that head learns to sit near the top of the plausible range. For
tau = 0.05 it's the reverse; for tau = 0.50 the penalty is symmetric and this reduces to
ordinary absolute error. One loss, three percentiles, three behaviours.
"""
from __future__ import annotations

import torch


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """Mean pinball loss over a batch and all quantiles.

    preds    : (B, Q) predicted quantiles (ascending), in the training target space.
    target   : (B,) or (B, 1) true values (same space as preds, i.e. log-AQI).
    quantiles: iterable of Q tau values, e.g. (0.05, 0.50, 0.95).
    """
    if target.ndim == 1:
        target = target.unsqueeze(1)                      # (B, 1)
    q = torch.as_tensor(quantiles, device=preds.device, dtype=preds.dtype).unsqueeze(0)  # (1, Q)
    error = target - preds                                # (B, Q)
    loss = torch.maximum(q * error, (q - 1.0) * error)    # asymmetric penalty
    return loss.mean()
