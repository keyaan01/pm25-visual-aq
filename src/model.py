"""The model: EfficientNet-B0 backbone + monotone quantile heads (paper Sec 3.6).

WHAT / WHY
----------
* **Backbone.** EfficientNet-B0 (5.3 M parameters), pretrained on ImageNet. It's small
  enough to train on our budget, its pretrained features help given our modest sample
  size, and it's the strongest published baseline on PM25Vision, so our numbers are
  directly comparable. We widen its first layer from 3 to 5 input channels (RGB +
  transmission + inverted saturation); `timm` initialises the two new channels from the
  pretrained RGB weights — a routine engineering step, not a research risk.

* **Three heads, guaranteed ordered.** We predict the 5th, 50th, and 95th percentiles.
  If we predicted them independently, the model could output a 95th percentile *below*
  its 50th — nonsense. We prevent this **by construction** (paper Eq. 1): predict the
  lowest quantile freely, then add strictly-positive `softplus` increments for each step
  up. Because every increment is positive, the outputs can only ever increase, so the
  low/median/high bounds never cross — no penalty term or post-hoc sorting needed.

The heads output in the model's target space (we train on `log(AQI)`), and because
exponentiating is monotone, the ordering is preserved when we convert back to AQI.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneQuantiles(nn.Module):
    """Map pooled features to `n` strictly-increasing quantiles (cumulative softplus)."""

    def __init__(self, in_features: int, n_quantiles: int = 3):
        super().__init__()
        # one base value + (n-1) positive increments
        self.fc = nn.Linear(in_features, n_quantiles)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        z = self.fc(feats)                       # (B, n)
        base = z[:, :1]                          # lowest quantile, unconstrained
        increments = F.softplus(z[:, 1:])        # (B, n-1), all > 0
        steps = torch.cumsum(increments, dim=1)  # running total
        return torch.cat([base, base + steps], dim=1)   # (B, n), ascending


class PM25QuantileNet(nn.Module):
    """EfficientNet-B0 (in_chans=5) + monotone quantile head + optional point head.

    `forward` returns a dict:
      - "quantiles": (B, n) ascending quantile predictions (log-target space) — for intervals.
      - "point":     (B,) a dedicated point estimate (log-target space) — for accuracy (R²/MAE),
                     trained with Huber loss. Present only when `point_head=True`.

    Why a separate point head? The median quantile minimises absolute error, which on a
    right-skewed target predicts low and hurts R² (R² rewards matching the conditional mean).
    A Huber-trained point head (plus a smearing correction at eval) gives a much better point
    estimate, while the quantile heads still provide the calibrated interval.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_chans: int = 5,
        pretrained: bool = True,
        quantiles=(0.05, 0.50, 0.95),
        point_head: bool = True,
    ):
        super().__init__()
        self.quantiles = tuple(quantiles)
        # num_classes=0 -> the backbone returns pooled features, not logits
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.head = MonotoneQuantiles(self.backbone.num_features, len(quantiles))
        self.point_head = nn.Linear(self.backbone.num_features, 1) if point_head else None

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)
        out = {"quantiles": self.head(feats)}
        if self.point_head is not None:
            out["point"] = self.point_head(feats).squeeze(-1)   # (B,)
        return out


def build_model(cfg) -> PM25QuantileNet:
    """Construct the model from the config dict."""
    return PM25QuantileNet(
        backbone=cfg["model"]["backbone"],
        in_chans=cfg["model"]["in_chans"],
        pretrained=cfg["model"]["pretrained"],
        quantiles=tuple(cfg["quantiles"]),
        point_head=cfg["model"].get("point_head", True),
    )


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters (sanity check: EfficientNet-B0 ≈ 5.3 M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
