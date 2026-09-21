# Phase 4 — The model

Companion to `notebooks/04_model.ipynb`. Maps to **paper §3.6**.

## Backbone: EfficientNet-B0

A **backbone** is the shared trunk of a CNN that turns an image into a compact vector of
features; small **heads** on top turn those features into the actual output. We use
**EfficientNet-B0**, pretrained on ImageNet, for three reasons the paper gives:

- **Small** (~5.3 M params with its classifier) — trains within our compute budget.
- **Pretrained** — it already "knows" edges, textures, and shapes, which matters given we
  only have ~11 k images.
- **Comparable** — it's the strongest published baseline on PM25Vision (R²=0.550), so our
  numbers sit next to a known reference point.

`timm.create_model("efficientnet_b0", pretrained=True, in_chans=5, num_classes=0)` does two
things for us: `num_classes=0` drops the 1000-way ImageNet classifier so the backbone returns
pooled features, and `in_chans=5` **widens the first convolution from 3 to 5 channels**,
seeding the two new channels (transmission, inverted saturation) from the pretrained RGB
weights. (The trainable feature-extractor is ~4.0 M params; the original "5.3 M" includes the
discarded classifier.)

## Head: three quantiles that can't cross (paper Eq. 1)

We predict three numbers — the 5th, 50th, and 95th percentiles of the (log) AQI. Predicted
independently, nothing stops the model from putting the 95th *below* the 50th, which is
meaningless (a wider band must contain the narrower one). We forbid this **by construction**:

```
q05 = base                         # unconstrained
q50 = q05 + softplus(step1)        # softplus(...) is always > 0
q95 = q50 + softplus(step2)        # so each bound only ever goes UP
```

`MonotoneQuantiles` implements exactly this (a linear layer producing one base value plus
positive increments, accumulated with `cumsum`). Because every increment is strictly positive,
`q05 ≤ q50 ≤ q95` for **any** input — no extra loss term, no post-hoc sorting. The notebook
verifies this on random inputs.

The outputs live in **log(AQI)** space (we train on the log target, Phase 5). Exponentiating
back to AQI is monotone, so the ordering is preserved.

**Plus a point head (Phase 5b).** Alongside the three quantiles, the model has a 4th output — a
plain linear **point head** trained with Huber loss — used for the accuracy numbers (MAE/RMSE/R²).
`forward` returns a dict `{"quantiles", "point"}`. The quantiles give the calibrated interval; the
point head gives the best single estimate. See `docs/05_training.md`.

## What's next

Phase 5 (`05_train`) trains these heads with the **pinball loss** so each one learns its
percentile, on the log target, with physics-safe augmentation only.
