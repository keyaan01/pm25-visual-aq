# Phase 5 — Training

Companion to `notebooks/05_train.ipynb`. Maps to **paper §3.7 and §3.12**.

## What we train, and against what

The model outputs three numbers (the 5th/50th/95th percentiles). We train each with the
**pinball loss** (`src/losses.py`), which penalises being too-low and too-high by different
amounts depending on the percentile — that asymmetry is what makes a head learn a *specific*
percentile instead of the mean. The three losses are summed.

**We train on `log(AQI)`, not raw AQI.** The label is right-skewed (many moderate days, a
few extreme ones); on the raw scale a handful of extreme days would dominate the loss. Log
compresses that tail. Because log is monotone, it doesn't disturb the ordering guarantee, and
we convert predictions back to AQI at the end.

## The data pipeline (`src/dataset.py`)

Each training example is a `(5, 224, 224)` tensor built from the cached physics maps plus a
freshly-decoded RGB image. **Augmentation is geometric only** — horizontal flips, small
rotations, mild crops — and is applied to **all five channels together** so colour and physics
stay aligned. **Photometric augmentation is excluded entirely**: changing brightness, contrast,
saturation, or hue would alter the very image statistics that encode pollution, corrupting the
signal while leaving the label untouched.

## The loop (`src/train.py`)

Mirrors the paper's implementation details:

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Learning rate | 3×10⁻⁴, cosine-annealed after a 2-epoch warm-up |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | ≤ 40, **early stopping** on calibration loss (patience 6) |
| Precision | mixed (AMP) on GPU |
| Seeds | fixed (reproducible) |

Each epoch trains on `train`, then measures the pinball loss on the held-out `cal` split.
The **best** model (lowest calibration loss) is saved to Drive (`outputs_dir/<strategy>/
best_model.pth`) so it survives Colab sessions and can be loaded by Phase 6 and the demo.

## Why calibration loss (not test) drives early stopping

`test` must stay untouched until the very end, or we'd be tuning to it (a subtle leak). The
`cal` split is the honest signal for "is it still improving?", and it's the same split the
conformal step (Phase 6) uses.

## Local verification

The whole chain (split → cache → loaders → model → 2-epoch mini-run → predictions) is
smoke-tested on the 100-image fixture on CPU: batches are `(B,5,224,224)`, the target is
`log(AQI)`, the loss computes, a checkpoint saves, and predictions convert back to AQI while
staying ascending (q05 ≤ q50 ≤ q95). The **full** run happens on the Colab GPU.

## What's next

Phase 6 (`06_calibrate_evaluate`) turns these ordered-but-uncalibrated outputs into intervals
with a guaranteed ~90% coverage, then reports the full metrics.
