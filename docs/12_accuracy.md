# Maximizing accuracy (honestly)

How we push predictive accuracy as high as the data allows **without cheating** — keeping the
leakage-safe station-grouped split, the calibrated intervals, and the abstention gate intact.

## The starting point

The first honest run: **R²=0.153, MAE=55.5, Spearman=0.715**. The high Spearman means the model
already *ranks* pollution well — it "sees" haze. The low R² is mostly about *how we read the
output* and *which photos it sees*, not a failure to learn. So the fixes are cheap and reliable.

## Why R² was low (the diagnosis)

1. **We reported the log-median.** The median of a right-skewed distribution sits *below* the mean,
   and R² rewards matching the conditional **mean**. So reporting the median structurally
   understates R². **This is the biggest single cause.**
2. **Class imbalance.** Extreme-pollution photos are the rarest slice, so the model rarely saw them
   and refused to predict high values (MAE 257 in the 300+ band).
3. Possible under-training.

## The fix (Stage A) — reliable, high-impact

- **A standardized point head for the mean.** A separate output, trained with **Huber loss** on the
  **standardized** target `(AQI − mean) / std`. Standardizing is the key reliability choice: a fresh
  head starts by predicting the dataset **mean** (a good starting point) and there is no unstable
  log→AQI retransformation. The train-set mean/std are stored *in the model* (as buffers) so they
  travel with the checkpoint. Accuracy (MAE/RMSE/R²) is reported from this head; the interval still
  comes from the quantile heads.
- **Class-balanced sampling** (train only): a weighted sampler oversamples rare high-AQI bands, so
  the extremes are actually learned. Calibration and test stay at the natural distribution, so the
  coverage guarantee and the reported numbers remain honest.
- **Regularization + convergence:** stochastic depth (`drop_path_rate`), weight decay, geometric-only
  augmentation (no photometric — it would corrupt the haze signal), 60 epochs with patience 10,
  **early stopping on calibration point-MAE** (the metric we care about).

## The rigor protocol (so the number can't fool us)

- Fix the **station-grouped** split and the seed; build the physics cache once.
- **Select every hyperparameter on the calibration split.** The **test set is evaluated once**, at
  the end. Conformal `Q` comes from calibration; nothing is tuned on test.
- Report the **single-model** number (the honest result) and, later, the ensemble number.

## If we need more (Stages B & C)

- **Stage B — capacity + inference:** try stronger backbones (`efficientnet_b2`,
  `tf_efficientnetv2_s`, `convnext_tiny`) but *measure* on calibration — the PM25Vision paper found
  EfficientNet-B0 beat ResNet50 and ViT on ~7k images, so bigger is not automatically better. Add
  weight EMA and test-time augmentation (config flags `ema`, `tta`).
- **Stage C — ensemble:** train the best config with a few seeds (`ensemble_seeds`) and average —
  the most reliable way to gain the last couple of points.

## The honest ceiling

Accuracy is ultimately capped by **C2** (`docs/08_error_ceiling.md`): our labels are daily
averages, so no model reading an instantaneous photo can exceed `R²_max`. We report that ceiling
alongside the accuracy, so "how good is good enough" has a principled answer — and we never inflate
by switching to the leaky random split.
