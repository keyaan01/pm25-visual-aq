# Phase 6a — Conformal calibration

Companion to the first half of `notebooks/06_calibrate_evaluate.ipynb`. Maps to **paper §3.8**.

## The problem

After training, the model gives a low/median/high range — but nothing yet *guarantees* the
truth lands inside it 90% of the time. The model might be over-confident (intervals too
narrow) or under-confident (too wide). We need a guarantee.

## Conformalized Quantile Regression (Romano, Patterson & Candès, 2019)

The fix needs only the held-out **calibration** split and a sorting step:

1. For each calibration example, record how far the truth fell *outside* the predicted range:
   `E_i = max(q05(x_i) − y_i, y_i − q95(x_i))`. It's negative when the truth was comfortably
   inside, and positive by exactly the size of the miss when it was outside.
2. Take **Q** = the 90% mark of those `E_i` (with a small finite-sample correction:
   the `ceil((n+1)·0.9)/n` empirical quantile).
3. Widen every future interval by **Q** at both ends, clipping the lower end at 0 (AQI can't
   be negative). `apply_conformal` does exactly this.

`src/calibrate.py` implements `conformal_Q`, `apply_conformal`, `coverage`, and `mean_width`.

## Why this is powerful

The guarantee holds for **any** model, however good or bad, as long as the calibration and
test data come from the same distribution. A weak model doesn't break it — it just produces
**honestly wide** intervals. We proved this on a synthetic set where the raw intervals covered
only **19%** of truths; after calibration, coverage was **90.2%**, exactly on target (the
width grew from 10 to 64 AQI to earn it). We reuse this check on the real test set.

## Two honest caveats (from the paper)

- **Calibrate in AQI units, not log units.** We train on `log(AQI)` but convert predictions
  back to AQI *before* calibrating, so the widening is symmetric in the units people read,
  avoiding lopsided intervals from exponentiating a symmetric correction.
- **The guarantee assumes exchangeability.** The **geographic** hold-out split deliberately
  breaks that (test regions differ from calibration), so there we report coverage as a
  *measurement of how the guarantee degrades under distribution shift*, not as a confirmation
  that it holds.

## Next

Phase 6b (`07_evaluation`) reports the full metrics on these calibrated intervals.
