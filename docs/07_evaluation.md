# Phase 6b — Evaluation

Companion to the second half of `notebooks/06_calibrate_evaluate.ipynb`. Maps to **paper §3.11**.
Everything is reported in **AQI points**.

> **Phase 5b note.** Accuracy metrics (MAE/RMSE/R²) are computed from the **smeared point head**
> (a Huber-trained estimate corrected for the log→AQI bias), not the median quantile — the median
> is biased low on skewed data and understates R². Interval metrics (coverage/width) still come
> from the conformal-calibrated quantiles. `Mx.report(point, intervals, y)` combines the two.

## What we measure (`src/metrics.py`)

**Point accuracy of the median:**
- **MAE** — average absolute error, in AQI points (directly interpretable).
- **RMSE** — like MAE but squares errors first, so it punishes occasional big misses more.
- **R²** — the fraction of the variation in AQI the model explains (1 = perfect, 0 = no better
  than always guessing the average). The published baseline is R² = 0.550.
- **Spearman** — rank correlation: does the model order photos by pollution correctly, even if
  the absolute numbers are off?

**Interval quality:**
- **coverage** — how often the truth actually landed inside the interval (target 0.90).
- **mean width** — the average interval size. Coverage is meaningless without width — an
  infinitely wide interval always "covers", so we always report the two together.

**Two extra views:**
- **error by magnitude** — MAE within each true-AQI band. The visual signal is weak at low
  pollution (clear skies look similar), so aggregate numbers hide where the model struggles;
  this breakdown shows it.
- **EPA-category accuracy** — accuracy after mapping predictions onto the six EPA categories,
  to compare with the classification literature.

## The leakage-gap comparison

The notebook can evaluate **every split strategy you trained** (each saved under
`outputs_dir/<strategy>/`) and tabulate them side by side. Expect `random` to look *better*
than `station_grouped` — that gap is leakage inflation, **measured, not assumed**. Because our
audit found low near-duplicate redundancy (distinct-ratio 0.987), we expect the gap to be
modest, which is itself an honest, reportable result. Results are saved to
`outputs_dir/results_by_split.csv` for the paper.

## Reading the plots

- **Error by pollution level** — bars of MAE per AQI band.
- **Interval widths** — the distribution of calibrated interval sizes (wider = the model is
  less sure; often wider at low AQI).
- **Predictions vs truth** — for a sample of test photos, the predicted median (dot), its 90%
  interval (bar), and the truth (✕). Roughly 90% of the ✕'s should sit inside their bars.

## What "done" means here

Completing this notebook means the **core pipeline is finished**: leakage-safe evaluation, a
physics-guided model, and calibrated intervals with a real ~90% guarantee. The remaining
phases add the two extra contributions (C2 error ceiling, C3 abstention) and the demo. Paste
the numbers back and they'll be written into `docs/RESULTS.md`.
