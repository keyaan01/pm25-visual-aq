# Cross-validation — is the honest R² robust?

*(Companion to `notebooks/10_crossval.ipynb` and `src/crossval.py`.)*

## The question

Our headline honest number, **R² = 0.22**, comes from **one** train/cal/test split. A fair worry:
*was that split lucky (or unlucky)?* Cross-validation answers it by re-running the evaluation on
several different splits and reporting the spread.

## Why plain k-fold would be wrong here

Standard k-fold cross-validation shuffles all rows and rotates which fifth is the test set. On this
dataset that **leaks**: photos from the same monitoring station (which share a near-identical
daily-average label) would land in both train and test — the exact flaw the project exposes. A
k-fold number would be inflated and meaningless.

## What we do instead: leakage-safe *grouped* cross-validation

We use **`GroupKFold` on `station_id`**: the stations are divided into 5 groups, and each fold uses
one group as test and the other four for training — so **every station is tested exactly once, and
never appears in its own fold's training data**. Within each fold's training stations we carve a
station-grouped **calibration** set (for the conformal intervals), giving the usual
**0.65 / 0.15 / 0.20** train / cal / test per fold. Each fold is therefore leakage-safe by
construction (its `straddling` count is 0).

`src/crossval.py`:
- `make_cv_folds(df, n_splits=5, …)` builds the 5 fold assignments (reusing `splits._group_split`
  for the cal carve).
- `run_cv(…)` trains + evaluates each fold by reusing `leakage.train_and_evaluate` — so every fold is
  set up **identically** to the headline single-split run; only the fold changes.
- `summarize_cv(…)` reports each metric as **mean, std, and a 95% Student-t confidence interval**
  across the folds.

## What you get

Instead of a single "R² = 0.22", you get **"R² = 0.22 ± CI across 5 station-disjoint folds"** — plus
the same for MAE, RMSE, Spearman, and interval coverage. That's the robustness statement a reviewer
or grader looks for, and it confirms the honest number isn't an artifact of one particular split.

## How to run

`notebooks/10_crossval.ipynb` on a Kaggle GPU as **Save & Run All (Commit)** (headless). It trains 5
folds (~5–6 h) and saves `cv_folds.csv` (per-fold metrics) and `cv_summary.json` (the mean ± CI). To
go faster, set `N_SPLITS = 3` in the run cell (~3 h). We use **EfficientNet-B0** (the paper's
strongest backbone on this dataset), so the CV numbers are directly comparable to the single-split
result.
