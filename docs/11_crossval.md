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

- `summarize_cv(…)` also exposes `ci_half` (the CI half-width) separately from `std`, and
  `nadeau_bengio_ci(…)` gives a corrected-resampled-t interval for repeated CV (see the caveat below).

## The result (5 folds)

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R² | 0.405 | 0.574 | 0.365 | 0.429 | 0.152 |

**Honest R² = 0.385** (SD 0.15; 95% CI [0.196, 0.574]) · MAE 46.3 · Spearman 0.679 · **coverage 0.895**.

Three things this tells us:
1. **The robust honest number is 0.385 — higher than the single split's 0.220**, which turns out to have
   been one hard, high-variance draw (≈ fold 4's 0.152). Cross-validation *raised* our headline.
2. **The conformal intervals are validated:** mean coverage **0.895 ≈ the 0.90 target** across five
   independent folds — the calibration works, not just on one split.
3. **Honest R² is highly split-dependent (0.15–0.57).** A *single* R² on this task is unreliable — which
   is exactly why single-number benchmarks (the paper's 0.55 included) must be distrusted.

**The honest correction this forces:** one fold reached **0.574 > 0.55**, so we retract any earlier
claim that "0.55 is unreachable honestly." 0.55 is *within* the honest range; leakage is proven by the
causal contaminated-vs-clean test and by the leaky 0.759 exceeding the *whole* honest CI — not by 0.55
being impossible.

> **Caveat (honest):** these 5 folds are one deterministic GroupKFold partition, so they are correlated
> (shared training data) and the plain Student-t CI is mildly **optimistic**. A fully rigorous interval
> would repeat grouped CV over several seeds and use the Nadeau–Bengio correction
> (`crossval.nadeau_bengio_ci`). The qualitative story (0.385, split-dependent, validated coverage) is
> unchanged.

## How to run

`notebooks/10_crossval.ipynb` on a Kaggle GPU as **Save & Run All (Commit)** (headless). It trains 5
folds (~5–6 h) and saves `cv_folds.csv` (per-fold metrics) and `cv_summary.json` (the mean ± CI). To
go faster, set `N_SPLITS = 3` in the run cell (~3 h). We use **EfficientNet-B0** (the paper's
strongest backbone on this dataset), so the CV numbers are directly comparable to the single-split
result.
