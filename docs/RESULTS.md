# Results ledger

> This file collates every headline number the project has produced, in the order the *story*
> unfolds, so the research write-up can be assembled directly from it. Each result names the code
> that produced it and the paper section it maps to. All numbers in the leakage table come from a
> **single clean run** (same model, same seed, only the split strategy changed), so they are directly
> comparable.
>
> **Units:** the label is a US-EPA **AQI index** (roughly 1–530, a unitless index — *not* µg/m³), so
> every MAE / RMSE / interval width below is in **AQI points**.

---

## 0. The dataset (context)

[`DeadCardassian/PM25Vision`](https://huggingface.co/datasets/DeadCardassian/PM25Vision): street-level
photos each labelled with that location's **daily-average** PM2.5 AQI. After removing 123 duplicate
image ids we use **11,096 photos across 3,259 monitoring stations** (median 2 photos/station, max 91).
A falsification check confirmed the images and labels are correctly paired: mean Dark-Channel-Prior
transmission correlates **negatively** with AQI (Pearson r ≈ −0.14, p ≈ 5e-6) — hazier photo, higher
pollution, exactly as physics predicts. *(Paper §3.2–3.3; code: `src/data.py`, `src/audit.py`.)*

---

## 1. The honest pipeline (what we built — the method)

The model we evaluate throughout: *(Paper §3.5–3.8; code walkthrough in [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).)*

| Piece | What it does | File |
|---|---|---|
| 5-channel input | RGB + Dark-Channel-Prior **transmission** + **inverted-saturation** maps | `src/physics.py` |
| Backbone | EfficientNet-B0 (ImageNet-pretrained, stem widened 3→5 channels) | `src/model.py` |
| Interval heads | 3 monotone quantile heads (5th/50th/95th), non-crossing **by construction** | `src/model.py` |
| Point head | a dedicated **MSE**-trained head on a standardized target → estimates the **mean** (good R²) | `src/model.py`, `src/losses.py` |
| Calibration | Conformalized Quantile Regression → guaranteed ~90% interval coverage | `src/calibrate.py` |
| Recalibration | isotonic pred→truth map fit on the calibration split (removes systematic bias) | `src/recalibrate.py` |
| Split (primary) | **station-grouped**: every station's photos live in exactly one of train/cal/test | `src/splits.py` |

**Why leakage-safe splitting is the whole point:** photos from the *same* station share a
near-identical daily-average label. If such photos land in both train and test, the model can score
well by *recognising the place* rather than *reading the haze*. The station-grouped split forbids that,
so it measures genuine generalisation to **unseen locations**.

---

## 2. Headline honest result — station-grouped (leakage-safe)

*Primary result. Generalisation to monitoring stations never seen in training. (Paper §3.11.)*

| R² | r²(Pearson) | MAE | RMSE | Spearman | coverage | SD(y_test) | n_train | n_test |
|----|-------------|-----|------|----------|----------|------------|---------|--------|
| **0.220** | 0.230 | **54.2** | 95.5 | 0.649 | 0.872 | 108.1 | 7,217 | 2,216 |

**Reading it:** the model **ranks** pollution well (Spearman 0.65) and its 90% intervals are close to
calibrated (0.87). But the point-accuracy R² is modest — and well below the benchmark paper's reported
**0.55**. That gap is the question the rest of this ledger answers.

> **R² vs r²:** we report the strict **coefficient of determination** (`R2`, penalises bias/scale) as
> the headline, *and* the squared-Pearson correlation (`r2_pearson`, association only) because looser
> papers quote the latter. `R2 ≤ r2_pearson` always.

---

## 3. The puzzle, and how we investigated it

The PM25Vision paper reports **EfficientNet-B0 R² = 0.550** (MAE 36.6, RMSE 54.6) on an "80/20 split"
whose station-handling is unspecified. Two checks told us our 0.22 was not a mistake:

1. **Literature.** Honest single-photo, cross-location, daily-average AQI regression sits at
   **R² ≈ 0.1–0.35**. The closest analogue (Mondal 2024) reaches R² ≈ 0.39 on an *easier* setup
   (single city, not station-disjoint). So **0.22 is field-normal and 0.55 is the outlier.**

2. **We reproduced the dataset's own station-disjoint 80/20 ("shipped") split** with our correct
   model → **R² = 0.378** (see table below). Still far from 0.55 — so the gap is **not** our split
   choice and **not** the physics channels.

That left one hypothesis: the benchmark's unspecified split **admits station-level leakage**. So we
measured it directly.

---

## 4. The decisive experiment — the leakage gradient

*Same model, same seed, same everything — only `split.strategy` changes. (Code: `notebooks/09_leakage.ipynb`,
`src/leakage.py`; narrative: [`10_leakage.md`](10_leakage.md).)*

| split | R² | r²(Pearson) | MAE | RMSE | SD(y_test) | Spearman | coverage | n_train | n_test | stations in >1 split |
|---|---|---|---|---|---|---|---|---|---|---|
| **random** (leaky control) | **0.759** | 0.760 | 26.4 | 42.9 | 87.4 | 0.854 | 0.905 | 7,212 | 2,220 | **1,221** |
| temporal | 0.380 | 0.404 | 43.5 | 58.7 | 74.5 | 0.635 | 0.814 | 7,212 | 2,220 | 393 |
| shipped (paper's split geometry) | 0.378 | 0.425 | 44.5 | 64.3 | 81.5 | 0.683 | 0.877 | 6,658 | 2,902 | 0 |
| **station_grouped** (honest) | **0.220** | 0.230 | 54.2 | 95.5 | 108.1 | 0.649 | 0.872 | 7,217 | 2,216 | 0 |
| geographic (hardest) | −0.079 | 0.053 | 59.9 | 104.5 | 100.6 | 0.862 | 0.862 | 7,086 | 2,104 | 0 |

**Headline:**
- **Δ_leak(R²) = R²(random) − R²(station_grouped) = 0.759 − 0.220 = 0.539.**
- **Δ_leak(MAE) = 54.2 − 26.4 = 27.8 AQI** (MAE, unlike R², does not depend on the test-set variance).

**Reading it:**
- The leaky **random** split scores **0.759 — above the paper's 0.55.** So a station-leaky split
  easily produces (and exceeds) the benchmark's number.
- Every **station-disjoint** split is far lower (shipped 0.378, station_grouped 0.220, geographic
  −0.079). The score falls as fewer stations are shared across train/test (the last column).
- We **could not reproduce 0.55 on any station-disjoint split.**

> **Confounder control:** R² depends on the test set's own variance, so `SD(y_test)`, `MAE` and `RMSE`
> are shown alongside every R². The MAE gap (27.8 AQI) tells the same story variance-free.

---

## 5. The causal clincher — contaminated vs clean (no extra training)

*Within the **random** model's own test set, split photos into **contaminated** (their station OR a
perceptual-hash near-duplicate also appears in that model's training set) vs **clean**. (Code:
`leakage.contaminated_vs_clean` + `audit.redundancy_report`.)*

| subset | n | share | R² | MAE |
|---|---|---|---|---|
| **contaminated** (leaked) | 1,757 | 79.1% | **0.762** | 25.4 |
| **clean** (no leakage) | 463 | 20.9% | **−0.86** | 30.4 |

**Reading it:** the random split's high score lives **entirely in the leaked photos**. On genuinely
unseen locations the same model is **worse than predicting the average** (negative R²). This proves the
inflation is *leakage*, not the random test set merely being easier.

> **Honest caveat (kept in the write-up):** the clean subset's R² comes from the *random-trained*
> model on its unseen-station slice, so "clean ≈ station_grouped" is a **qualitative** statement (both
> generalise poorly) — clean is in fact a little worse, because a leakage-trained model generalises
> badly to truly-unseen places. The direction (contaminated ≫ clean) is what carries the argument.

**Conclusion:** the benchmark's 0.55 is fully consistent with station-level leakage; **our honest
0.22 is the real generalisation number.** The project's deliverable is that honest number *plus the
measured leakage gap* — not matching an inflated benchmark.

---

## 6. Correctness verification (2026-09-25)

Before building on these numbers, three independent read-only audit agents read every module in full
and ran numerical spot-checks. **No result-affecting bug in any layer.** Verified: leakage
prediction-to-row ordering is correct (so the contaminated/clean split is meaningful); the physics
cache aligns with the right image; splits are station-disjoint *by construction*; conformal coverage
math is correct (sim 0.9006); the point head is genuinely mean-seeking; and **every row above
reconciles** R² ≈ 1 − (RMSE/SD)² to within 0.0008. Findings were limited to harmless cleanups (since
applied).

---

## Artifacts (for the paper)

- `outputs/leakage_gradient.csv` — the table in §4.
- `outputs/contaminated_vs_clean.json` — the numbers in §5.
- The leakage-gradient figure (bars per split, with the 0.55 line) — from `notebooks/09_leakage.ipynb`.
- Reproduce: `notebooks/09_leakage.ipynb` on a Kaggle GPU (one commit run). Tag the producing commit
  `honest-baseline-v1` so this exact state is citable.

## Still to come (will be appended here, not overwritten)

- **C2 — error ceiling:** how much of the honest residual is irreducible daily-average label noise
  (an upper bound on any model's R²). *(Paper §3.10.)*
- **C3 — abstention:** refuse-to-answer on unusable photos; risk–coverage curves. *(Paper §3.9.)*
- **Bigger model:** a larger backbone is planned; its honest result will be added as a **new row**
  beside the EfficientNet-B0 baseline, so both remain on record.
