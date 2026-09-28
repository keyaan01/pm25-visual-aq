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

## 2. Honest result — station-grouped (leakage-safe), single split

*Generalisation to monitoring stations never seen in training. This is **one** split; the robust,
cross-validated estimate is in §4b. (Paper §3.11.)*

| R² | r²(Pearson) | MAE | RMSE | Spearman | coverage | SD(y_test) | n_train | n_test |
|----|-------------|-----|------|----------|----------|------------|---------|--------|
| 0.220 | 0.230 | 54.2 | 95.5 | 0.649 | 0.872 | 108.1 | 7,217 | 2,216 |

**Reading it:** the model **ranks** pollution well (Spearman 0.65) and its 90% intervals are close to
calibrated (0.87). Crucially, **this split was a hard, high-variance draw** — its test-set SD (108.1)
is the highest of any split we ran, and R² depends on the test set's spread. Five-fold cross-validation
(§4b) puts the honest R² at **0.385 ± 0.19**, so read 0.220 as the *pessimistic end* of the honest
range, not the single true value.

> **R² vs r²:** we report the strict **coefficient of determination** (`R2`, penalises bias/scale) as
> the headline, *and* the squared-Pearson correlation (`r2_pearson`, association only) because looser
> papers quote the latter. `R2 ≤ r2_pearson` always.

---

## 3. The puzzle, and how we investigated it

The PM25Vision paper reports **EfficientNet-B0 R² = 0.550** (MAE 36.6, RMSE 54.6) on an "80/20 split"
whose station-handling is unspecified. Two checks told us our 0.22 was not a mistake:

1. **Literature.** Honest single-photo, cross-location, daily-average AQI regression sits at
   **R² ≈ 0.1–0.35**; the closest analogue (Mondal 2024) reaches R² ≈ 0.39 on an *easier* setup
   (single city, not station-disjoint). Our cross-validated honest **R² ≈ 0.385 (§4b) matches that
   analogue** — squarely field-normal. So the benchmark's 0.55 sits at the *optimistic edge* of what
   honest evaluation gives, not in a different league.

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
- **Δ_leak(MAE) = 54.2 − 26.4 = 27.8 AQI** — the **primary, variance-free** effect size (MAE, unlike
  R², does not depend on the test set's spread). The leaky split roughly *halves* the error.
- **Δ_leak(R²):** against the single hard draw it is 0.759 − 0.220 = **0.539**; against the robust
  cross-validated honest mean (§4b) it is 0.759 − 0.385 = **0.374**. Both are large; we report the pair
  rather than only the most dramatic one, because the single-split 0.220 is the pessimistic end.

**Reading it:**
- The leaky **random** split scores **0.759 — above the paper's 0.55**, and above the *entire* honest
  cross-validated range (§4b; upper 95% CI ≈ 0.574). So a station-leaky split inflates beyond what
  honesty can reach.
- The station-disjoint single splits (shipped 0.378, station_grouped 0.220, geographic −0.079) look far
  lower — but these are **single draws**, and §4b shows honest R² swings 0.15–0.57 depending on which
  stations are held out, so no single one of them is "the" honest number.
- **Honest correction:** we do **not** claim 0.55 is unreachable honestly — cross-validation (§4b) found
  an honest fold at **0.574 > 0.55**. The leakage evidence is the *random* split exceeding the honest CI
  and the causal test below (§5), **not** the impossibility of 0.55.

> **Confounder control:** R² depends on the test set's own variance, so `SD(y_test)`, `MAE` and `RMSE`
> are shown alongside every R². The MAE gap (27.8 AQI) tells the same story variance-free.

---

## 4b. Cross-validation — how robust is the honest number?

*Leakage-safe **5-fold GroupKFold on `station_id`** — every station tested exactly once, never in its
own fold's training; each fold station-disjoint (straddling = 0). Same model/config as the single split.
(Code: `src/crossval.py`, `notebooks/10_crossval.ipynb`; narrative: [`11_crossval.md`](11_crossval.md).)*

| fold | R² | r²(Pearson) | MAE | RMSE | Spearman | coverage | SD(y_test) |
|---|---|---|---|---|---|---|---|
| 0 | 0.405 | 0.444 | 43.3 | 57.3 | 0.679 | 0.838 | 74.3 |
| 1 | 0.574 | 0.587 | 45.0 | 64.5 | 0.751 | 0.926 | 98.8 |
| 2 | 0.365 | 0.372 | 43.1 | 62.7 | 0.628 | 0.856 | 78.7 |
| 3 | 0.429 | 0.491 | 44.5 | 59.3 | 0.718 | 0.949 | 78.5 |
| 4 | 0.152 | 0.160 | 55.7 | 91.1 | 0.619 | 0.908 | 98.9 |

**Across the 5 folds:** **R² = 0.385** (SD 0.15; 95% CI [0.196, 0.574]) · MAE = 46.3 (SD 5.3) ·
Spearman = 0.679 (SD 0.06) · **coverage = 0.895 (SD 0.05)**.

> **Reading the ±:** the R² "95% CI [0.196, 0.574]" is a Student-t interval with **half-width ≈ 0.19**
> (the across-fold **SD is 0.15** — a different quantity, don't confuse them). For MAE/Spearman/coverage
> we quote the **SD** across folds. **Caveat:** these 5 folds come from a *single* deterministic
> GroupKFold partition, so they are **correlated** (they share training data); the plain Student-t CI is
> therefore mildly **optimistic** (anti-conservative). A fully rigorous interval would use repeated
> grouped CV over several seeds with the Nadeau–Bengio corrected-resampled-t
> (`crossval.nadeau_bengio_ci`); the qualitative conclusion is unchanged.

**Three findings:**
1. **The robust honest number is R² = 0.385 ± 0.19**, MAE 46.3 — *higher* than the single-split 0.220,
   which was one hard, high-variance draw (≈ fold 4's 0.152).
2. **The conformal intervals are validated:** mean coverage **0.895 ≈ the 0.90 target** across
   independent folds — strong evidence the calibration works, not just on one split.
3. **Honest R² is highly split-dependent (0.15–0.57).** This is itself a result: a *single* R² on this
   task is unreliable — which is exactly why single-number benchmarks (the paper's 0.55 included) must
   be distrusted, and why we report a CI, calibrated intervals, and an error ceiling.

> **The honest correction this forces:** one honest fold reached **0.574 > 0.55**, so we retract any
> claim that "0.55 is unreachable honestly." 0.55 is within the honest range. The leakage is proven by
> §5 (the causal test) and by the leaky 0.759 exceeding the whole honest CI — not by 0.55 being
> impossible.

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
inflation is *leakage*, not the random test set merely being easier. **Lead with the MAE gap (25.4 vs
30.4):** it is the honest, variance-free effect size and shows the accuracy difference is real but
*modest in absolute terms* — the drama is in R², for the reason below.

> **Honest caveat (kept in the write-up):** the −0.86 is **variance-amplified, not a catastrophic
> prediction failure.** The clean subset is small (n = 463) and dominated by single-image, low-variance
> stations, so its label spread SD(y) is small; since R² = 1 − (RMSE/SD)², a *modest* RMSE on a
> low-variance set drives R² sharply negative. The MAE (30.4) confirms the predictions are not wild —
> only a little worse than on contaminated photos. So treat clean R² = −0.86 as **qualitative** (the
> model generalises poorly to truly-unseen places), and let the **MAE gap + the direction
> (contaminated ≫ clean)** carry the argument, not the −0.86 magnitude. *(The re-run reports each
> subset's SD_y and RMSE, added in `leakage.contaminated_vs_clean`, to make this explicit.)*

**Conclusion (two honest findings):**
1. **The data is dramatically leakage-prone.** A leaky split inflates to **0.759** — above the entire
   honest range — and the causal test proves the inflation is the *leaked photos*, not an easier test set.
2. **Honest, cross-validated performance is R² = 0.385 ± 0.19** (§4b), and single splits swing 0.15–0.57,
   so *any* single number on this task — including the benchmark's 0.55 — is unreliable on its own.

The deliverable is therefore honest, robust reporting (cross-validation + calibrated intervals + an
error ceiling) **plus** a measured, causally-proven leakage effect — **not** a claim that a specific
published number was faked (0.55 is within the honest range; we don't accuse, we measure).

---

## 6. Correctness verification (two independent audits)

**Audit 1 (2026-09-25).** Three independent read-only agents read every module in full and ran
numerical spot-checks. **No result-affecting bug in any layer.** Verified: leakage prediction-to-row
ordering is correct (so the contaminated/clean split is meaningful); the physics cache aligns with the
right image; splits are station-disjoint *by construction*; conformal coverage math is correct (sim
0.9006); the point head is genuinely mean-seeking; and **every row above reconciles** R² ≈ 1 − (RMSE/SD)²
to within 0.0008. Findings were limited to harmless cleanups (since applied).

**Audit 2 (2026-09-28) — re-verification after cross-validation + C2.** Three more independent
read-only agents re-checked everything that feeds a reported number. Verdict: **the R² (0.220 single,
0.385 ± 0.19 CV), the leakage gradient, and the causal contaminated/clean result are all computed
correctly and honestly — no result-affecting bug.** The one over-statement they found was in **C2**: the
first-run R²_max ≈ 0.977 is a *loose, optimistically-biased* upper bound (see "Provisional" below), not
a precise figure — the *math* is right but the sample was too small and skewed low-AQI. Everything else
was reporting/robustness polish, now applied: the ±-convention is disambiguated (§4b), the leakage gap
is reported at both endpoints and led by the variance-free MAE (§4), the −0.86 is flagged as
variance-amplified (§5), the CV CI is noted as mildly optimistic on correlated folds (§4b), one notebook
that skipped the isotonic step was fixed to match the ledger, and defensive asserts + tests were added
(end-to-end label↔prediction alignment, shipped station-disjointness, physics-cache alignment).

---

## 7. The error ceiling (C2) — how much error is unavoidable

*How good could ANY model be, given the labels are **daily averages** but photos are **instants**?
`R²_max = 1 − Var(ε)/Var(y)`, an UPPER bound from label noise only. (Paper §3.10; code `src/ceiling.py`;
narrative [`08_error_ceiling.md`](08_error_ceiling.md).) Reference: hourly PM2.5 from **32 OpenAQ
stations across 6 countries, 1,241 station-days.*

- **Var(ε)** (within-day label noise) = **208.6** optimistic / **247.9** conservative (SD ≈ 14–16 AQI).
- **Var(y)** (station-grouped test spread) = **11,708** (SD ≈ 108).
- **R²_max ≈ 0.98** — **0.982** (optimistic) … **0.979** (conservative, imputing the uncovered high
  bands); breakpoint-insensitive (historical 0.982 / 2024 revision 0.980). Cluster-bootstrap band
  [0.970, 0.993] from 32 stations — indicative, not a tight CI.
- **Interval-width floor ≈ 35 AQI** — no honest 90% interval should be narrower than the pollution's own
  within-day spread.

**v(m) — within-day AQI variance by pollution band** (the error-by-band story):

| band (AQI) | station-days | v (variance) | within-day SD | dataset weight |
|---|---|---|---|---|
| 0–50 | 715 | 129.4 | 11.4 | 0.245 |
| 50–100 | 430 | 104.7 | 10.2 | 0.255 |
| 100–150 | 86 | 501.8 | 22.4 | 0.246 |
| 150–200 | 10 | 104.9 | 10.2 | 0.254 |

**What it means:** label noise is only **~2% of the label variance**, so daily-average labeling costs at
most ~2 points of R². The honest **0.22 (single) / 0.385 (CV)** sits far below the ~0.98 ceiling — so the
gap is the **difficulty of reading pollution from one photo (the visual task), not noisy labels.** This
is exactly what C2 was built to establish.

> **Honest caveat.** The reference still under-covers the very-high-AQI bands (200+, ≈13% of label mass,
> and the 150–200 band has only n=10), which are imputed with a **conservative (high)** within-day
> variance. Even so the ceiling only drops from 0.982 to **0.979**, so the conclusion is robust. (An
> earlier 13-station pass gave a looser 0.977 and we feared the true value might fall to ~0.90–0.95;
> this 32-station / 6-country run shows it does not — it holds at ~0.98.)

---

## Artifacts (for the paper)

- `outputs/leakage_gradient.csv` — the table in §4.
- `outputs/contaminated_vs_clean.json` — the numbers in §5.
- `outputs/cv_folds.csv` + `outputs/cv_summary.json` — the cross-validation in §4b.
- The leakage-gradient figure (bars per split, with the 0.55 line) — from `notebooks/09_leakage.ipynb`.
- Reproduce: `09_leakage.ipynb` (gradient) and `10_crossval.ipynb` (CV) on a Kaggle GPU (commit runs).

## Provisional / still to come

- **C2 — error ceiling: ✅ done, see §7.** Measured on 32 stations / 6 countries / 1,241 station-days:
  **R²_max ≈ 0.98** (0.979 conservative – 0.982 optimistic, breakpoint-insensitive). Only the top ~13%
  of the label distribution (200+ AQI) is still extrapolated (conservatively); an optional wider-coverage
  re-run would refine that without changing the ~0.98 conclusion.
- **C3 — abstention (built + smoke-tested; pending one Kaggle run):** two gates — an **OOD gate**
  (shrinkage-Mahalanobis distance in backbone-feature space, + interpretable brightness/detail/sky
  checks) checked first, then an **uncertainty gate** (log-space interval width); both thresholds fit on
  calibration only (5% false-refusal budget). `notebooks/08_abstention.ipynb` reports selective
  **risk–coverage/AURC** (+ metrics at 90/80/70% coverage) and **OOD AUROC/AUPR/FPR@95** against
  ExDark (night) / DTD (textures) / MIT-Indoor (indoor), near- vs far-OOD, and a refuse-rate table.
  Numbers land here after the run. *(Code: `src/abstain.py`, `train.collect_features`; narrative:
  [`09_abstention.md`](09_abstention.md); §3.9.)*
- **Live demo (built):** `src/inference.py` + `app/app.py` (Gradio) — upload a photo → AQI, honest
  interval, EPA category, an **answer/abstain badge**, and the physics maps. Deploys as a permanent
  Hugging Face Space; steps in [`11_demo.md`](11_demo.md). *(Phase 10.)*
- **Bigger model:** a larger backbone can be added as a **new row** beside EfficientNet-B0 (both on
  record). Backbone search was declined; B0 is the paper's strongest on this dataset.
