# Physics-Guided Visual Air-Quality Estimation

## The project explained in full — and what every result means

*A from-zero companion to the results. Read this to understand not just the numbers, but what each
one actually tells us, what it does not, and what to say about it. Generated 2026-09-28.*

---

## 1. What this project is (in one breath)

We predict **air quality from a single street photo** — and we do it **honestly**. Instead of spitting
out one confident number, the system gives a **low–median–high range**, can **refuse to answer** on
unusable photos, estimates **how much error is physically unavoidable**, and is **evaluated in a way
that can't cheat**. The scientific centre of the project is a discovery: the usual way these models are
scored **inflates the score through data leakage**, and we measure exactly how much.

**The one-sentence version:** *estimate air quality from a photo, report it with honest uncertainty,
and prove that a popular benchmark's high score is inflated by data leakage.*

---

## 2. The units rule (read this first, it trips everyone up)

The label we predict is a **US-EPA Air Quality Index (AQI)** value — a unitless index roughly from
**1 to 530**. It is **not** a concentration in micrograms per cubic metre (µg/m³). Every error number
in this document — MAE, RMSE, interval width — is in **AQI points**. (We only touch µg/m³ inside the
error-ceiling analysis, where we convert it to AQI with the official EPA table.)

One more crucial fact about the label: each photo's label is the **daily-average** AQI for that
location. A photo is a single instant. Pollution rises and falls during a day, so the label is a
slightly "smudged" answer key — a fact that matters a lot in Section 8.

---

## 3. How the system works (the pipeline, plainly)

| Piece | What it does, in plain words |
|---|---|
| **5-channel input** | We feed the network the ordinary photo (Red, Green, Blue) **plus two physics maps**: a *transmission map* (how much haze sits between camera and scene, from the Dark Channel Prior) and an *inverted-saturation map* (haze washes colour out; this catches it, especially in sky where the first map fails). |
| **Backbone (EfficientNet-B0)** | A compact, pretrained image network that turns the 5-channel image into features. It is the benchmark's own strongest backbone, so our numbers are comparable. |
| **Interval heads** | Three outputs predicting the 5th, 50th, and 95th percentiles — a low, middle, and high estimate. They are built so they can **never cross** (high ≥ middle ≥ low, always). |
| **Point head** | A separate output trained to hit the **average** (mean), because the accuracy score R² rewards matching the mean. This is what our MAE/RMSE/R² come from. |
| **Conformal calibration** | A post-processing step that **guarantees** the low–high range contains the truth about 90% of the time, for any model. |
| **Isotonic recalibration** | A gentle correction, learned on held-out data, that removes systematic under-prediction of high pollution. |
| **Station-grouped split** | The evaluation rule: all photos from one monitoring station go entirely into *one* of train/calibration/test. This is the anti-cheating measure — Section 6 explains why. |

---

## 4. The measuring sticks — what each metric MEANS

Before any result, here is what each number is actually saying.

- **MAE (Mean Absolute Error).** The average size of the miss, in AQI points. *"On a typical photo we
  are off by about this many AQI points."* Directly interpretable; a MAE of 46 means the typical miss
  is ~46 AQI points. It does **not** depend on how spread out the test set is, which makes it the most
  honest single accuracy number.

- **RMSE (Root Mean Squared Error).** Like MAE but squares the errors first, so it punishes occasional
  big misses more. Always ≥ MAE. Useful next to MAE: if RMSE ≫ MAE, a few large errors dominate.

- **R² (coefficient of determination).** *"What fraction of the variation in the truth does the model
  explain, compared to just always guessing the average?"* **1.0** = perfect, **0** = no better than
  guessing the average, **negative** = worse than guessing the average. It is the strict field-standard
  score. **Key subtlety:** R² depends on how spread out the test set is (its variance), so the *same
  model* can score very different R² on different test sets. This is why we always show MAE and the
  test-set spread next to R².

- **r²(Pearson) (squared correlation).** A looser cousin: it measures only whether the model's ranking
  *lines up* with the truth, ignoring bias and scale. It is always ≥ R². Some papers call this "R²." We
  report **both** so nobody can quote the looser one against us. If someone's "R²" looks suspiciously
  high, they may be reporting this one.

- **Spearman (rank correlation).** *"Does the model put photos in the right order from cleanest to most
  polluted?"* 1.0 = perfect ordering. A high Spearman with a modest R² means the model *sees* pollution
  (ranks it well) but is off on the exact number.

- **Coverage.** For the low–high range: *"How often did the truth actually land inside the range?"* We
  target **0.90**. Coverage is meaningless without width (an infinitely wide range always "covers"), so
  we always report the two together.

- **Interval width.** The average size of the low–high range, in AQI points. Wider = the model is less
  sure.

---

## 5. What every honest-accuracy result means

### 5.1 The headline: cross-validated R² = 0.385 ± 0.19

We evaluated the model on **5 different station-disjoint splits** (5-fold cross-validation) and
averaged:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R² | 0.405 | 0.574 | 0.365 | 0.429 | 0.152 |

**R² = 0.385** (SD across folds 0.15; 95% CI [0.196, 0.574]) · **MAE = 46.3** · Spearman 0.679 ·
**coverage = 0.895**.

**What it means:**
- **0.385** = across genuinely unseen locations, the model explains about **39% of the variation** in
  air quality, versus 0% for guessing the average. For predicting pollution from a *single* photo of a
  *new* place, that is **squarely field-normal** — the closest published analogue (an easier setup)
  reaches ≈0.39.
- **MAE 46.3** = the typical miss is about **46 AQI points**. On the 1–530 scale, and given the model is
  reading an instant while the label is a daily average, this is a real but expected error.
- **The ± is a confidence interval, not a spread.** The "95% CI [0.196, 0.574]" has half-width ≈0.19;
  the plain standard deviation across folds is 0.15 (a different quantity — don't confuse them).

### 5.2 Why the single-split number (0.220) was lower

An earlier single split gave R² = 0.220. Cross-validation showed that split was simply a **hard,
high-variance draw** — it happens to be close to the *worst* of the five folds (0.152). **Read 0.220 as
the pessimistic end; 0.385 is the robust estimate.** Cross-validation *raised* our honest headline.

### 5.3 The most important finding hiding in the folds: 0.15–0.57

The five folds range from **0.152 to 0.574**. That huge swing is itself a result: **a single R² on this
task is unreliable.** Which stations you happen to hold out changes the score dramatically. This is the
deepest reason to distrust any single-number benchmark on this dataset — including the paper's 0.55.

### 5.4 Coverage = 0.895 — the intervals actually work

Across five independent folds, the truth landed inside the 90% range **89.5%** of the time — almost
exactly the 0.90 target. **This validates the honesty machinery**: the calibrated ranges are trustworthy,
not just on one lucky split.

> **Honest caveat.** These 5 folds come from one deterministic partition, so they are *correlated*
> (they share training data). That makes the plain confidence interval a little **optimistic**. A fully
> rigorous version repeats the whole thing over several seeds (Nadeau–Bengio correction). The story —
> 0.385, split-dependent, validated coverage — does not change.

---

## 6. What the leakage results mean (the heart of the project)

**Data leakage** = test information sneaking into training, so the score measures *memorisation*
instead of *understanding*. Here the leak is specific: many photos come from the **same monitoring
station** and therefore share a **near-identical daily-average label**. If such photos land on both
sides of the split, the model can score well by **recognising the place**, not by reading the haze.

### 6.1 The leakage gradient — same model, only the split changes

| split | R² | MAE | SD(y_test) | what it is |
|---|---|---|---|---|
| **random** (leaky control) | **0.759** | 26.4 | 87.4 | photos assigned independently → same-station photos on both sides |
| temporal | 0.380 | 43.5 | 74.5 | train on earlier dates, test on later |
| shipped | 0.378 | 44.5 | 81.5 | the dataset's own 80/20 split (station-disjoint) |
| **station_grouped** (honest) | **0.220** | 54.2 | 108.1 | every station in exactly one split — the honest protocol |
| geographic (hardest) | −0.079 | 59.9 | 100.6 | whole regions held out |

**What it means, row by row:**
- **random = 0.759.** Deliberately leaky. It scores **above the benchmark's 0.55**, and above the
  *entire* honest cross-validated range. A leaky split can manufacture a very high score.
- **temporal / shipped = ~0.38.** Station-disjoint (or nearly), and land near our honest CV mean.
- **station_grouped = 0.220.** The honest single number (the hard draw from Section 5.2).
- **geographic = −0.079.** Holding out whole regions is so hard the model does slightly worse than
  guessing — an honest measure of how badly single-photo AQI transfers across the globe.

### 6.2 The leakage gap (Δ_leak)

- **Lead with the MAE gap: Δ_leak(MAE) = 54.2 − 26.4 = 27.8 AQI.** The leaky split roughly **halves the
  error**. We lead with MAE because, unlike R², it does not depend on the test set's spread — it is the
  clean, variance-free measure of the effect.
- **Δ_leak(R²)** is 0.539 against the single hard draw (0.220), or 0.374 against the robust CV mean
  (0.385). We report the pair rather than only the most dramatic number.

### 6.3 The causal proof — contaminated vs clean

Within the leaky (random) model's own test set, we split photos into **contaminated** (their station or
a near-duplicate is also in training) vs **clean** (genuinely unseen):

| subset | n | R² | MAE |
|---|---|---|---|
| contaminated (leaked) | 1,757 | **0.762** | 25.4 |
| clean (no leakage) | 463 | **−0.86** | 30.4 |

**What it means:** the leaky split's high score lives **entirely in the leaked photos**. On genuinely
unseen photos the same model is worse than guessing. This is the clincher: the inflation is **leakage**,
not the random test set merely being easier.

> **Read the −0.86 correctly — it is variance-amplified, not a catastrophe.** The clean subset is small
> (463 photos) and dominated by low-variance stations, so its label spread is small. Because
> R² = 1 − (RMSE/SD)², a *modest* error on a low-spread set sends R² sharply negative. Look at the MAE:
> **25.4 vs 30.4** — the predictions on clean photos are only *a little* worse, not wild. **Lead with
> the MAE gap and the direction (contaminated ≫ clean); treat −0.86 as qualitative.**

### 6.4 How to talk about the benchmark's 0.55 (honestly)

We do **not** claim the authors faked anything, and we do **not** claim 0.55 is impossible honestly —
one of our honest folds reached **0.574**. What we claim, and prove, is:
1. the data is **dramatically leakage-prone** (a leaky split inflates past 0.55, to 0.759), and
2. single numbers on this task are **unreliable** (honest R² swings 0.15–0.57).

The benchmark's unspecified split *admits* the leakage mechanism; our contribution is to **measure and
causally prove** the inflation, and to report honestly (cross-validation + calibrated intervals + an
error ceiling).

---

## 7. What the error-ceiling (C2) result means

Because each label is a **daily average** but each photo is an **instant**, even a *perfect* photo-reader
would disagree with the label by the day's within-day swing. That puts a hard **upper bound** on
achievable R²: `R²_max = 1 − Var(ε)/Var(y)`, where Var(ε) is the within-day noise.

**The result: R²_max ≈ 0.98** — measured on 32 OpenAQ stations across 6 countries (1,241 station-days):
0.982 (optimistic) to **0.979** (conservative, imputing the still-uncovered top bands), and ~0.98 under
both the 2012 and 2024 EPA breakpoint tables. Report it as **~0.98, an upper bound** — not a single
decimal, and not with a tight confidence interval (the cluster-bootstrap band [0.970, 0.993] comes from
only 32 stations). An earlier 13-station pass gave a looser 0.977, and we worried the true value might
fall to ~0.90–0.95 once the most-polluted bands were covered — the better-sampled re-run shows it does
**not**: even imputing the uncovered top ~13% (200+ AQI) with a high within-day variance, the ceiling
only moves to 0.979.

**What it means for the project:** label noise is only **~2% of the label variance** (Var(ε) ≈ 208–248 vs
Var(y) ≈ 11,708). So daily-average labeling costs at most ~2 points of R², and our honest 0.22–0.39 is
limited by **how hard it is to read pollution from one photo**, not by noisy labels. The ceiling's *job*
was to answer "is the gap difficulty or bad labels?" — and the answer is emphatically **difficulty**.

---

## 8. Were the numbers checked? Two independent audits

**Audit 1 (2026-09-25)** and **Audit 2 (2026-09-28)** each used three independent read-only agents to
re-read every module and re-check the math. Combined verdict:

- **The accuracy numbers (0.220 single, 0.385 ± 0.19 CV), the leakage gradient, and the causal
  contaminated/clean result are computed correctly and honestly — no result-affecting bug.**
  Predictions line up with the right photos and labels; splits are station-disjoint by construction;
  the coverage guarantee is implemented correctly; the point head genuinely targets the mean; and every
  table row reconciles with the identity R² ≈ 1 − (RMSE/SD)².
- **The one over-statement was the first-run C2** (a loose 0.977 from 13 stations); the proper re-run
  gives R²_max ≈ 0.98 (Section 7).
- Everything else was reporting polish, now applied (leading with the MAE gap, flagging the −0.86 as
  variance-amplified, disambiguating the confidence interval, noting the correlated-fold caveat), plus
  defensive safety checks and tests added to the code.

---

## 9. What to conclude — and what NOT to conclude

**Conclude:**
- Honest, robust accuracy is **R² ≈ 0.385** with **validated 90% intervals** and MAE ≈ 46 AQI.
- The data is **dramatically leakage-prone**: a leaky split inflates to 0.759, proven causally.
- The honest score is limited by the **visual task**, not by label noise (the ceiling).
- Single-number scores on this task are **unreliable** (0.15–0.57 across honest folds).

**Do NOT conclude:**
- ✗ "0.55 is impossible / the authors cheated." (One honest fold hit 0.574; their split is merely
  unspecified and *admits* leakage.)
- ✗ "Clean photos give R² = −0.86, so the model is broken." (That −0.86 is variance-amplified; the MAE
  gap is modest.)
- ✗ "The ceiling is exactly 0.98 / 0.977." (It is an upper bound ~0.98 with the top ~13% extrapolated;
  report it as ~0.98, not a single decimal, and not with a tight CI.)
- ✗ "R² alone tells the story." (Always read it beside MAE and the test-set spread.)

---

## 10. What still needs running (short list)

Done: core pipeline, leakage, cross-validation, and the **C2 error ceiling** (≈0.98, Section 7). C3
abstention and the demo are **built and locally tested** — they just need one run each:

1. **`08_abstention.ipynb`** on Kaggle — attach ExDark (night) / DTD (textures) / MIT-Indoor (indoor),
   run it (uses the trained model, no retrain). Produces the risk–coverage/AURC + OOD tables and saves
   the demo bundle. *This is the one worth doing next.*
2. **Deploy the demo** — download the bundle, then follow `docs/11_demo.md` to create the free Hugging
   Face Space (Gradio) and go live at a permanent URL.
3. **(Optional) `10_crossval.ipynb` with `ACCURACY_PUSH = True`** — ensemble + EMA + TTA to try to raise
   R² (~15 h on a GPU); reported as a *new row* beside the baseline. Skip unless you want a higher number.

---

## 11. Mini-glossary

- **AQI** — Air Quality Index, a unitless 1–530 index (what we predict). Not µg/m³.
- **Leakage** — test info reaching training, so the score measures memorisation, not understanding.
- **Station-grouped split** — all of one station's photos go to one split; the honest protocol.
- **Cross-validation (GroupKFold)** — rotate which fifth of the *stations* is the test set, five times;
  report mean ± CI.
- **R² / MAE / RMSE / Spearman / coverage** — see Section 4.
- **Conformal calibration** — the procedure that guarantees ~90% interval coverage for any model.
- **Δ_leak** — how much a leaky split inflates the score over the honest one.
- **R²_max (error ceiling)** — the best R² any model could reach given the daily-average labels (an
  *upper bound*, measured ≈0.98).
