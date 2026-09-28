# Physics-Guided Visual Air-Quality Estimation

## The complete project, explained from the ground up

*A sequential, no-prior-knowledge-assumed walkthrough of the entire project — every concept, every
equation, every number, and how each piece builds on the last. Written so any team member can present
and defend the whole thing to a professor without a single gap. Read it top to bottom; each section is
set up by the one before it.*

*Team: Keyaan · Anup · Lamiya · Asif. Generated 2026-09-28.*

---

## How to use this document

- **Read in order the first time.** The story is a chain: metrics set up leakage, leakage sets up the
  honest model, the model sets up calibration, and so on. Skipping ahead breaks the logic.
- **Three passes.** Each concept is explained (1) with an everyday intuition, (2) with a precise
  definition/equation, and (3) with *how this project uses it and the exact numbers*. If you only have a
  few minutes, read the **bold sentences** and the **grey callout boxes**.
- **Boxes.** A box beginning **"Worked example"** walks through real arithmetic. A box beginning
  **"Key idea"** is the one sentence to memorise. A box beginning **"Trap"** is a mistake to avoid (and a
  likely professor probe).
- **The defense-ready summary** is Section 12 (five-minute story + ~50-question Q&A + traps). The exact
  numbers are collected in the Appendix.

---

## 1. Orientation — what we built, and why it's unusual

### 1.1 The one-sentence thesis

We estimate air quality (a US-EPA **AQI** value) from a single street photo; we do it **honestly** (a
calibrated range, a refuse-to-answer option, and a physical error ceiling); and along the way we **prove
that a popular benchmark's high accuracy is inflated by data leakage.**

Most "predict X from a photo" projects chase one headline accuracy number and stop. Ours is deliberately
different: **the contribution is honesty and measurement**, not a bigger number. That is the single most
important framing to have ready, because the natural first question — *"but your accuracy is lower than
the benchmark, so isn't your project worse?"* — is answered by this framing (Section 4).

### 1.2 The four honest behaviours (= the four contributions)

- **C1 — a calibrated range, not a single number.** Instead of "AQI = 137", we output a low–median–high
  interval with a *mathematical guarantee* that the truth falls inside about 90% of the time.
- **C2 — an error ceiling.** We estimate the best R² that *any* model could possibly achieve on this
  data, so we know how much of the remaining error is genuinely unbeatable versus fixable.
- **C3 — abstention.** The system **refuses to answer** on photos it should not guess on (night or
  featureless; indoor is a documented limitation of the deployed station-invariant gate), and we measure
  how well it refuses.
- **Leakage-safe evaluation.** We evaluate in a way that cannot cheat, and we *measure* how much a common
  shortcut inflates the score — the project's headline scientific finding.

### 1.3 The units rule (memorise this — it trips everyone up)

The label we predict is a **US-EPA Air Quality Index (AQI)** value: a unitless index that runs roughly
**1–530**. It is **not** a concentration in micrograms per cubic metre (µg/m³). Every error number in
this document — MAE, RMSE, interval width — is in **AQI points**. We touch µg/m³ only inside the
error-ceiling analysis (C2), where we convert it to AQI with the official EPA table.

> **Key idea.** The output is an *index* (AQI 1–530), not a raw concentration. All errors are in AQI
> points. Say this if a professor asks about units.

### 1.4 The fact that drives half the project

Each photo's label is the **daily-average** AQI for that location, but a photo is a single **instant**.
Pollution rises and falls through a day, so the label is a slightly "smudged" answer key: even a perfect
photo-reader would disagree with the daily average by the day's within-day swing. This one fact is the
root of the **error ceiling** (Section 8) — hold onto it.

> **Worked example (why the smudged key matters).** Suppose a street's pollution over a day is
> [80, 120, 160, 120, 80] AQI at five moments; the **daily average** (the label) is 112. A photo taken
> at the 160 moment shows "160", but it is scored against 112 — a 48-point "error" that is *not the
> model's fault*. Averaging away time creates unavoidable error.

---

## 2. Machine learning from absolute zero

Everything later stands on these foundations. If you are shaky on any one of them, a professor's first
question can unravel the whole defense — so we build them slowly and completely.

### 2.1 What a "model" is

A **model** is a mathematical function with adjustable numbers inside it, called **parameters** (or
**weights**). You feed it an input and it computes an output. The simplest example is a straight line
`output = w · input + b`, with two parameters `w` (slope) and `b` (intercept). A deep network is the same
idea with *millions* of parameters arranged in layers, capable of representing very complicated functions.

- **Feature:** a piece of the input the model reads. Here: the pixels of a photo, plus two physics maps
  we compute (Section 5.1).
- **Label (target):** the correct answer for a training example. Here: the photo's AQI.
- **Parameters/weights:** the adjustable numbers the model learns.

### 2.2 How training works (gradient descent, intuitively)

- We define a **loss**: a single number measuring how wrong the model's outputs are on a batch of
  training examples (small loss = good).
- Training repeatedly (a) runs the model, (b) measures the loss, (c) computes how the loss would change
  if we nudged each parameter a tiny bit (the **gradient**), and (d) nudges every parameter a small step
  in the direction that *reduces* the loss. Repeat millions of times. This is **gradient descent**; the
  algorithm that does the nudging is the **optimiser** (we use one called AdamW).
- **Epoch:** one full pass over the training data. We train for up to 60 epochs.

> **Key idea.** Training = automatically tuning millions of parameters to make the loss small, one small
> gradient step at a time.

### 2.3 Generalisation, overfitting, and the three data splits

- **Generalisation** = performing well on inputs the model **never saw during training**. This is the
  entire point; a model that only does well on its training data is useless.
- **Overfitting** = memorising the training examples (including their noise) instead of learning the
  underlying pattern. Training looks great, new data fails.
- To detect and control this we split data into **train** (fit the parameters), **calibration/validation**
  (tune choices and, here, build the interval guarantee), and **test** (evaluated once, at the end, for
  the honest score). **Leakage** (Section 3) is a sneaky way apparent success on "test" is really
  memorisation — the central villain of this project.

### 2.4 Regression vs classification

- **Classification** picks a category (cat vs dog; or one of the six EPA AQI bands).
- **Regression** predicts a number. Predicting AQI is **regression**. (We also report an EPA-category
  accuracy as a secondary, classification-style view, by bucketing predicted AQI into the six EPA bands.)

### 2.5 Neural networks and CNNs

- A **neuron** computes a weighted sum of its inputs, adds a bias, and passes it through a simple
  non-linear function (an **activation**, e.g. ReLU which is `max(0, x)`). Non-linearity is what lets
  stacked layers represent complex patterns.
- A **neural network** stacks many neurons in **layers**; "**deep**" just means many layers.
- A **convolutional neural network (CNN)** is the variant built for images. Instead of connecting every
  pixel to every neuron, it slides small learnable **filters** (e.g. 3×3) across the image. Early filters
  learn edges; deeper layers combine them into textures, then object parts, then whole-scene properties
  (like *haze*). **Pooling** layers shrink the spatial size so the network sees larger context. The
  result of the CNN body is a compact **feature vector** summarising the image.

[[FIG:cnn]]

### 2.6 Backbones, pretraining, and transfer learning

- A **backbone** is a CNN body that turns an image into a feature vector. We use **EfficientNet-B0**
  (~5.3 million parameters) — small, efficient, and the strongest backbone the source paper found on this
  dataset.
- **Pretrained** means the backbone was already trained on a giant generic dataset (**ImageNet**,
  ~1.2 million labelled photos), so it already "knows" edges and textures.
- **Transfer learning / fine-tuning** = we start from those pretrained weights and continue training on
  our task. This works far better than random initialisation because our dataset (~7k training photos) is
  too small to learn good visual features from scratch. The backbone brings general vision knowledge; we
  adapt it to read haze.

> **Trap.** "Why not a bigger network?" → The source paper found EfficientNet-B0 *beats* ResNet50 and ViT
> on this ~7k-image dataset; with little data, extra capacity overfits. The bottleneck here is data and
> label noise, not model size.

### 2.7 The measuring sticks (know these cold — the whole story is told in them)

All in AQI points. Notation: `y` = truth, `yhat` = prediction, `n` = number of examples, `ybar` = mean of the
truths.

**MAE (Mean Absolute Error)** — the average size of the miss:

$$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

*"On a typical photo we're off by this many AQI points."* MAE does **not** depend on how spread out the
test set is, which makes it the most honest single accuracy number.

**RMSE (Root Mean Squared Error)** — like MAE but squares the errors first, so large misses are punished
more:

$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}$$

RMSE ≥ MAE always. If RMSE ≫ MAE, a few big errors dominate.

> **Worked example (MAE vs RMSE).** Errors on five photos: [5, 5, 5, 5, 60]. MAE = (5+5+5+5+60)/5 =
> **16**. RMSE = √((25+25+25+25+3600)/5) = √(3700/5) = √740 = **27.2**. The single 60-point miss barely
> moves MAE but nearly doubles RMSE — that is RMSE "punishing" the big miss. Our honest split has MAE 54
> and RMSE 95, a large gap, which tells us a subset of photos (the extreme-AQI tail) is missed badly.

**R² (the coefficient of determination) — the headline score.** It answers: *"what fraction of the
variation in the truth does the model explain, compared to the dumbest baseline of always guessing the
average?"*

$$R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2} = 1 - \frac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}}$$

- `SS_res` = the model's squared error. `SS_tot` = the squared error of always guessing the mean `ybar`.
- **R² = 1** perfect; **R² = 0** exactly as good as guessing the mean; **R² < 0** *worse* than guessing
  the mean (yes, it can be negative — not a bug).

[[FIG:r2]]

> **Worked example (R²).** Truths [10, 20, 30, 40, 50], mean ybar = 30, so
> SS_tot = 20²+10²+0+10²+20² = 1000. Model A predicts perfectly → SS_res = 0 → R² = 1. Model B always
> says 30 → SS_res = 1000 → R² = 0. Model C predicts [14, 18, 30, 42, 46] → SS_res = 4²+2²+0+2²+4² = 40
> → R² = 1 − 40/1000 = **0.96**. Model D (inverted) [50,40,30,20,10] → SS_res = 40²+20²+0+20²+40² = 4000
> → R² = 1 − 4000/1000 = **−3.0** (much worse than guessing).

**The crucial subtlety — R² depends on the test set's spread.** Because `SS_tot` grows with the test
set's variance, the *same model* can score very different R² on test sets of different spread. So we
**always** report MAE and the test-set standard deviation `SD(y)` next to R². A very useful identity we
use as a self-check (it follows from the definitions when SD uses the population form):

$$R^2 \approx 1 - \left( \frac{\mathrm{RMSE}}{\mathrm{SD}(y)} \right)^2$$

> **Worked example (the identity).** Our honest split has RMSE = 95.5 and SD(y) = 108.1, so
> 1 − (95.5/108.1)² = 1 − 0.780 = **0.220** — exactly the reported R². Every results row reconciles this
> way; that is one of our built-in correctness checks (Section 7.3).

**r²(Pearson) — a looser cousin, reported for honesty.** This is the *squared correlation coefficient*:
it measures only whether the model's ranking lines up with the truth, ignoring bias and scale, so it is
**always ≥ R²**. Some papers quietly call this "R²." We print **both** so nobody can quote the looser one
against us.

> **Trap.** If a reported "R²" looks suspiciously high, check whether it is really squared correlation.
> The strict coefficient of determination (ours) penalises systematic bias and wrong scale; squared
> correlation forgives both.

**Spearman (rank correlation).** *"Does the model order photos correctly from cleanest to dirtiest?"*
1.0 = perfect ordering. A **high Spearman with a modest R²** means the model *sees* pollution (ranks it
well) but is off on the exact number — which is exactly our situation (Spearman 0.65, R² 0.22–0.39).

**Coverage & interval width (for C1, Section 6).** For the low–high range: **coverage** = how often the
truth actually lands inside the range (target 0.90); **width** = the average size of the range. Coverage
is meaningless without width (an infinitely wide range always "covers"), so we always report both.

---

## 3. The data, and the honesty problem that shapes everything

### 3.1 The dataset

**PM25Vision**: street-level photos, each labelled with that location's **daily-average PM2.5 AQI**.
After removing 123 duplicate image IDs we use **11,096 photos across 3,259 monitoring stations** (median
2 photos per station, max 91). The imagery comes from Mapillary; the labels from the World Air Quality
Index (WAQI) project. The label distribution is **right-skewed** — many moderate-pollution photos, few
extreme ones — which matters for training (Section 5.4) and for the accuracy metric.

### 3.2 Auditing the data before modelling

A modelling result is only as trustworthy as the data under it, so we run three checks first.

- **Exact duplicates.** 123 rows shared an `image_id`; we drop them so the same photo cannot sit on both
  sides of a split.
- **Near-duplicates (perceptual hash).** A **perceptual hash** turns each image into a short fingerprint
  such that *visually similar images get similar fingerprints* (unlike a cryptographic hash, where one
  changed pixel changes everything). We group near-identical photos this way. This matters because
  near-duplicates from the same place are a leakage risk (Section 3.3).
- **The physics falsification test (the clever one).** Before trusting that images and labels are
  correctly paired, we compute a simple "how hazy is this photo" number — the mean **transmission**
  (Section 5.1) — and correlate it with the AQI label across the whole dataset. Physics predicts *hazier
  photo → more pollution → higher AQI*, i.e. a **negative** correlation between transmission and AQI. We
  measured **Pearson r ≈ −0.14** (highly significant, p ≈ 5e-6). It is a weak-but-real negative
  correlation in the predicted direction — so the pairing is correct, and the visual signal is real but
  *subtle*, which foreshadows why the honest R² is modest rather than near-zero or near-one.

> **Key idea.** We *falsification-tested* the data: if images and labels were mispaired, haziness would
> be unrelated to AQI. It correlates negatively exactly as physics predicts, so the data is sound.

### 3.3 Data leakage — the trap the whole project is built around

**Leakage** = information from the test set sneaking into training, so the score measures **memorisation**
rather than understanding.

> **Everyday analogy.** If a few real exam questions leak into your practice set, your mock score looks
> great — but it measures memorising those answers, not understanding the subject. On the real exam (new
> questions), you fall apart.

**Why THIS data leaks so easily.** Many photos come from the **same monitoring station**, and all photos
from one station on one day share a **near-identical daily-average label**. If a random split puts some of
a station's photos in training and others in test, the model can score well by **recognising the place**
("I've seen this exact street corner; its label is ~120") instead of **reading the haze**. That is
leakage, and our evaluation is specifically designed to forbid it — and then to *measure* how big it is.

### 3.4 The split family (how we divide photos into train / calibration / test)

We use a **0.65 / 0.15 / 0.20** split and build several **strategies**, changing only *how photos are
assigned* so any score difference is purely the split:

- **station-grouped (our primary, honest protocol):** every station's photos go entirely into one of
  train/cal/test. No station straddles the boundary, so the model is tested on **genuinely unseen
  places**. This measures real generalisation.
- **random (the leaky control we construct):** photos assigned independently → same-station photos land
  on both sides. This is the shortcut that leaks; we build it deliberately to measure the inflation.
- **geographic:** whole lat/lon regions held out (the hardest test — did it learn haze, or local
  scenery?).
- **temporal:** train on earlier dates, test on later ones.
- **shipped:** the dataset's own 80/20 split (which happens to be station-disjoint), used as a
  station-disjoint reference.

[[FIG:splits]]

**The calibration set is held strictly separate** from train and test, because the coverage guarantee in
Section 6 depends on the calibration data never being seen during training.

---

## 4. The decisive experiment — measuring the leakage (the headline result)

Here is the project's central finding. We train the **same model with the same random seed** and change
**only the split strategy**, so any difference in score is purely the split — a clean controlled
experiment.

[[FIG:leakage]]

### 4.1 The leakage gradient

| split | R² | MAE | RMSE | SD(y_test) | stations in >1 split |
|---|---|---|---|---|---|
| **random** (leaky control) | **0.759** | 26.4 | 42.9 | 87.4 | **1,221** |
| temporal | 0.380 | 43.5 | 58.7 | 74.5 | 393 |
| shipped | 0.378 | 44.5 | 64.3 | 81.5 | 0 |
| **station_grouped** (honest) | **0.220** | 54.2 | 95.5 | 108.1 | 0 |
| geographic (hardest) | −0.079 | 59.9 | 104.5 | 100.6 | 0 |

**Row-by-row reading:**
- **random = 0.759.** Deliberately leaky (1,221 stations appear on both sides). It scores **higher than
  the benchmark's reported 0.550**, and higher than our *entire* honest cross-validated range
  (Section 7). A leaky split manufactures a big score out of memorised places.
- **temporal = 0.380, shipped = 0.378.** Station-disjoint (or nearly), landing near our honest
  cross-validated mean.
- **station_grouped = 0.220.** The honest single number — and, we'll learn, a *hard, high-variance draw*.
- **geographic = −0.079.** Holding out whole regions is so hard the model does slightly *worse than
  guessing* — an honest measure of how badly single-photo AQI transfers across the globe.

### 4.2 The leakage gap (Δ_leak)

- **Lead with the variance-free MAE gap:** Δ_leak(MAE) = 54.2 − 26.4 = **27.8 AQI**. The leaky split
  roughly **halves the error.** We lead with MAE because, unlike R², it does **not** depend on the test
  set's spread — so it is the clean, honest measure of the effect.
- **Δ_leak(R²)** is 0.539 against the single hard draw (0.220), or 0.374 against the robust cross-validated
  mean (0.385). We report the pair, not just the most dramatic number.

### 4.3 The causal clincher — contaminated vs clean (no extra training)

A skeptic could object: *"maybe the random test set is just easier."* We rule that out **within the random
model's own test set** — same model, same test photos — by splitting them into **contaminated** (their
station or a near-duplicate is *also* in that model's training set) vs **clean** (genuinely unseen):

| subset | n | R² | MAE |
|---|---|---|---|
| contaminated (leaked) | 1,757 | **0.762** | 25.4 |
| clean (no leakage) | 463 | **−0.86** | 30.4 |

The high score lives **entirely in the leaked photos**. On genuinely unseen photos the same model is
worse than guessing. Because the test set and model are held fixed and only the leakage differs, this is
**causal evidence** that the inflation is leakage, not an easier test.

> **Trap — read the −0.86 correctly (a professor WILL probe this).** It is **variance-amplified, not a
> catastrophic failure.** The clean subset is small (463 photos) and dominated by low-variance stations,
> so its SD(y) is small; since R² = 1 − (RMSE/SD)², a *modest* error on a low-spread set drives R² sharply
> negative. Look at MAE: **25.4 vs 30.4** — only a little worse. **Lead with the MAE gap and the direction
> (contaminated ≫ clean); treat −0.86 as qualitative, not a precise honest estimate.**

> **Worked example (why small SD makes R² explode).** Suppose clean truths have SD = 20 and the model's
> RMSE on them is 30. Then R² ≈ 1 − (30/20)² = 1 − 2.25 = **−1.25** — very negative, from an RMSE
> (30 AQI) that is not even large. Same RMSE on a set with SD = 100 gives R² ≈ 1 − 0.09 = **+0.91**. The
> R² number is dominated by the denominator, which is why we lead with MAE.

### 4.4 What we claim, and what we do NOT claim

We do **not** accuse the benchmark authors of cheating, and we do **not** claim 0.55 is impossible
honestly (one honest cross-validation fold reaches 0.574, Section 7). We claim two things, and prove them:
(1) the data is **dramatically leakage-prone** (a leaky split inflates past 0.55, to 0.759), and (2)
single numbers on this task are **unreliable** (honest R² swings 0.15–0.57). The benchmark's *unspecified*
split merely **admits** the leakage mechanism.

> **Key idea (the whole thesis in one breath).** Same model, only the split changes: leaky 0.759, honest
> ~0.385; the advantage lives entirely in leaked photos (contaminated 0.762 vs clean −0.86). So the data
> leaks dramatically and single numbers are unreliable — that is the finding, honestly framed.

---

## 5. Building the honest model (in depth)

Now the machinery that produces the honest predictions. The guiding choice: **feed the network physics,
not just pixels.**

### 5.1 Physics features — teaching the model what haze looks like

Rather than hope the CNN rediscovers atmospheric optics from raw pixels, we hand it two pre-computed maps
that behave the same way in every city. The physical model of a hazy image (Koschmieder / He et al.) is:

$$I(x) = J(x)\,t(x) + A\,\big(1 - t(x)\big)$$

where `I` is the observed pixel, `J` is the true (haze-free) scene, `A` is the atmospheric light (the haze
colour), and `t(x) ∈ [0,1]` is the **transmission** (how much light survives to the camera). `t = 1` means
no haze; `t → 0` means dense haze washing everything toward `A`.

- **Transmission map (Dark Channel Prior, He et al. 2011).** In a clear outdoor photo, almost every tiny
  patch has at least one very dark pixel in some colour channel (a shadow, a dark window, tree bark). Haze
  adds a pale grey veil that **lifts** those dark pixels. So *how dark the darkest local pixel still is*
  tells you how much haze sits between camera and scene. We compute the **dark channel** (the per-pixel
  minimum over colour channels, then a local minimum-filter over a 15×15 patch), estimate the atmospheric
  light `A` from the haziest/brightest pixels, and then the transmission:

$$t(x) = 1 - \omega\ \min_{c \in \{R,G,B\}} \Big( \min_{y \in \Omega(x)} \frac{I^c(y)}{A^c} \Big),\qquad \omega = 0.95$$

  (`Ω(x)` is the 15×15 patch around pixel `x`; `ω<1` keeps a little haze so distant scenes still look
  natural.) Result: `transmission ∈ [0,1]`, **LOW where haze is dense, HIGH in clear air.**

- **Inverted-saturation map (Fang et al. 2024).** The dark-channel trick **fails on sky** (sky has nothing
  dark in it). But haze also **washes colour out** (mixes in white), lowering **saturation** (colour
  vividness). Sky has colour to lose, so an *inverted* saturation map (HIGH where saturation is LOW) stays
  informative exactly where the dark channel breaks down. The two maps cover each other's blind spots.

- **The 5-channel input.** We stack **[R, G, B, transmission, inverted-saturation]** into a 5-channel
  image. The RGB channels are normalised with ImageNet statistics (so the pretrained backbone sees what it
  expects); the two physics channels are standardised to a similar scale (otherwise their small [0,1]
  variance would make the pretrained stem down-weight them). These maps are **cached to disk once** —
  recomputing the Dark Channel Prior every training epoch would be far too slow (~1.1 GB cache for the
  full dataset).

> **Key idea.** Low transmission = dense haze; the inverted-saturation map catches haze in the sky where
> the dark channel can't. We give the network these two maps so it reads haze directly.

### 5.2 The backbone and the 3→5 channel stem

We use `EfficientNet-B0`, pretrained on ImageNet, but our input has **5** channels, not the usual 3. We
**widen the first convolutional layer** from 3 to 5 input channels; the `timm` library initialises the two
new channels from the pretrained RGB weights (a routine engineering step, not a research risk). We ask the
backbone for its **pooled feature vector** (by setting `num_classes=0`), not a classification — we attach
our own prediction heads to that vector. (Those same features later power the abstention OOD gate,
Section 8.2.)

### 5.3 The interval heads — three quantiles that can never cross

A **quantile** answers "what value is this fraction of outcomes below?" The 95th percentile is a value the
truth is below 95% of the time; the 5th percentile is a value it is above 95% of the time; the 50th is the
median. We predict the **5th, 50th, and 95th percentiles** of the AQI so the 5th–95th pair forms a
low–high interval.

> **Worked example (quantiles).** For the numbers [10, 20, 30, 40, 100], the 50th percentile (median) is
> 30, the "low" (5th-ish) is near 10, and the "high" (95th-ish) is near 100. Notice the median (30) sits
> *below* the mean (40) because the 100 pulls the mean up — this skew fact returns in Section 5.4.

If we predicted the three percentiles freely, the model could output a 95th **below** its 50th — nonsense.
We prevent that **by construction**: predict the lowest quantile freely, then add strictly-positive
`softplus` increments and cumulatively sum them:

$$q_1 = z_1,\qquad q_k = q_{k-1} + \mathrm{softplus}(z_k)\ \ (k>1),\qquad \mathrm{softplus}(z)=\log(1+e^{z})>0$$

Because every increment is strictly positive, the outputs can only increase → **q05 ≤ q50 ≤ q95 always**,
with no penalty term and no post-hoc sorting. The heads work in **log-AQI** space (stable on a skewed
target); exponentiating back to AQI preserves the ordering (exp is monotone).

### 5.4 The mean-vs-median problem, and the dedicated point head

This is a subtle but important design decision, and a favourite professor question.

- On a **right-skewed** target (few very-high-AQI photos), the **median sits below the mean** — the long
  high tail pulls the mean up but not the median.
- The **median minimises absolute error**, so the 50th-percentile head naturally predicts the median.
- But **R² rewards matching the conditional mean**, not the median (R² is built on squared error, whose
  minimiser is the mean). So reporting the median would **structurally depress R²** even if the model is
  good.

[[FIG:skew]]

Our fix: a **separate point head** trained with **MSE (squared error) on a standardised target**. MSE's
minimiser is the mean — exactly what R² rewards. We standardise the target so a fresh head starts by
predicting the dataset mean (a good starting point) and there is no unstable retransformation:

$$\tilde{y} = \frac{y - \mu}{\sigma}\ \ \longrightarrow\ \ \hat{y}_{\mathrm{AQI}} = \sigma\,\hat{\tilde{y}} + \mu$$

where `μ, σ` are the **train-set** AQI mean and standard deviation (stored inside the model so they travel
with the checkpoint). At inference we de-standardise linearly back to AQI. **The point head is our
accuracy output (MAE/RMSE/R²); the quantile heads still give the interval.**

> **Trap.** An earlier version mistakenly trained this head with a robust (Huber) loss whose small delta
> made it learn the **median** — which crushed R². Switching to **MSE on the standardised target** was the
> fix. Say: "point head = MSE + standardise + isotonic — mean-seeking; not the median, not Huber."

### 5.5 The losses

**Pinball (quantile) loss** teaches each head its percentile with an **asymmetric** penalty:

$$L_\tau(y, \hat{y}) = \max\big(\tau\,(y-\hat{y}),\ (\tau-1)\,(y-\hat{y})\big)$$

[[FIG:pinball]]

- For **τ = 0.95**, under-prediction (guessing too low, so `y − yhat > 0`) is penalised with weight 0.95,
  while over-prediction is penalised with weight `|0.95 − 1| = 0.05`. The ratio is 0.95/0.05 = **19×** —
  under-shooting the 95th percentile hurts 19 times more than over-shooting, so that head learns to sit
  near the *top* of the plausible range.
- For **τ = 0.05** it is the mirror (over-prediction hurts 19×), so that head sits near the bottom.
- For **τ = 0.50** the penalty is symmetric and this reduces to ordinary absolute error (the median).

> **Worked example (pinball asymmetry).** Truth y = 100, τ = 0.95. If the head predicts 80 (under by 20):
> loss = max(0.95·20, −0.05·20) = **19**. If it predicts 120 (over by 20): loss = max(0.95·(−20),
> −0.05·(−20)) = max(−19, 1) = **1**. Same 20-point error, but under-shooting costs 19× more — that is how
> the loss pushes the 95th-percentile head upward.

**Point loss (MSE)** on the standardised target, as in 5.4. The total training loss is
`pinball(quantile heads) + weight · MSE(point head)`.

### 5.6 Training

AdamW optimiser (learning rate 3e-4, weight decay 1e-4), a **cosine-annealed** learning rate with a
2-epoch warm-up (start slow, ramp up, then smoothly decay — helps stability and final accuracy), batch
size 32, up to 60 epochs with **early stopping on the calibration point-MAE** (stop when the metric we
care about stops improving, and keep the *best* checkpoint, not the last), mixed precision (faster on GPU),
fixed seeds (reproducibility). Two small-data helpers:

- **Class-balanced sampling** (train loader only): a weighted sampler oversamples the rare high-AQI photos
  so the model actually sees the extremes; calibration and test keep the **natural** distribution so the
  guarantee and the reported numbers stay honest.
- **Stochastic depth / drop-path**: randomly skips residual branches during training — a regulariser that
  helps on our modest dataset.

We also implemented three optional **accuracy levers**, all selected on calibration and all honest:
**EMA** (an exponential moving average of the weights, usually a bit more accurate and stable),
**test-time augmentation** (average the prediction over the photo and its mirror image), and a **seed
ensemble** (train several models with different seeds and average their outputs).

---

## 6. Honest uncertainty — turning predictions into a *guaranteed* range (C1)

A raw model can be over-confident (interval too narrow) or under-confident (too wide). We fix that with
two post-processing steps that need no retraining.

### 6.1 Conformalized Quantile Regression (CQR) — the coverage guarantee

CQR gives a **distribution-free, finite-sample guarantee**: the true value lands inside the interval
≥ 90% of the time, for **any** model (a weak model just gets honestly-wide intervals), as long as
calibration and test data are **exchangeable** (drawn from the same distribution). The recipe (Romano,
Patterson & Candès, 2019):

**Step 1.** On the **calibration** set, record how far outside its predicted range each truth fell:

$$E_i = \max\big(\hat{q}_{05}(x_i) - y_i,\ \ y_i - \hat{q}_{95}(x_i)\big)$$

`E_i` is negative when the truth was comfortably inside the interval, and positive by exactly the size of
the miss when it fell outside.

**Step 2.** Take **Q** = the finite-sample (1−α) quantile of those scores (α = 0.10 for 90% coverage):

$$Q = \mathrm{quantile}\left( \{E_i\},\ \ \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$

(The `(n+1)` correction is what makes the guarantee exact for a finite calibration set rather than only
asymptotically.)

**Step 3.** Widen **every** future interval by Q at both ends, and clip the lower end at 0 (AQI can't be
negative): `[q05 − Q, q95 + Q]`.

[[FIG:conformal]]

*Why it works, intuitively:* Q is your model's **own typical "how much I miss by"**, measured on held-out
data. Padding every interval by that amount buys back exactly the coverage you were missing. It is one
number and it is model-agnostic.

> **Worked example (conformal Q).** Say the calibration miss-scores `E_i`, sorted, are
> [−40, −22, −10, −3, 5, 18] (n = 6). For 90% coverage the level is ⌈7·0.9⌉/6 = ⌈6.3⌉/6 = 7/6 → clipped to
> the top value → Q = **18**. Every future interval is widened by 18 AQI at each end. Empirically we then
> verify coverage ≈ 0.90.

> **What is exchangeability, and can it break?** Exchangeability means calibration and test points are
> "interchangeable" draws from one distribution. Testing on **unseen stations** mildly breaks this (new
> places differ a little from calibration places), so our single-split coverage drifts to 0.87 — slightly
> below 0.90. This is honest, expected drift, and cross-validation coverage is 0.895 (Section 7.2).

### 6.2 Isotonic recalibration — removing systematic bias

The point estimate tends to **under-predict** the rare high-AQI photos, and that systematic bias hurts R².
**Isotonic regression** learns a **monotonic** (non-decreasing) map from prediction → truth, fit on the
**calibration** set only, and applied to test (never fit on test). It says e.g. "when the model outputs
~260, the truth averages ~340, so map 260 → 340", removing the bias while preserving the ordering (so it
cannot scramble the ranking). It is a standard, low-risk post-hoc step and needs no retraining.

> **Worked example (isotonic).** Suppose on the calibration set, whenever the raw head outputs ~200 the
> truth averages 210, and whenever it outputs ~300 the truth averages 360 — the head under-predicts, and
> more so at the top. Isotonic learns the non-decreasing map 200→210, 300→360 (interpolating between). At
> test, a raw 260 is mapped up to roughly 285. Because the map only ever *increases* with the input, two
> photos the model ranked in a given order stay in that order — so it fixes the *level* without touching
> the *ranking*. That is why it can lift R² and MAE without risk of scrambling Spearman.

### 6.3 Coverage vs width

We always report the two together. Our honest single split gives coverage ≈ 0.87 (the exchangeability
drift above); cross-validation gives mean coverage **0.895 ≈ 0.90**, which **validates** the whole
calibration machinery across independent folds.

---

## 7. How good is good? — evaluation and rigor

### 7.1 The honest single-split result

On the station-grouped (leakage-safe) split: **R² = 0.220, MAE 54.2, RMSE 95.5, Spearman 0.649, coverage
0.872, SD(y) 108.1** (n_train 7,217, n_test 2,216). Reading: the model **ranks** pollution well (Spearman
0.65) and its intervals are near-calibrated, but its point-R² is modest — and this particular split, as
cross-validation reveals, was a **hard, high-variance draw**.

### 7.2 Cross-validation — is that 0.22 trustworthy?

A single split could be lucky or unlucky. **Cross-validation** re-runs the whole evaluation on several
different splits and reports the spread. But **ordinary k-fold would leak here** (it shuffles all photos,
re-mixing same-station photos across train/test — exactly the flaw we expose). So we use **GroupKFold on
`station_id`**: the stations are divided into 5 groups; each fold tests one group and trains on the other
four, so **every station is tested exactly once and never appears in its own fold's training data** (0
straddling). Within each fold's training stations we carve a station-grouped calibration set, keeping the
0.65/0.15/0.20 shape.

**Result (5 folds):**

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R² | 0.405 | 0.574 | 0.365 | 0.429 | 0.152 |

**Honest R² = 0.385** (SD across folds 0.15; 95% CI [0.196, 0.574]) · MAE 46.3 · Spearman 0.679 ·
**coverage 0.895**.

Three things this tells us:
1. **The robust honest number is 0.385 — higher than the single split's 0.220**, which turns out to be one
   hard, high-variance draw (≈ fold 4's 0.152). Cross-validation *raised* our headline.
2. **The conformal intervals are validated** — mean coverage 0.895 ≈ 0.90 across five independent folds.
3. **Honest R² is highly split-dependent (0.15–0.57).** A *single* R² on this task is unreliable — which
   is exactly why single-number benchmarks (including 0.55) must be distrusted.

**The honest correction this forces:** one fold reached **0.574 > 0.55**, so we retract any claim that
"0.55 is unreachable honestly." 0.55 is *within* the honest range; the leakage proof is the causal test
(§4.3) and the leaky 0.759 exceeding the whole CI, **not** the impossibility of 0.55.

> **On the ± and the CI (a professor may probe this).** "95% CI [0.196, 0.574]" is a Student-t interval
> with half-width ≈ 0.19; the plain standard deviation across folds is 0.15 (a *different* quantity — do
> not confuse them). And these 5 folds come from **one** deterministic GroupKFold partition, so they are
> **correlated** (they share training data), which makes the plain t-interval mildly **optimistic**. A
> fully rigorous version repeats grouped CV over several seeds with the Nadeau–Bengio correction. The
> qualitative story (0.385, split-dependent, validated coverage) is unchanged.

### 7.3 How we know none of this is a bug — two correctness audits

Twice (2026-09-25 and 2026-09-28) we ran **three independent read-only audit agents** over every module.
Combined verdict: **the accuracy numbers, the leakage gradient, and the causal result are computed
correctly and honestly — no result-affecting bug.** They confirmed: prediction↔label↔row alignment (so the
contaminated/clean split is meaningful); station-disjointness holds *by construction*; the conformal math
is correct (a simulation hit coverage 0.9006); the point head is genuinely mean-seeking; and **every
results row reconciles** with the identity R² ≈ 1 − (RMSE/SD)² (e.g. 1 − (95.5/108.1)² = 0.220). The one
over-statement they caught was C2's first run (Section 8.1), now fixed.

---

## 8. The two extra contributions

### 8.1 C2 — the unavoidable-error ceiling

**The idea in one line:** you can't score full marks on an exam graded with a smudged answer key — and our
answer key (daily-average AQI) *is* partly smudged for an instantaneous photo. So there is a hard **upper
bound** on achievable R².

**The formula.** Write `y` for the daily-average label and `y*` for the instantaneous truth a perfect
photo-reader would output. The residual it still suffers is the within-day deviation `ε = y* − y`, with
variance `Var(ε)`. The best possible R² on predicting the labels is then:

$$R^2_{\max} = 1 - \frac{\mathrm{Var}(\varepsilon)}{\mathrm{Var}(y)}$$

where `Var(y)` is the spread of the labels **on the evaluation split** and `Var(ε)` is the typical
**within-day variance** of AQI.

**How we estimate Var(ε) — three careful refinements:**
1. **Per-hour InstantCast conversion.** We pull **hourly** PM2.5 from **OpenAQ** (a free global
   air-quality API), convert *each hourly reading* to AQI, and take the variance of those hourly AQIs
   about the day's mean — matching how the labels are built (not NowCast, not variance in µg/m³ space).
2. **Level-reweighting (the load-bearing fix).** Within-day swing depends on the pollution level, so a
   single average is misleading if the reference stations have a different pollution mix than our photos.
   We estimate the **noise curve** `v(m) = E[Var(ε) | day-mean AQI band]` and reweight it to PM25Vision's
   *own* label histogram `p(m)`:

$$\mathrm{Var}(\varepsilon) = \sum_m p(m)\,v(m)$$

3. **Split-specific Var(y).** The ceiling scales with the test set's spread, so we use the station-grouped
   test split's label variance (SD ≈ 108, so Var(y) ≈ 11,708).

**The measured result (32 OpenAQ stations, 6 countries, 1,241 station-days):** Var(ε) ≈ 208.6 (optimistic)
to 247.9 (conservative), Var(y) ≈ 11,708, giving **R²_max ≈ 0.98** — specifically 0.982 (optimistic) to
**0.979** (conservative, imputing the uncovered high bands), and 0.980 under the 2024 EPA table
(**breakpoint-insensitive**). The **interval-width floor** is ≈ 35 AQI (no honest 90% interval should be
narrower than the pollution's own within-day spread).

> **Worked arithmetic (the ceiling).** 1 − 208.6/11,708 = 1 − 0.0178 = **0.982** (optimistic).
> 1 − 247.9/11,708 = 1 − 0.0212 = **0.979** (conservative). Both round to ~0.98.

> **Worked example (level-reweighting).** Using the v(m) curve and dataset weights below:
> Var(ε) = 0.245·129.4 + 0.255·104.7 + 0.246·501.8 + 0.254·104.9 ≈ 31.7 + 26.7 + 123.4 + 26.6 =
> **≈ 208** — the optimistic Var(ε). Notice the 100–150 band contributes the most (its within-day swing,
> v = 501.8, is by far the largest), which is exactly why *missing* the even-higher bands would bias the
> ceiling; the conservative estimate fills them with a high v and still only reaches Var(ε) ≈ 248.

**The v(m) curve (within-day AQI variance by pollution band):**

| band (AQI) | station-days | v (variance) | within-day SD | dataset weight |
|---|---|---|---|---|
| 0–50 | 715 | 129.4 | 11.4 | 0.245 |
| 50–100 | 430 | 104.7 | 10.2 | 0.255 |
| 100–150 | 86 | 501.8 | 22.4 | 0.246 |
| 150–200 | 10 | 104.9 | 10.2 | 0.254 |

**What it means (the payoff):** label noise is only **~2% of the label variance** (Var(ε) ≈ 208–248 vs
Var(y) ≈ 11,708). So daily-average labeling costs at most ~2 points of R², and our honest **0.22 / 0.385
sits far below the ~0.98 ceiling.** Therefore the gap is the **difficulty of reading pollution from one
photo (the visual task), NOT noisy labels.** That is exactly what C2 exists to establish: the ceiling's
job was to answer *"is the remaining gap difficulty or bad labels?"* — and the answer is emphatically
**difficulty**.

> **Trap — own these caveats before the professor states them.** (i) The first pass used only ~13 stations
> and gave a *looser* 0.977; we feared the true ceiling might drop to ~0.90–0.95 once high-AQI bands were
> covered. The proper 32-station re-run shows it does **not** (holds at ~0.98). (ii) The reference still
> under-covers the very-high-AQI bands (200+, ≈13% of label mass), which are **imputed conservatively**;
> even so the ceiling only drops to 0.979. (iii) Report it as **~0.98, an upper bound** — not a single
> decimal, and not with a tight CI (the bootstrap band comes from few stations). (iv) The ceiling is
> **split-specific**, so compare a split's R² only to *its own* ceiling — we prove leakage with the causal
> test, not a "R² > ceiling" overlay.

### 8.2 C3 — inference-time abstention (knowing when to refuse)

[[FIG:riskcov]]

An honest system must sometimes say **"I can't answer this."** Two **independent** gates, checked in a
deliberate order:

**Gate 1 — OOD / unusable (checked FIRST): "is this even a valid daytime street photo?"** Night, indoor,
and blank-texture photos are **out of distribution (OOD)** — the model was never trained on anything like
them, so any number it produces is meaningless. Our gate uses **interpretable, station-invariant checks**:
a photo is flagged if it is **too dark** (night) or **too flat / featureless** (near-zero detail). These
are the right signal because a valid daytime street photo from *any* station is bright and detailed, so
the gate does not punish unseen locations. **One honest caveat:** a well-lit **indoor** scene is bright
*and* detailed, so these checks do **not** catch it — indoor is a documented near-OOD limitation of the
deployed gate (the Mahalanobis detector below would catch indoor, but at the cost of over-refusing valid
unseen-station outdoor photos — so neither is free, and we choose the station-robust one).

We also *tried* the textbook approach — a shape-aware **Mahalanobis distance** in the backbone feature
space —

$$D_M(f) = \sqrt{(f - \mu)^\top\,\Sigma^{-1}\,(f - \mu)}$$

where `f` is the photo's feature vector and `μ, Σ` are the mean and (Ledoit–Wolf-shrunk) covariance of the
good-image features; it measures *how unusual* a photo is relative to real street photos. **But we found
it over-refuses valid photos from unseen stations** (it refused ~99% of good test photos in one run): in
~1280-dim feature space, a new station's photos are genuinely far from the calibration photos, so the
detector confuses "new place" (benign) with "unusable photo". That is an honest, instructive finding — a
raw feature-distance detector is the wrong tool for a *station-generalisation* task — so we report its
AUROC for comparison but use the robust handcrafted gate. (We deliberately do **NOT** use interval width
for OOD either — the conformal guarantee is void off-distribution, so a width-based test would be
meaningless.)

**Gate 2 — uncertainty (checked second): "the photo is valid, but is the model too unsure?"** We refuse
the valid photos with the widest **log-space** interval width, `q95_log − q05_log`. Log space matters:
because we train on log(AQI), the raw-AQI width grows with pollution level, so a raw-width rule would just
refuse every high-AQI photo — the log width is the level-fair uncertainty signal.

**Thresholds are set on good images only** (never on labels or the OOD sets). For `τ_ood` we split the
held-out unseen-station good photos into **two disjoint slices**: we *fit* the 5% false-refusal budget on
one (`good_fit`) and *report* the achieved refuse-rate on the other (`good_eval`) — otherwise "≈5% refused"
would be true by construction rather than a real check. The threshold uses an exact **top-k rule** (refuse
the ⌊5%·n⌋ highest-scoring good photos) instead of a raw percentile, because the handcrafted score is
spiked at exactly 0 for most good photos and a plain percentile could collapse to 0 and refuse arbitrary
mildly-dim photos. `τ_width` = the calibration log-width at the target coverage. The gate logic: if
`ood_score > τ_ood` → abstain (OOD); else if `log_width > τ_width` → abstain (uncertain); else answer.

**What we measure.** Selective prediction via **risk–coverage** curves (accept the most-confident
fraction, watch the error), the **AURC** (area under the risk–coverage curve — lower is better), and
selective MAE/RMSE/category/coverage at 90/80/70% coverage; plus baselines (random, oracle). We report
these for the **deployed log-width signal** as the headline (report == deploy), with the raw-AQI width
shown only as a labelled diagnostic — and we state the honest finding that on this hard task log-width is a
*weak* error predictor, so the OOD gate, not the uncertainty gate, is C3's main practical value. For the
OOD gate, **OOD-detection** metrics against three external "should-refuse" sets pushed through the
*identical* physics pipeline — **ExDark** (night, far-OOD), **DTD** textures (featureless, far-OOD), and
**MIT-Indoor** (indoor, near-OOD): **AUROC** (ranking quality, 0.5 = chance, 1.0 = perfect), **AUPR**, and
**FPR@95TPR** (the false-positive rate when we catch 95% of the OOD images — lower is better), plus a
**refuse-rate table** (high on the *far*-OOD sets, ≈5% on held-out `good_eval`, and — honestly — near
chance on MIT-Indoor for the handcrafted gate, the indoor limitation made measurable). To show the
Mahalanobis over-refusal is genuine *station shift* and not just in-sample optimism, we set its threshold
on a **held-out same-station** slice and report the *excess* refusal on unseen-station test.

> **Worked example (AUROC).** Give every photo its OOD score (here the handcrafted too-dark/too-flat
> score). AUROC is the
> probability that a randomly chosen *unusable* photo scores higher than a randomly chosen *good* photo. If
> good photos score, say, [1, 2, 3] and night photos score [4, 5, 6], every night photo out-scores every
> good photo → AUROC = 1.0 (perfect separation). If the two sets overlap completely → AUROC = 0.5
> (chance). **FPR@95TPR** asks: to catch 95% of the unusable photos, how many *good* photos do we wrongly
> refuse? Lower is better. These let us report OOD-detection quality as single, comparable numbers.

*(C3 is built and locally tested; the exact metric values land after the `08_abstention.ipynb` run.)*

---

## 9. Delivering it — the live demo and the full picture

### 9.1 The one-photo pipeline (the demo)

`predict(photo)` runs the whole trained pipeline on a single image: build the 5-channel physics input →
EfficientNet-B0 → invert the log quantiles → apply the conformal widening Q → isotonic-recalibrate the
point estimate → run the abstention gate. It returns the **AQI estimate**, the **low–high interval**, the
**EPA category + colour**, an **answer / abstain** decision with a reason, and the two physics maps. A
**Gradio** app wraps this, deployed as a permanent **Hugging Face Space** (a free, always-on public web
app) — so the honesty is *visible*: upload a night photo and the system tells you it won't guess.

> **Worked example (one photo, end to end).** A daytime street photo comes in. (1) We compute its
> transmission and inverted-saturation maps and stack them with RGB → a (5, 224, 224) tensor. (2)
> EfficientNet-B0 turns it into a ~1280-number feature vector. (3) The quantile heads output, in log
> space, [q05, q50, q95]; exponentiating gives, say, [95, 130, 190] AQI. (4) Conformal widening adds
> Q ≈ 30: the interval becomes [65, 220]. (5) The point head outputs a standardized value; de-standardise
> and isotonic-recalibrate → AQI ≈ **148**. (6) The abstention gate computes the handcrafted OOD score
> (too-dark/too-flat — here below τ_ood → a valid daytime photo) and the log width (below τ_width →
> confident) → **answer**. Output: "AQI ≈ 148, range 65–220, category *Unhealthy for Sensitive Groups*,
> ✅ answer." A night photo would instead fail the OOD gate at step 6 (too dark) and return "🚫 abstain —
> too dark (night)."

### 9.2 How everything connects (the end-to-end picture)

[[FIG:pipeline]]

Read the pipeline as one honest chain: **audit the data → split it so it can't cheat → turn each photo
into physics-aware features → predict a calibrated range and a mean-seeking point → guarantee the range's
coverage and de-bias the point → evaluate leakage-safely and cross-validate → bound the achievable
accuracy (C2) → refuse when unsure (C3) → serve it live.** Every box exists to make the final number
**trustworthy**, not merely high. That is the project's identity.

---

## 10. Presenting and defending

### 10.1 The five-minute story (rehearse out loud)

1. **Goal & honesty (30s).** Estimate AQI from one street photo, honestly — a calibrated range, an error
   ceiling, refuse-to-answer, and leakage-free evaluation.
2. **The data & the trap (45s).** ~11k photos, ~3,260 stations, many photos per station sharing one
   daily-average label — which makes the data dangerously easy to leak.
3. **The ruler (45s).** Split into train/cal/test; grade with R² (how much better than guessing the
   average; can be negative), MAE, and Spearman.
4. **The finding (90s).** Same model, only the split changes: honest (cross-validated) **0.385 ± 0.19**
   (folds 0.15–0.57), leaky **0.759** — above the whole honest range. Inside the leaky split the advantage
   is entirely in leaked photos (contaminated 0.762 vs clean −0.86) — proof it's leakage, not an easier
   test. We don't claim 0.55 is impossible (a fold hit 0.574); the point is single numbers are unreliable
   and the data leaks dramatically.
5. **The honest machine (60s).** Physics 5-channel input, EfficientNet-B0, a mean-seeking point head for
   accuracy, conformal-calibrated intervals for honesty (~90% coverage), isotonic de-biasing.
6. **Bulletproofing (30s).** An error ceiling (~0.98, an upper bound) shows the score is task-limited not
   label-limited; leakage-safe cross-validation gives 0.385 ± CI with validated coverage; two 3-agent
   audits found no bug.
7. **Contribution (20s).** An honest number **plus** a measured, causally-proven flaw in how these
   benchmarks are usually scored — plus abstention and a live demo.

### 10.2 The professor Q&A bank

*Bold = the answer to say; the rest is backup if pushed.*

**Framing & motivation**
- *In one sentence, what did you do?* → **Estimate AQI from a street photo honestly, and prove a
  benchmark's high score is inflated by leakage.**
- *Why AQI and not µg/m³?* → **It's the label the dataset ships and what apps show users;** training in AQI
  avoids a lossy conversion and stays comparable to the benchmark. µg/m³ appears only in the C2 ceiling.
- *What's the real contribution — image→AQI already exists?* → **Honesty and measurement:** we expose and
  *measure* leakage inflation (Δ_leak), prove it causally, and add calibrated intervals + an error ceiling
  + cross-validation + abstention.
- *Why should anyone trust a lower number than the benchmark?* → **Because the higher number is inflated
  by leakage, which we prove; the honest, robust number is 0.385 with validated intervals.**

**Metrics**
- *What is R²?* → **The fraction of the truth's variation the model explains vs always guessing the mean;**
  1 perfect, 0 = mean, negative = worse than the mean.
- *R² can be negative?* → **Yes** — geographic split (−0.079) and the clean subset (−0.86) are real
  negatives; the 0-to-1 intuition is for *squared correlation*, a different quantity.
- *Why report both R² and r²(Pearson)?* → **Because papers use "R²" for two different things;** the strict
  R² penalises bias/scale, squared correlation doesn't and is always ≥ it. We print both so nobody quotes
  the looser one against us.
- *Why is 0.385 good when the benchmark got 0.55?* → **It's field-normal and honestly measured;** the
  closest honest analogue (Mondal 2024) is ≈ 0.39, and 0.55 sits at the optimistic edge, *within* our fold
  range.
- *Isn't 0.22 just a bad model?* → **No:** Spearman 0.65 (it ranks pollution), near-calibrated intervals,
  and it sits far below the ~0.98 ceiling — the low point-R² is the task's difficulty, not a failure to
  learn.
- *Why is MAE 46–54 acceptable on a 1–530 scale?* → **The task is genuinely hard (single instant vs
  daily-average label) and MAE is variance-free;** it's in the field-normal range and we report it beside
  R² precisely so the picture is honest.

**Leakage**
- *What is leakage here?* → **Same-station photos share a near-identical label;** a random split lets the
  model win by recognising the place, not reading the haze.
- *How did you PROVE it, not just claim it?* → **Two experiments:** the gradient (same model, only the
  split changes → 0.759 down to −0.079) and the causal test (contaminated 0.762 vs clean −0.86 — the
  advantage lives entirely in leaked photos).
- *Couldn't the random test set just be easier?* → **No — contaminated-vs-clean uses the same test set and
  model;** only leakage differs and it explains the whole gap.
- *Do you claim the authors used a random split?* → **No; their split is unspecified.** We claim it
  *admits* the mechanism; a leaky split reproduces and exceeds 0.55, and the causal test proves the
  inflation is leaked photos.
- *Why is clean-R² only "qualitative"?* → **It's from the random-trained model on a small, low-variance
  subset, so −0.86 is variance-amplified;** the load-bearing claim is the direction and the MAE gap.

**Splits & cross-validation**
- *Why station-grouped splitting?* → **Because the unit of leakage is the station;** grouping tests the
  model on genuinely unseen places.
- *Why GroupKFold, not plain k-fold?* → **Plain k-fold re-leaks same-station photos;** grouping by station
  keeps every fold honest. It turns one number (0.22) into 0.385 ± CI.
- *Is your CI rigorous?* → **It's a Student-t CI across 5 folds, which is mildly optimistic because the
  folds are correlated;** the rigorous version is repeated CV with the Nadeau–Bengio correction — we state
  this openly.

**Model & physics**
- *Why physics channels instead of raw pixels?* → **They encode the haze signal explicitly and generalise
  across cities,** so the model doesn't have to rediscover atmospheric optics from pixels.
- *How does the Dark Channel Prior measure haze?* → **Clear patches have a very dark pixel; haze lifts it;**
  so `1 − ω·darkchannel` estimates transmission (low = dense haze). The inverted-saturation map catches
  haze in the sky where the dark channel fails.
- *Why a separate point head, not the median?* → **The median under-shoots on skewed data and depresses
  R²;** the MSE point head estimates the mean, which R² rewards.
- *Why are the quantiles guaranteed not to cross?* → **Cumulative softplus:** the lowest is free, each step
  up adds a strictly-positive increment, so outputs can only increase.
- *Why train on log(AQI)?* → **The label is right-skewed;** log stabilises the loss so a few extreme photos
  don't dominate, and exponentiating back preserves ordering.

**Uncertainty & calibration**
- *Why a range instead of a number?* → **Honesty about uncertainty** — and conformal calibration
  *guarantees* ~90% coverage, which a lone number can't.
- *Why does the conformal guarantee hold?* → **Measure your typical miss on held-out calibration data,
  widen every interval by it;** under exchangeability this provably gives ≥90% coverage for any model.
- *Coverage is 0.87 not 0.90 — is it broken?* → **No — testing on unseen stations slightly breaks
  exchangeability,** so coverage drifts a little; CV coverage is 0.895. Honest, not a bug.
- *What does isotonic recalibration do?* → **Removes the systematic tail bias** with a monotone
  prediction→truth map fit on calibration; it can't reorder predictions, so it's low-risk.

**Ceiling & abstention**
- *What is the error ceiling and why an upper bound?* → **The best R² possible given daily-average labels
  vs instant photos;** an upper bound because it counts label noise only (real models also lose to weak
  signal).
- *Is R²_max ≈ 0.98 trustworthy?* → **As an upper bound, yes** — 0.979–0.982, breakpoint-insensitive, 32
  stations / 6 countries; report ~0.98 (not a tight CI), with the top ~13% extrapolated conservatively.
- *What does the ceiling actually prove?* → **That label noise is tiny (~2%), so our gap to it is the
  visual task, not the labels** — it reframes "our R² is low" as "the task is hard, and we're honest about
  it."
- *How does the model decide to refuse?* → **Two gates:** an OOD gate (interpretable, station-invariant
  "too dark / too flat" checks) first, then an uncertainty gate (log interval width); a 5% false-refusal
  budget on good photos. *(We also tried a feature-space Mahalanobis detector but it over-refuses unseen
  stations — an honest finding, so we use the robust semantic checks instead.)*
- *Why not use interval width for OOD?* → **Conformal validity is void off-distribution,** so width is
  meaningless there; we use feature distance instead.
- *What are AUROC and FPR@95?* → **AUROC = how well the OOD score ranks unusable above usable photos**
  (0.5 chance, 1 perfect); **FPR@95 = the false-refusal rate when we catch 95% of the unusable ones**
  (lower is better).

**Delivery & rigor**
- *How do you know there are no bugs?* → **Two independent 3-agent audits verified every module,** plus the
  R² ≈ 1−(RMSE/SD)² self-check reconciles every row.
- *What's the demo?* → **A Gradio web app** (Hugging Face Space) that runs the exact trained pipeline on an
  uploaded photo and shows AQI + interval + category + an answer/abstain badge.

**Practical & scope**
- *What dataset, and how big?* → **PM25Vision:** 11,096 photos across 3,259 monitoring stations after
  dedup; imagery from Mapillary, labels from WAQI.
- *How big is the model and how long to train?* → **EfficientNet-B0, ~5.3M parameters;** a single model
  trains in ~30 minutes on a free T4 GPU; cross-validation is 5 such runs.
- *Why a 0.65 / 0.15 / 0.20 split?* → **Enough to train, a dedicated calibration set for the conformal
  guarantee, and a 20% held-out test evaluated once.**
- *What exactly is the calibration set for?* → **Two honesty steps that must never see training or test
  data:** computing the conformal widening Q, and fitting the isotonic map. Selecting on it also drives
  early stopping.
- *Why the 5th/95th percentiles, not 10th/90th?* → **5/95 gives a nominal 90% central interval,** which
  matches our target coverage; conformal then adjusts it to the exact guarantee.
- *Could you just use more data / a bigger model?* → **More honest data would help most** (the bottleneck
  is leakage-safe training images, not compute); a bigger backbone is a config change we can add as a new
  row, but the paper found B0 best at this scale.
- *How is this different from ordinary dehazing?* → **We don't remove haze; we read it.** The Dark Channel
  Prior gives us a haze *measurement* (transmission) that we feed the model as a feature.
- *Is the abstention just a confidence threshold?* → **No — it's two gates:** an out-of-distribution gate
  first (station-invariant "too dark / too flat" checks, for unusable photos), then a calibrated
  uncertainty gate (for valid but hard photos).
- *What would make this production-ready?* → **A second dataset for external validity, group-conformal for
  per-region coverage, the C3 run, and a larger honest training set;** the architecture and honesty
  machinery are already in place.
- *What is the single biggest limitation?* → **The task is intrinsically hard** (one instant vs a
  daily-average label across new places), so honest accuracy is modest; we are transparent about that and
  bound it with the C2 ceiling.

### 10.3 Traps & honest caveats (own them before you're asked)

1. R² depends on the test set's variance — always cite MAE + SD next to it.
2. "R²" is ambiguous — we report the strict coefficient of determination **and** squared-Pearson.
3. The −0.86 clean R² is **variance-amplified**; lead with the MAE gap (25.4 vs 30.4).
4. 0.55 is **not** unreachable honestly (a fold hit 0.574) — the claim is unreliability + leakage, not
   impossibility.
5. The CV CI is mildly optimistic (correlated folds); the rigorous fix is repeated CV + Nadeau–Bengio.
6. The ceiling is an *upper bound* (~0.98), split-specific, with the top ~13% extrapolated conservatively.
7. Coverage 0.87 < 0.90 on one split is expected drift (broken exchangeability on unseen stations), not a
   bug; CV coverage 0.895 confirms the method.
8. The point head is **MSE + standardise + isotonic** — not the median, and not the old Huber approach.
9. We never tune on test — every choice is made on calibration/CV, and test is evaluated once.
10. We don't accuse anyone of cheating; we measure a mechanism the benchmark's unspecified split admits.

---

## 11. Who did what — the 4-person work split

*(Full detail, with each member's files and likely questions, is in `docs/TEAM_CONTRIBUTIONS.md`.)*

| Member | Owns | Presents |
|---|---|---|
| **Keyaan** | Data load/clean/audit, leakage-safe splits, the leakage experiment, integration + results ledger | The central leakage finding + the honest-evaluation thesis |
| **Anup** | Physics features (DCP + inverted saturation), EfficientNet-B0 + quantile/point heads, losses, training | How the model works |
| **Lamiya** | Conformal calibration, isotonic recalibration, metrics, cross-validation, the correctness audits | How we know the numbers are honest and the intervals hold |
| **Asif** | C2 error ceiling, C3 abstention, the inference pipeline + Gradio demo (HF Space) | The ceiling, abstention, and the live demo |

Everyone should know the **five-minute story** and the **top-level thesis**, so any member can field a
cross-cutting question.

---

## 12. Glossary (say-it-in-one-breath, then deeper)

- **AQI** — one-breath: the pollution number apps show (1–530). Deeper: a piecewise-linear index of
  pollutant concentration; ours is the daily-average PM2.5 AQI.
- **Model / parameters / feature / label** — a tunable function / its adjustable numbers / the input it
  reads / the correct answer.
- **Training / gradient descent / epoch** — tuning parameters to shrink the loss / the small-step
  optimisation that does it / one pass over the data.
- **Generalisation / overfitting** — doing well on unseen inputs / memorising the training set.
- **CNN / filter / feature map / pooling** — an image network / a small sliding detector / its output map /
  a shrink-and-summarise step.
- **Backbone / pretrained / transfer learning** — a reusable CNN body / already trained on ImageNet /
  adapting those weights to our task.
- **MAE / RMSE** — average absolute miss / root-mean-square miss (punishes big misses more).
- **R² (coefficient of determination)** — fraction of variance explained vs guessing the mean; can be
  negative; depends on test-set spread.
- **r²(Pearson)** — squared correlation; ranking only; always ≥ R².
- **Spearman** — rank correlation; are photos ordered correctly?
- **Leakage** — test info reaching training, so the score measures memorisation.
- **Station-grouped split / GroupKFold** — all of one station's photos in one split / the cross-validation
  version (rotate which fifth of the *stations* is tested).
- **Δ_leak** — how much a leaky split inflates the score over the honest one.
- **Dark Channel Prior / transmission** — the haze-estimation trick / low = dense haze.
- **Inverted saturation** — high where colour is washed out (hazy/sky).
- **Quantile / pinball loss** — a "fraction-below" value / the asymmetric loss that teaches a percentile.
- **Softplus / monotone heads** — a smooth positive function / heads whose outputs can't cross by
  construction.
- **Conformal prediction (CQR) / exchangeability** — the procedure that guarantees ~90% coverage for any
  model / the assumption (cal and test drawn alike) it needs.
- **Isotonic recalibration** — a monotone prediction→truth map that removes systematic bias.
- **Coverage / width** — how often the truth is inside the range / how wide the range is.
- **R²_max (error ceiling) / Var(ε) / Var(y)** — the best R² achievable given daily-average labels (an
  upper bound, ~0.98) / within-day label noise / label spread.
- **Abstention / OOD / Mahalanobis distance** — refusing to answer / an input unlike the training data /
  a shape-aware distance in feature space used to detect OOD.
- **AURC / AUROC / FPR@95** — area under the risk–coverage curve (abstention quality) / OOD-ranking
  quality / false-refusal rate at 95% true-positive rate.

---

## 13. Design decisions and the alternatives we rejected

A professor's favourite move is *"why did you do X and not Y?"*. Here is every major choice, what we
rejected, and the reason — so you can answer instantly. The meta-principle: **every choice defends
honesty and generalisation**, sometimes at the cost of a higher-but-fake number.

| Decision | We chose | We rejected | Why |
|---|---|---|---|
| Evaluation split | **station-grouped** | random split | random puts same-station photos on both sides → leakage → inflated score |
| Point objective | **MSE on a standardized target** (mean) | median / Huber | the median under-shoots on right-skewed labels and depresses R² |
| Interval method | **conformal (CQR)** | Gaussian / Bayesian intervals | CQR *guarantees* ~90% coverage for any model, distribution-free |
| Quantile ordering | **cumulative softplus** (by construction) | penalty term or post-hoc sort | guaranteed non-crossing with no extra loss and no hacks |
| Backbone | **EfficientNet-B0** | ResNet50 / ViT / bigger nets | the source paper found B0 best on ~7k images; extra capacity overfits |
| Input | **5-channel (RGB + 2 physics maps)** | RGB only | physics generalises across cities; RGB-only scored lower and leaks scenery |
| Augmentation | **geometric only** | photometric (brightness/contrast) | photometric jitter corrupts the exact haze signal we rely on |
| Quantile target space | **log(AQI)** | raw AQI | log stabilises the skewed target so a few extremes don't dominate |
| Robustness check | **GroupKFold by station** | plain k-fold | plain k-fold re-leaks same-station photos |
| OOD score | **handcrafted station-invariant checks** (too dark / too flat) | Mahalanobis on features; interval width | Mahalanobis over-refuses unseen-station photos (station shift); interval width is invalid off-distribution (conformal void there) |
| De-biasing | **isotonic on calibration** | none / parametric fit | monotone (can't reorder), removes tail bias, low-risk, no retraining |
| Ceiling noise estimate | **hourly InstantCast, level-reweighted** | unweighted average / NowCast | matches how the labels are built and our dataset's pollution mix |
| Early-stopping metric | **calibration point-MAE** | training loss / last epoch | we select on the metric we care about, on held-out data, and keep the best checkpoint |

> **Key idea.** If a professor asks "why not the higher-scoring option?", the answer is almost always:
> "because that option leaks or over-fits, and our goal is an honest number, not a fake one."

---

## 14. Limitations and honest future work

Stating limitations *before* you're asked is the strongest defensive move — it shows you understand the
work's edges. Ours:

- **Honest accuracy is modest (R² ≈ 0.385).** Reading instantaneous pollution from a *single* photo of a
  *new* location is genuinely hard; the honest number is field-normal, not a failure — but it is not
  high, and we say so. More photos per station, multi-photo aggregation, or temporal context would help.
- **The C2 ceiling's top ~13% is extrapolated.** The OpenAQ reference still under-covers the very-high-AQI
  bands (200+); we impute them conservatively (the ceiling barely moves), but a targeted high-pollution
  fetch would let us measure them directly and tighten ~0.98.
- **The cross-validation CI is mildly optimistic.** Five folds from one GroupKFold partition are
  correlated; a repeated grouped CV with the Nadeau–Bengio correction would give a fully rigorous
  interval. The point estimate (0.385) and the spread (0.15–0.57) are unaffected.
- **Coverage drifts under distribution shift.** On unseen stations, exchangeability is mildly broken, so
  single-split coverage is 0.87 rather than 0.90. Group-balanced or Mondrian conformal could recover the
  target per subgroup.
- **The C3 abstention numbers are pending one run.** The method is built and locally tested; the AURC /
  AUROC / FPR95 values land after `08_abstention.ipynb` runs on the three external OOD sets.
- **We evaluate one dataset.** PM25Vision is large and global, but a second dataset would strengthen the
  external validity of both the honest number and the leakage effect.

**Future work (all low-risk extensions):** a larger backbone reported as a *new row* beside B0 (both on
record); additional physics channels; multi-photo or temporal aggregation to beat the single-instant
ceiling; group-conformal for per-region coverage; and a repeated-seed CV for the rigorous CI.

---

## Appendix A — the number sheet (every headline figure)

- **Data:** 11,096 photos / 3,259 stations (median 2, max 91); 123 duplicate image IDs removed;
  falsification Pearson r ≈ −0.14.
- **Honest single split (station_grouped):** R² 0.220 · MAE 54.2 · RMSE 95.5 · Spearman 0.649 · coverage
  0.872 · SD(y) 108.1 · n_train 7,217 · n_test 2,216.
- **Leakage gradient (R²):** random 0.759 · temporal 0.380 · shipped 0.378 · station_grouped 0.220 ·
  geographic −0.079. Δ_leak(MAE) = 27.8 AQI; Δ_leak(R²) = 0.539 (vs 0.220) / 0.374 (vs 0.385).
- **Causal test:** contaminated R² 0.762 (n 1,757, MAE 25.4) vs clean R² −0.86 (n 463, MAE 30.4).
- **Cross-validation:** R² 0.385 (SD 0.15; 95% CI [0.196, 0.574]) · MAE 46.3 · Spearman 0.679 · coverage
  0.895; folds {0.405, 0.574, 0.365, 0.429, 0.152}.
- **C2 ceiling:** R²_max ≈ 0.98 (0.982 optimistic / 0.979 conservative; 0.980 under 2024 breakpoints);
  Var(ε) 208.6–247.9; Var(y) 11,708; interval-width floor ≈ 35 AQI; 32 stations / 6 countries / 1,241
  station-days.
- **Benchmark:** EfficientNet-B0 R² 0.550 (MAE 36.6, RMSE 54.6). Closest honest analogue Mondal 2024 ≈
  0.39.
- **C3:** built; risk–coverage/AURC + OOD AUROC/FPR95 land after the `08_abstention.ipynb` run.

## Appendix B — the equation sheet

- MAE, RMSE, R², and the identity R² ≈ 1 − (RMSE/SD)² — Section 2.7.
- Hazy-image model I = J·t + A(1−t) and the DCP transmission — Section 5.1.
- Monotone quantiles via cumulative softplus — Section 5.3.
- Standardise/de-standardise the point head — Section 5.4.
- Pinball loss L_τ — Section 5.5.
- Conformal score E_i and widening Q — Section 6.1.
- Error ceiling R²_max = 1 − Var(ε)/Var(y) and Var(ε) = Σ p(m)v(m) — Section 8.1.
- Mahalanobis distance D_M — Section 8.2.
