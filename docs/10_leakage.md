# The leakage experiment — why a benchmark's high score can be an illusion

*(Companion to `notebooks/09_leakage.ipynb`. Read this if you want the story without running anything.
Numbers live in [`RESULTS.md`](RESULTS.md); the code that produced them is in
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).)*

---

## 1. What "data leakage" means (the exam analogy)

Imagine studying for an exam and, by accident, your practice questions include some of the *actual*
exam questions with their answers. You'd score brilliantly — but the score wouldn't mean you understood
the subject. You'd just have **memorised specific answers**. When you later face genuinely new
questions, you'd do far worse.

**Data leakage** is that mistake in machine learning: information from the test set sneaks into
training, so the test score measures *memorisation*, not *understanding*. The model looks great in the
lab and fails in the real world.

## 2. Why this dataset is especially prone to it

Our label is the **daily-average** air quality at a monitoring station. A single station often has
**many photos on the same day** (median 2, up to 91 per station overall). Every photo from one station
on one day carries **the same, or nearly the same, label**.

So if you split the photos *randomly*, near-identical photos of the *same street corner with the same
answer* end up on **both** sides — some in training, some in the test set. The model doesn't need to
learn what haze looks like; it just needs to recognise "ah, this is that corner in Beijing → the answer
is 180." That's the exam leak, built right into a careless split.

**The fix** is a **station-grouped split**: every photo from a given station goes *entirely* to one
side (train **or** cal **or** test, never split across them). Now the test stations are genuinely
**unseen places**, and the score measures real generalisation. This is our primary protocol.

## 3. The experiment: change *only* the split, measure the score

The cleanest way to prove leakage inflates scores is to hold **everything** constant — same model, same
random seed, same physics cache, same training recipe — and vary **only how the data is split**. Any
change in the score is then attributable to the split alone. We ran five splits:

| split | what it holds out | honest? |
|---|---|---|
| **random** | nothing — photos assigned independently | ❌ leaky control |
| **temporal** | later dates (train on the past) | partly |
| **shipped** | the dataset's own 80/20 split (the paper's geometry) | ✅ station-disjoint |
| **station_grouped** | whole stations | ✅ **our primary** |
| **geographic** | whole world regions (lat/lon cells) | ✅ hardest |

### What we found (the "gradient")

- **random: R² = 0.759** — the leaky control.
- **station_grouped: R² = 0.220** — the honest number.
- **Δ_leak(R²) = 0.539** — a random split more than triples the honest R².

The leaky split's **0.759 sails past the benchmark paper's reported 0.55.** Meanwhile *every*
station-disjoint split lands far lower (shipped 0.378, station_grouped 0.220, geographic −0.079). We
**could not reach 0.55 on any honest split**, but a leaky one overshoots it easily. That's the headline:
the benchmark's number is exactly what you'd expect from a station-leaky split.

> We are careful **not** to claim the paper *definitely* used a random split — its split is unspecified.
> We claim something more precise and defensible: its unspecified 80/20 split *admits* this leakage
> mechanism, and the mechanism reproduces (and exceeds) its number.

## 4. The clincher: proving it's leakage, not an easier test set

A skeptic could object: "maybe the random test set is just *easier*, not leaked." So we did a check
that needs **no extra training** and settles it.

Take the random model's *own* test set and label each photo:
- **contaminated** = its station (or a near-duplicate image, found by perceptual hashing) also appears
  in that model's training data — i.e. leaked;
- **clean** = neither does — genuinely unseen.

Then score the two groups separately:

| subset | share | R² |
|---|---|---|
| contaminated (leaked) | 79.1% | **0.762** |
| clean (unseen) | 20.9% | **−0.86** |

The high score lives **entirely in the leaked photos**. On the clean, genuinely-unseen photos the very
same model is **worse than just guessing the average** (a negative R²). It didn't learn to read haze; it
learned to recognise places. That is leakage, demonstrated causally.

*(Honest note we keep in the write-up: the clean group comes from the random-trained model, so it isn't
a perfect stand-in for the station_grouped number — it's actually a bit worse, because a model trained
with leakage generalises poorly to new places. The point that carries the argument is the direction:
contaminated ≫ clean.)*

## 5. Why this is a strong result, not a disappointment

It's natural to feel the honest R² = 0.22 is "low." But:

- It is **field-normal** for this task (honest single-photo cross-location AQI sits at R² ≈ 0.1–0.35).
- The model still **ranks** pollution well (Spearman 0.65) and gives **calibrated** 90% intervals.
- And we didn't just *assert* the benchmark is inflated — we **measured** the inflation (Δ = 0.539) and
  **proved its cause** (contaminated vs clean).

Exposing leakage in a published benchmark, with a controlled causal check, is a genuine contribution.
The project's deliverable is the **honest number + the measured leakage gap + calibrated intervals +
(coming next) an error ceiling and abstention** — not chasing a score that only leakage can produce.

## 6. How to reproduce

Run `notebooks/09_leakage.ipynb` on a Kaggle GPU as a **Save & Run All (Commit)** job (headless, so it
survives you closing the tab). It trains one model per split, prints the gradient table, computes
Δ_leak, runs the contaminated-vs-clean check, and draws the gradient figure with the 0.55 line. To go
faster, trim `STRATEGIES` to `["random", "station_grouped"]` — that alone gives the core headline.
