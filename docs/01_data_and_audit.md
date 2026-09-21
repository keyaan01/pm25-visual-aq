# Phase 1 — Data loading, cleaning, and auditing

Companion to `notebooks/01_data_audit.ipynb`. Maps to **paper §3.2–3.3**.

## Why this phase exists

A model is only as good as the data under it. Before writing a single line of model
code, we (1) understand the dataset, (2) fix its defects, and (3) run checks that
would catch a broken dataset *before* it wastes days of training. The paper calls this
"auditing the data before modelling," and treats a failed audit as a reason to stop.

## The dataset in one paragraph

**PM25Vision**: 11,219 street-level photos, each paired with a **daily-average US-EPA
AQI value** (the label, column `pm25`, range 1–530). Each photo also has a
`station_id` (which air-quality monitor it was matched to), a `captured_at` date,
and GPS `longitude`/`latitude`. The photos come from **Mapillary** (crowdsourced
street imagery) and the labels from **WAQI** (World Air Quality Index). 3,261
distinct stations are represented.

> **Units matter.** The label is an **AQI *index*** (unitless, 1–530), **not** a
> concentration in µg/m³. Everything we report later is in **AQI points**. Mixing
> these up is the single most common mistake with this dataset.

## What `data.py` cleans, and why

Our vetting of the raw data turned up three issues; `clean_metadata` fixes all three:

| Issue we found | Fix | Why it matters |
|----------------|-----|----------------|
| `camera_angle` and `quality_score` are **null in every row**; `downloaded_at` is just the scrape date; `filename` duplicates `image_id`; `quality` is `"good"` for every row. | Drop these dead columns. | They carry no signal; keeping them invites accidental misuse. |
| **123 duplicate `image_id` rows.** | Drop duplicates (keep first). | Exact duplicates would inflate scores and could straddle a split. |
| Rows are **stored in order** (a contiguous block is all one AQI band / station). | Shuffle with a fixed seed. | A naive "first 80% / last 20%" split would be wildly biased. |

We also **pool the shipped train and test splits into one pool**, because the paper
makes its *own* splits (by station, region, time) in Phase 2 — the dataset's built-in
split is not what we evaluate on.

**Design note (memory):** the JPEG bytes stay inside the Hugging Face `Dataset`
object; we only pull a lightweight table of *metadata* into pandas for auditing. Each
metadata row keeps a `_row` pointer, so `get_image(ds, row)` fetches the picture on
demand instead of holding 11k images in RAM.

## The three audit checks

### 1. Near-duplicate redundancy (perceptual hashing)
Consecutive street photos look near-identical. A **perceptual hash** is a fingerprint
where *similar images get similar fingerprints* (unlike a normal hash, which changes
completely for a one-pixel difference). We fingerprint every image and union ones
whose fingerprints differ by ≤ 5 bits, then report the **distinct-ratio** =
groups ÷ images. Close to 1.0 = little redundancy; lower = near-duplicates we must
keep out of the split (Phase 2 handles that by grouping on `station_id`).

### 2. Images per station
**Result from the full dataset:** 3,261 stations; median **2** images each, mean 3.4,
max **91**; ~46% of stations have only a single image. This means a **station-grouped
split is trivially feasible** (thousands of groups to distribute), and that
near-duplicate risk is concentrated in the few busy stations (60–91 images).

### 3. Physics falsification test
Physics says a hazier photo has **lower transmission** (less of the scene's light
survives the trip to the camera) and a **higher AQI**. So across the dataset, average
transmission should be **negatively correlated** with the label. We compute a
transmission scalar per image (via the Dark Channel Prior — see Phase 3) and correlate
it with `pm25`. A clearly negative correlation confirms images and labels are paired
correctly. **If this test failed, we would stop and fix the pairing** — a broken
answer key can't be learned, and we'd want to know now, not after training.

## An honest finding that shapes the write-up

The proposal assumes the *published* split is naively random and therefore leaks
near-duplicates. We checked: the shipped train/test split actually shares **zero
stations** — it is already station-disjoint. So we cannot call the shipped split
"leaky." Instead, in Phase 2 we **construct our own random-mixed split** as the leaky
control and measure the gap against the grouped split. This keeps the leakage
contribution honest: we *measure* the inflation rather than *assume* it.

## Geography

Coverage is concentrated in **East Asia, Europe, and India**, with little in the
Americas or Africa. Any claim about generalisation must therefore be scoped to the
regions actually present — which is exactly why Phase 2 includes a geographic
hold-out split.

## What's next

Phase 2 (`02_splits`) turns this cleaned, audited data into leakage-safe
train / calibration / test splits.
