# Phase 7 — C2: the unavoidable-error ceiling

Companion to `notebooks/07_error_ceiling.ipynb`. Maps to **paper §3.10** (contribution C2).

## The idea in one line

If the answer key is itself partly wrong, no model can score full marks — and our answer key
(daily-average AQI) *is* partly wrong for an instantaneous photo.

## Why there's a ceiling

Each PM25Vision label is a **daily-average** AQI. Each photo captures a **single moment**. Pollution
rises and falls through a day, so the instantaneous truth the photo shows can differ from the daily
average. A perfect image model would predict the instant; scored against the daily average, it still
has a residual equal to the within-day deviation. That residual is **irreducible** — it comes from the
labels, not the model.

## The formula

```
R²_max = 1 − Var(ε) / Var(y)
```

**We report R²_max explicitly as an OPTIMISTIC UPPER bound, and as a RANGE — not one precise number.**
It accounts for **label noise only** (daily averaging). A real model *also* loses accuracy to limited
visual signal, class imbalance, and domain shift — so our honest R² (0.22 single, 0.385 cross-validated)
can sit well below R²_max with nothing wrong. Read the ceiling as "the most any model could hope for on
these labels," not "what we should be getting."

> **Measured value (2026-09-28 re-run, 32 stations / 6 countries / 1,241 station-days).**
> **R²_max ≈ 0.98** — 0.982 (optimistic) to **0.979** (conservative, imputing the uncovered top bands),
> and breakpoint-insensitive (historical 0.982 / 2024 revision 0.980). Do **not** quote the
> cluster-bootstrap band [0.970, 0.993] as a tight 95% CI (only 32 stations). An earlier 13-station pass
> gave a looser 0.977 and we feared the true value might fall to ~0.90–0.95 once high-AQI bands were
> covered — the better-sampled run shows it does **not**: it holds at ~0.98. The top ~13% of the label
> distribution (200+ AQI) is still extrapolated conservatively, which only moves the ceiling to 0.979.

## How we estimate it — three best-practice refinements

**1. Per-hour InstantCast conversion.** For each station-day we convert *every hourly µg/m³ reading*
to AQI, then take the variance of those hourly AQIs about the day's mean. Converting per hour (not
NowCast, not variance in µg/m³ space) matches how the labels themselves are built.

**2. Level-reweighting (the load-bearing fix).** Within-day swing depends on the pollution level, so
a single average is misleading if the reference stations have a different pollution mix than our
photos. We instead estimate the **noise curve** `v(m) = E[Var(ε) | day-mean AQI band]`, then reweight
it to PM25Vision's *own* label histogram `p(m)`:

```
Var(ε)_dataset = Σ_m  p(m) · v(m)
```

So the ceiling reflects the dataset we actually evaluate on. (`level_reweighted_var_epsilon` in
`src/ceiling.py`; the same `v(m)` curve is what explains why error grows in the high-AQI bands.)

**3. Split-specific Var(y).** The ceiling depends on how spread out the *test* labels are, so we use
the **station-grouped test split's** label variance (SD ≈ 108 AQI) — the split we headline — not the
whole-dataset variance.

**4. High-AQI coverage + a conservative estimate.** Within-day swing is *largest* at high AQI, so a
reference sample that misses the high bands understates Var(ε) and inflates R²_max. We now (a) report
`p_mass_covered` — the fraction of the dataset's label mass the noise curve actually covers — and flag
the ceiling **optimistic** when it is below ~0.98; and (b) compute a **conservative** R²_max that imputes
the uncovered high bands with a high v (monotone-extrapolated), giving the lower end of the honest range.
The OpenAQ fetch also now deliberately over-samples high-pollution regions (`HIGH_AQI_COUNTRIES`).

On top of these we report a station cluster-bootstrap band and a **2012-vs-2024 breakpoint sensitivity**
check. **Caveat:** the bootstrap band comes from very few stations (~13 in the first run), where cluster
bootstrap under-covers badly and cannot recover the missing high bands — so treat it as *indicative*,
not a tight 95% CI. Lean on the optimistic→conservative range and the breakpoint sensitivity instead.

`src/ceiling.py` keeps the math (`within_day_aqi_variance`, `conditional_noise_curve`,
`level_reweighted_var_epsilon`, `error_ceiling`, `bootstrap_r2max_ci`, `compute_ceiling`) pure and
unit-tested (`tests/smoke_ceiling.py`), separate from the network code (`fetch_openaq_hourly`).

## The honest interval-width floor

Instead of assuming the noise is Gaussian, we take the **empirical** central-90% spread of the
within-day deviations (the 5th-to-95th percentile of hourly-minus-daily-mean AQI). No honest 90%
interval should be narrower than this — it's a lower bound on how precise the intervals can
legitimately be. (`empirical_interval_floor`.)

## Getting the data (OpenAQ)

OpenAQ aggregates government air-quality monitors worldwide and offers a free API. Register at
explore.openaq.org, copy an API key, and paste it into the notebook. The code finds PM2.5 sensors —
**capping stations per country so the sample spreads across regions** — pulls their hourly series, and
returns a tidy table. Exact matching to PM25Vision's stations isn't needed: we want the *typical*
within-day AQI curve, and a region-stratified sample of ~40 stations over ~45 days is a defensible
estimate.

## How to read the answer

- **The gap 0.22–0.39 → R²_max** is the room that *better vision* could still recover; the gap
  **R²_max → 1.0** is forever lost to daily-average labels.
- **The leakage link (careful — the ceiling is split-specific).** R²_max scales with a split's label
  variance, so compare each split's R² only to *its own* ceiling (`ceiling.per_split_r2max`). The
  leakage proof does **not** rest on "R²(random) > R²_max" (that would need the random split's own
  Var(y)); it rests on the causal contaminated-vs-clean test (0.762 vs −0.86) and the leaky 0.759
  exceeding the whole honest cross-validated range. The notebook still saves `outputs/error_ceiling.json`
  and the leakage-gradient figure draws a ceiling reference line from it.
- **Interval-width floor** → no honest 90% interval should be narrower than the pollution's own
  within-day spread.

## Caveats (stated honestly in the paper)

- Converting *hourly* µg/m³ to an "instantaneous AQI" is an approximation (AQI is officially a
  24-hour construct). Note the **direction of the bias**: hourly readings are themselves 1-hour
  averages, so they *smooth* sub-hourly variation → Var(ε) is under-estimated → R²_max comes out a
  little *too high*. One of the reasons we treat the ceiling as an optimistic bound.
- The first-run sample **under-covered high-AQI conditions**, where within-day swings are largest —
  another upward bias on R²_max. The `p_mass_covered` flag, the conservative estimate, and the
  high-pollution-targeted fetch address it; until the re-run, read R²_max as the *range* above.
- Within-day variance varies by place and season; the level-reweighting + conservative bound +
  breakpoint sensitivity are how we make the ceiling defensible **as a range**, rather than pretending
  a tiny sample gives one precise number.

## Next

Phase 8 (`08_abstention`) — C3, teaching the model to refuse unusable inputs.
