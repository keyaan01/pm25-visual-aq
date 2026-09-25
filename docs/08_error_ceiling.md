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

**We report R²_max explicitly as an UPPER bound.** It accounts for **label noise only** (daily
averaging). A real model *also* loses accuracy to limited visual signal, class imbalance, and
domain shift — so our honest R² (0.22) can sit well below R²_max with nothing wrong. Read the ceiling
as "the most any model could hope for on these labels," not "what we should be getting."

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

On top of these we report a **station cluster-bootstrap 95% CI** on R²_max (resampling whole
stations, which respects the dependence between a station's days) and a **2012-vs-2024 breakpoint
sensitivity** check, so the number is shown to be robust rather than a single fragile point estimate.

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

- **The gap 0.22 → R²_max** is the room that *better vision* could still recover; the gap
  **R²_max → 1.0** is forever lost to daily-average labels.
- **The leakage link:** if `09_leakage`'s random-split R² (0.759) is **above** R²_max, that split is
  *provably* leaky — no honest model can beat the label-noise ceiling. The notebook saves
  `outputs/error_ceiling.json`, and the leakage-gradient figure draws its ceiling line from it.
- **Interval-width floor** → no honest 90% interval should be narrower than the pollution's own
  within-day spread.

## Caveats (stated honestly in the paper)

- Converting *hourly* µg/m³ to an "instantaneous AQI" is an approximation (AQI is officially a
  24-hour construct), but it's the right proxy for "how much would the label move if measured at this
  instant," and we work in AQI points throughout.
- Within-day variance varies by place and season; the level-reweighting + bootstrap CI + breakpoint
  sensitivity are exactly how we make the single reported number defensible rather than fragile.

## Next

Phase 8 (`08_abstention`) — C3, teaching the model to refuse unusable inputs.
