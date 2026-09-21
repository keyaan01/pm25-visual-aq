# Phase 7 — C2: the unavoidable-error ceiling

Companion to `notebooks/07_error_ceiling.ipynb`. Maps to **paper §3.10** (contribution C2).

## The idea in one line

If the answer key is itself partly wrong, no model can score full marks — and our answer key
(daily-average AQI) *is* partly wrong for an instantaneous photo.

## Why there's a ceiling

Each PM25Vision label is a **daily-average** AQI. Each photo captures a **single moment**.
Pollution rises and falls through a day, so the instantaneous truth the photo shows can differ
from the daily average. A perfect image model would predict the instant; scored against the
daily average, it still has a residual equal to the within-day deviation. That residual is
**irreducible** — it comes from the labels, not the model.

## The formula and how we estimate it

```
R²_max = 1 − Var(ε) / Var(y)
```

- **Var(y)** — the variance of the daily-average AQI labels in our dataset (we read it straight
  from `df["pm25"].var()`).
- **Var(ε)** — the *average within-day variance* of AQI. We estimate it from **hourly** reference
  data (OpenAQ): for each station-day, convert every hourly µg/m³ reading to AQI, take the
  variance of those hourly AQIs about the day's mean, and average over all station-days.

`src/ceiling.py` splits this cleanly: `within_day_aqi_variance()` and `error_ceiling()` are pure
computation (unit-tested on synthetic data), while `fetch_openaq_hourly()` handles the API.

**We don't need PM25Vision's exact stations.** The quantity we want is the *typical* within-day
AQI swing, so a global sample of ~40 stations over ~45 days is a defensible estimate.

## Getting the data (OpenAQ)

OpenAQ aggregates government air-quality monitors worldwide and offers a free API. You register
at explore.openaq.org, copy an API key, and paste it into the notebook. The code finds PM2.5
sensors, pulls their hourly series, and returns a tidy table. µg/m³ → AQI uses the official EPA
breakpoints in `src/aqi.py`.

## How to read the answer

- **R²_max ≫ 0.55** → real headroom remains above the published baseline.
- **R²_max ≈ 0.55** → the published result may already be near the best possible given
  daily-average labels — a strong, honest finding in itself.
- **Interval-width floor** (~ the 90% band of the label noise) → no honest interval should be
  narrower than this; it's a lower bound on how precise the intervals can legitimately be.

## Caveats (stated honestly in the paper)

- AQI is officially defined on a 24-hour average, so converting *hourly* µg/m³ to an
  "instantaneous AQI" is an approximation — but it's the right proxy for "how much would the
  label move if measured at this instant," and we work in AQI points throughout.
- Within-day variance varies by place and season; we report it as a dataset-wide average and
  note it's an estimate, not an exact per-photo quantity.

## Next

Phase 8 (`08_abstention`) — C3, teaching the model to refuse unusable inputs.
