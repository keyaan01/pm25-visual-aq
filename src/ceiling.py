"""C2 — the unavoidable-error ceiling (paper Sec 3.10).

THE PROBLEM
-----------
Our labels are **daily averages**, but each photo is an **instant**. Pollution genuinely
swings within a day, so even a perfect model that read the exact instantaneous pollution
from a photo would be "wrong" versus a daily-average label. That gap is noise no model can
beat — it puts a hard **upper bound** on achievable R2.

THE ESTIMATE
------------
Write y for the daily-average label and y* for the instantaneous truth. The residual a
perfect image model still suffers is the within-day deviation, with variance Var(eps).
The best possible R2 on predicting the labels is therefore:

    R2_max = 1 - Var(eps) / Var(y)

where Var(y) is the variance of the daily-average AQI labels **on the evaluation split**
(the ceiling is split-specific), and Var(eps) is the typical within-day variance of AQI.

We report R2_max explicitly as an **UPPER bound** — it accounts for *label noise only*
(daily-averaging). Real models also lose accuracy to limited visual signal, imbalance, and
domain shift, so the honest R2 can be well below R2_max without any bug.

TWO REFINEMENTS OVER THE NAIVE AVERAGE (best-practice, research-backed):
  1. **Per-hour InstantCast conversion.** We convert each hourly ug/m3 to AQI and take the
     variance of those hourly AQI about the day's mean — matching how WAQI builds the labels
     (NOT NowCast, NOT variance in ug/m3 space).
  2. **Level-reweighting.** Within-day swing depends on the pollution level. We estimate the
     conditional noise curve v(m) = E[Var(eps) | day-mean AQI band] from the reference data,
     then reweight it to PM25Vision's OWN label histogram p(m):
         Var(eps)_dataset = sum_m p(m) * v(m)
     so the ceiling reflects the dataset we actually evaluate on, not the reference sample's mix.

Exact station matching to PM25Vision is NOT required: a regional sample of reference stations
across pollution levels gives a defensible estimate of the within-day variance curve.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .aqi import PM25_BREAKPOINTS, pm25_to_aqi

OPENAQ_BASE = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2   # OpenAQ's parameter id for PM2.5

# AQI bands used for level-reweighting and the v(m) curve (match metrics.error_by_magnitude).
AQI_BAND_EDGES = (0, 50, 100, 150, 200, 300, 10_000)


def _clean_hourly_aqi(hourly, ugm3_col, time_col, table, max_ugm3=1000.0):
    """Return the hourly frame with a finite `aqi` column and a `date` column.

    OpenAQ occasionally returns negative or missing values; those convert to a non-finite AQI and
    would poison the within-day variance / deviations (a single NaN makes the whole day's variance
    NaN and the pooled percentile floor NaN). We drop them here, once, so every downstream group has
    only valid readings and `min_hours` counts VALID hours. We also drop absurd spikes above
    `max_ugm3` (OpenAQ sometimes emits erroneous sentinel/spike values); a single bad spike would
    otherwise blow up a day's within-day variance and destabilise the estimate.
    """
    df = hourly.copy()
    conc = pd.to_numeric(df[ugm3_col], errors="coerce")
    df["aqi"] = [pm25_to_aqi(c, table) if (c is not None and np.isfinite(c) and 0 <= c <= max_ugm3)
                 else float("nan") for c in conc]
    df["date"] = pd.to_datetime(df[time_col], errors="coerce").dt.date
    keep = np.isfinite(pd.to_numeric(df["aqi"], errors="coerce")) & df["date"].notna()
    return df[keep].copy()


# ---------------------------------------------------------------------------
# the math (independently testable — no network needed)
# ---------------------------------------------------------------------------
def within_day_aqi_variance(hourly, station_col="location_id", time_col="datetime",
                            ugm3_col="pm25_ugm3", min_hours=18, table=PM25_BREAKPOINTS):
    """Per-station-day within-day AQI variance, from hourly ug/m3 readings.

    For each station-day with at least `min_hours` readings: convert each hourly ug/m3 to AQI
    (per-hour InstantCast, using `table`), then take the variance of those hourly AQI about the
    day's mean. Returns (var_epsilon_unweighted, per_day_dataframe).

    `min_hours` defaults to 18 (a day with only a handful of readings gives a noisy, biased
    within-day variance; requiring good coverage makes the estimate defensible).

    per_day columns: station, date, n_hours, day_mean_aqi, within_day_var.
    """
    df = _clean_hourly_aqi(hourly, ugm3_col, time_col, table)   # drop invalid/NaN readings first
    rows = []
    for (stn, day), g in df.groupby([station_col, "date"]):
        if len(g) < min_hours:
            continue
        rows.append({"station": stn, "date": day, "n_hours": len(g),
                     "day_mean_aqi": g["aqi"].mean(),
                     "within_day_var": g["aqi"].var(ddof=1)})
    per_day = pd.DataFrame(rows)
    var_eps = float(per_day["within_day_var"].mean()) if len(per_day) else float("nan")
    return var_eps, per_day


def within_day_deviations(hourly, station_col="location_id", time_col="datetime",
                          ugm3_col="pm25_ugm3", min_hours=18, table=PM25_BREAKPOINTS):
    """Pooled (hourly_AQI - day_mean_AQI) deviations across all qualifying station-days.

    Used for the EMPIRICAL interval floor: the honest 90% band cannot be narrower than the actual
    5-95 percentile spread of within-day AQI (this replaces the Gaussian sigma assumption).
    """
    df = _clean_hourly_aqi(hourly, ugm3_col, time_col, table)   # drop invalid/NaN readings first
    devs = []
    for (_stn, _day), g in df.groupby([station_col, "date"]):
        if len(g) < min_hours:
            continue
        devs.append(g["aqi"].to_numpy() - g["aqi"].mean())
    return np.concatenate(devs) if devs else np.array([])


def conditional_noise_curve(per_day, edges=AQI_BAND_EDGES) -> pd.DataFrame:
    """v(m) = mean within-day variance in each day-mean AQI band. Returns a tidy DataFrame.

    Columns: band, band_lo, band_hi, n_days, v (mean within_day_var), rmse_hours (sqrt v).
    """
    cols = ["band", "band_idx", "band_lo", "band_hi", "n_days", "v", "rmse_within_day"]
    if len(per_day) == 0 or "day_mean_aqi" not in per_day.columns:
        return pd.DataFrame(columns=cols)                      # nothing to bin (empty fetch)
    p = per_day.copy()
    p["band"] = np.digitize(p["day_mean_aqi"], np.asarray(edges[1:-1]))
    rows = []
    for b in range(len(edges) - 1):
        g = p[p["band"] == b]
        if len(g) == 0:
            continue
        v = float(g["within_day_var"].mean())
        rows.append({"band": f"{edges[b]}-{edges[b+1] if edges[b+1] < 10_000 else '+'}",
                     "band_idx": b, "band_lo": edges[b], "band_hi": edges[b + 1],
                     "n_days": int(len(g)), "v": v, "rmse_within_day": float(np.sqrt(v))})
    return pd.DataFrame(rows)


def _label_hist(labels, edges=AQI_BAND_EDGES) -> np.ndarray:
    """p(m): fraction of dataset labels in each AQI band (aligned to `edges`).

    NaN labels are dropped first (otherwise np.digitize would silently bin them into the top band).
    """
    arr = np.asarray(labels, dtype=float)
    arr = arr[np.isfinite(arr)]
    band = np.digitize(arr, np.asarray(edges[1:-1]))
    counts = np.bincount(band, minlength=len(edges) - 1).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _impute_uncovered_v(curve, p, edges):
    """Conservative v(m) for dataset bands that carry label mass (p>0) but have NO reference data.

    Uncovered bands are almost always the HIGH-AQI bands, where within-day swings (v) are LARGEST.
    Dropping them (the default renormalisation) understates Var(eps) and so INFLATES R2_max. Here we
    fill each uncovered band with a conservative (high) v by monotone extrapolation of the increasing
    v-vs-band trend among covered bands — never below the largest covered v. Returns {band_idx: v}.
    """
    covered = curve[curve["v"].notna()][["band_idx", "v"]].sort_values("band_idx")
    if len(covered) == 0:
        return {}
    idx = covered["band_idx"].to_numpy(dtype=float)
    v = covered["v"].to_numpy(dtype=float)
    vmax = float(v.max())
    slope = max(float(np.polyfit(idx, v, 1)[0]), 0.0) if len(covered) >= 2 else 0.0
    top_idx, top_v = float(idx[-1]), float(v[-1])
    covered_set = {int(i) for i in idx}
    fill = {}
    for b in range(len(edges) - 1):
        if p[b] > 0 and b not in covered_set:
            fill[b] = max(vmax, top_v + slope * (b - top_idx)) if b > top_idx else vmax
    return fill


def level_reweighted_var_epsilon(per_day, dataset_labels, edges=AQI_BAND_EDGES,
                                 impute_uncovered=False):
    """Var(eps)_dataset = sum_m p(m) * v(m), reweighting reference noise to the dataset's own p(m).

    `impute_uncovered=False` (default): only bands present in BOTH the reference curve (v) and the
    dataset (p) are used, and p is renormalised over those covered bands. This is the OPTIMISTIC
    estimate — it silently drops any high-AQI band the reference sample missed, understating Var(eps)
    and inflating R2_max.

    `impute_uncovered=True`: uncovered dataset bands are filled with a conservative (high) v via
    `_impute_uncovered_v`, and p is reweighted over ALL bands with label mass. This is the tighter,
    more honest CONSERVATIVE estimate (a lower R2_max). Report the two as a range.

    Returns (var_eps_reweighted, merged_curve_dataframe).
    """
    curve = conditional_noise_curve(per_day, edges)
    p = _label_hist(dataset_labels, edges)
    curve = curve.copy()
    curve["p_dataset"] = curve["band_idx"].map(lambda b: p[b])
    if impute_uncovered:
        vmap = {int(r.band_idx): float(r.v) for _, r in curve.iterrows() if np.isfinite(r.v)}
        vmap.update(_impute_uncovered_v(curve, p, edges))
        bands = [b for b in range(len(edges) - 1) if p[b] > 0 and b in vmap]
        if not bands:
            return float("nan"), curve
        w = np.array([p[b] for b in bands], dtype=float)
        w = w / w.sum()
        var_eps = float(np.sum(w * np.array([vmap[b] for b in bands], dtype=float)))
        return var_eps, curve
    shared = curve[curve["p_dataset"] > 0]
    if shared["p_dataset"].sum() == 0 or len(shared) == 0:
        return float("nan"), curve
    w = shared["p_dataset"] / shared["p_dataset"].sum()      # renormalise over covered bands
    var_eps = float((w * shared["v"]).sum())
    curve["p_reweighted"] = curve["band_idx"].map(
        lambda b: (p[b] / shared["p_dataset"].sum()) if b in set(shared["band_idx"]) else 0.0)
    return var_eps, curve


def empirical_interval_floor(deviations, coverage=0.90) -> float:
    """Empirical central-`coverage` width of the within-day deviations (the honest interval floor).

    For coverage=0.90 this is percentile(95) - percentile(5) of (hourly - day_mean) — no Gaussian
    assumption. An honest interval cannot be narrower than this pure label-noise spread.
    """
    d = np.asarray(deviations, dtype=float)
    if d.size == 0:
        return float("nan")
    lo = (1.0 - coverage) / 2.0 * 100.0
    return float(np.percentile(d, 100.0 - lo) - np.percentile(d, lo))


def error_ceiling(var_epsilon: float, var_labels: float, deviations=None,
                  coverage: float = 0.90) -> dict:
    """R2 upper bound + honest interval-width floor from the label-noise and label variance.

    var_labels : variance of the daily-average AQI labels ON THE EVALUATION SPLIT (split-specific).
    deviations : optional pooled within-day deviations (from `within_day_deviations`) for the
                 empirical interval floor; if omitted, falls back to the Gaussian approximation.
    """
    r2_max_raw = 1.0 - var_epsilon / var_labels
    if deviations is not None and np.asarray(deviations).size:
        floor = empirical_interval_floor(deviations, coverage)
    else:
        z = 1.6448536269514722  # 90% two-sided normal
        floor = float(2 * z * np.sqrt(var_epsilon))
    return {
        "var_epsilon": float(var_epsilon),
        "var_labels": float(var_labels),
        "R2_max": float(np.clip(r2_max_raw, 0.0, 1.0)),
        "R2_max_raw": float(r2_max_raw),
        "R2_max_is_upper_bound": True,
        "noise_fraction": float(var_epsilon / var_labels),
        "interval_width_floor": float(floor),
        "interpretation": ("R2_max is an UPPER bound from daily-average label noise only; the honest "
                           "model R2 can be lower due to limited visual signal, imbalance and shift."),
    }


def bootstrap_r2max_ci(per_day, dataset_labels, var_labels, edges=AQI_BAND_EDGES,
                       n_boot=1000, seed=0, level_reweight=True, ci=0.95) -> dict:
    """Station cluster-bootstrap CI on R2_max (resample whole stations, not station-days).

    Resampling by station (the cluster) respects the dependence between a station's days and gives
    an honest uncertainty band on the ceiling.
    """
    rng = np.random.default_rng(seed)
    stations = per_day["station"].unique()
    by_station = {s: per_day[per_day["station"] == s] for s in stations}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(stations, size=len(stations), replace=True)
        boot = pd.concat([by_station[s] for s in pick], ignore_index=True)
        if level_reweight:
            var_eps, _ = level_reweighted_var_epsilon(boot, dataset_labels, edges)
        else:
            var_eps = float(boot["within_day_var"].mean())
        if np.isfinite(var_eps):
            vals.append(np.clip(1.0 - var_eps / var_labels, 0.0, 1.0))
    vals = np.asarray(vals)
    lo = (1.0 - ci) / 2.0 * 100.0
    return {"R2_max_median": float(np.median(vals)) if vals.size else float("nan"),
            "R2_max_lo": float(np.percentile(vals, lo)) if vals.size else float("nan"),
            "R2_max_hi": float(np.percentile(vals, 100 - lo)) if vals.size else float("nan"),
            "n_boot": int(vals.size)}


def compute_ceiling(hourly, dataset_labels, var_labels, station_col="location_id",
                    time_col="datetime", ugm3_col="pm25_ugm3", min_hours=18,
                    table=PM25_BREAKPOINTS, edges=AQI_BAND_EDGES, n_boot=1000, seed=0,
                    coverage=0.90) -> dict:
    """One-call C2 estimate: level-reweighted Var(eps), split-specific R2_max, CI, floor, and curve.

    Returns a dict ready to json.dump to `outputs/error_ceiling.json` (so the leakage figure can
    draw the ceiling line) plus the v(m) curve for the error-by-band story.
    """
    _, per_day = within_day_aqi_variance(hourly, station_col, time_col, ugm3_col, min_hours, table)
    if len(per_day) == 0:
        raise ValueError(
            "No station-days had >= %d hourly readings, so the ceiling can't be estimated. "
            "Check that Internet is On and the OpenAQ key is valid, widen the date window, "
            "or lower min_hours." % min_hours)
    var_eps, curve = level_reweighted_var_epsilon(per_day, dataset_labels, edges)          # optimistic
    var_eps_cons, _ = level_reweighted_var_epsilon(per_day, dataset_labels, edges,
                                                   impute_uncovered=True)                  # conservative
    devs = within_day_deviations(hourly, station_col, time_col, ugm3_col, min_hours, table)
    out = error_ceiling(var_eps, var_labels, deviations=devs, coverage=coverage)
    out.update(bootstrap_r2max_ci(per_day, dataset_labels, var_labels, edges, n_boot, seed))
    out["n_station_days"] = int(len(per_day))
    out["n_stations"] = int(per_day["station"].nunique()) if len(per_day) else 0
    out["min_hours"] = int(min_hours)
    out["curve"] = curve.to_dict(orient="records")
    # How much of the dataset's label distribution the noise curve actually covers. If < ~1.0, the
    # high-AQI bands (largest within-day swings) are under-sampled, so R2_max is a LOOSER upper bound.
    p = _label_hist(dataset_labels, edges)
    covered = set(int(b) for b in curve.loc[curve["v"].notna(), "band_idx"]) if len(curve) else set()
    out["p_mass_covered"] = float(sum(p[b] for b in range(len(p)) if b in covered))
    # Honest RANGE: R2_max is the OPTIMISTIC end (drops uncovered high bands); R2_max_conservative
    # imputes them with a high v (a tighter, lower ceiling). The true ceiling lies between. When
    # coverage is incomplete we flag the point estimate as optimistic so no one quotes it as precise.
    if np.isfinite(var_eps_cons):
        out["var_epsilon_conservative"] = float(var_eps_cons)
        out["R2_max_conservative"] = float(np.clip(1.0 - var_eps_cons / var_labels, 0.0, 1.0))
    out["ceiling_is_optimistic"] = bool(out["p_mass_covered"] < 0.98)
    return out


def per_split_r2max(var_epsilon: float, var_labels_by_split: dict) -> dict:
    """R2_max for each split from its OWN label variance (the ceiling is split-specific).

    var_labels_by_split : {split_name: Var(y_on_that_split)}, e.g. from each split/fold's SD_y**2.
    A split's R2 must only ever be compared to ITS OWN ceiling — a single global R2_max is misleading
    because R2_max scales with the split's label spread.
    """
    return {s: float(np.clip(1.0 - var_epsilon / vy, 0.0, 1.0))
            for s, vy in var_labels_by_split.items() if vy and np.isfinite(vy) and vy > 0}


# ---------------------------------------------------------------------------
# fetching hourly data from OpenAQ (needs a free API key)
# ---------------------------------------------------------------------------
def _get(url, api_key, params=None, timeout=60):
    r = requests.get(url, headers={"X-API-Key": api_key}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# Countries that routinely reach the HIGH-AQI bands (150-500+), where within-day PM2.5 swings are
# largest. Sampling these deliberately (not just "whatever OpenAQ lists first") is what makes the
# v(m) curve cover the high bands, so R2_max stops being an optimistic upper bound. Not exhaustive —
# a pragmatic AQI-level stratification for the free OpenAQ tier.
HIGH_AQI_COUNTRIES = ("IN", "PK", "BD", "NP", "CN", "MN", "ID", "VN", "IR", "IQ", "AE", "KW", "US")


def find_pm25_sensors(api_key, n_locations=80, page_limit=100, max_pages=15, max_per_country=10,
                      priority_countries=HIGH_AQI_COUNTRIES, priority_max_per_country=None):
    """Return a list of (location_id, sensor_id, country) for PM2.5 sensors worldwide.

    `max_per_country` caps how many stations come from any one country, so the sample is spread
    across regions (region-stratified) rather than dominated by whichever country OpenAQ lists first.

    `priority_countries` (high-pollution regions) get a HIGHER cap (`priority_max_per_country`,
    default 2x) so the sample deliberately reaches the high-AQI bands — an AQI-LEVEL stratification,
    not just a geographic one. This directly addresses the ceiling's high-band under-coverage: without
    it the reference sample skews low-AQI and R2_max comes out too optimistic.
    """
    priority = set(priority_countries or ())
    pcap = priority_max_per_country if priority_max_per_country is not None else max(max_per_country * 2, max_per_country)
    found, per_country = [], {}
    for page in range(1, max_pages + 1):
        data = _get(f"{OPENAQ_BASE}/locations", api_key,
                    {"parameters_id": PM25_PARAMETER_ID, "limit": page_limit, "page": page})
        for loc in data.get("results", []):
            country = (loc.get("country") or {}).get("code")
            cap = pcap if country in priority else max_per_country
            if per_country.get(country, 0) >= cap:
                continue
            for s in loc.get("sensors", []):
                if s.get("parameter", {}).get("id") == PM25_PARAMETER_ID:
                    found.append((loc["id"], s["id"], country))
                    per_country[country] = per_country.get(country, 0) + 1
                    break
            if len(found) >= n_locations:
                return found
        if not data.get("results"):
            break
    return found


def fetch_sensor_hours(api_key, sensor_id, date_from, date_to, limit=1000):
    """Fetch hourly aggregated values for one sensor over a date range."""
    data = _get(f"{OPENAQ_BASE}/sensors/{sensor_id}/hours", api_key,
                {"datetime_from": date_from, "datetime_to": date_to, "limit": limit})
    out = []
    for row in data.get("results", []):
        val = row.get("value")
        period = (row.get("period") or {}).get("datetimeFrom", {})
        dt = period.get("utc") if isinstance(period, dict) else None
        if val is not None and dt is not None:
            out.append({"datetime": dt, "pm25_ugm3": val})
    return out


def fetch_openaq_hourly(api_key, date_from, date_to, n_locations=80, pause=0.2,
                        max_per_country=10, priority_countries=HIGH_AQI_COUNTRIES,
                        priority_max_per_country=None) -> pd.DataFrame:
    """Collect hourly PM2.5 (ug/m3) for a region- AND AQI-level-stratified sample of stations.

    Columns: location_id, country, datetime, pm25_ugm3. Feed this to `compute_ceiling` /
    `within_day_aqi_variance`. `priority_countries` get a higher per-country cap so the sample
    reaches the high-AQI bands (see `find_pm25_sensors`). Free key: https://explore.openaq.org
    (Account -> API keys).
    """
    sensors = find_pm25_sensors(api_key, n_locations=n_locations, max_per_country=max_per_country,
                                priority_countries=priority_countries,
                                priority_max_per_country=priority_max_per_country)
    frames = []
    for loc_id, sensor_id, country in sensors:
        try:
            rows = fetch_sensor_hours(api_key, sensor_id, date_from, date_to)
        except Exception:
            continue
        for r in rows:
            r.update({"location_id": loc_id, "country": country})
        frames.extend(rows)
        time.sleep(pause)   # be polite to the API
    return pd.DataFrame(frames, columns=["location_id", "country", "datetime", "pm25_ugm3"])
