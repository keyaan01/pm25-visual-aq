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
    df = hourly.copy()
    df["aqi"] = df[ugm3_col].astype(float).map(lambda c: pm25_to_aqi(c, table))
    df["date"] = pd.to_datetime(df[time_col]).dt.date
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
    df = hourly.copy()
    df["aqi"] = df[ugm3_col].astype(float).map(lambda c: pm25_to_aqi(c, table))
    df["date"] = pd.to_datetime(df[time_col]).dt.date
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
    """p(m): fraction of dataset labels in each AQI band (aligned to `edges`)."""
    band = np.digitize(np.asarray(labels, dtype=float), np.asarray(edges[1:-1]))
    counts = np.bincount(band, minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def level_reweighted_var_epsilon(per_day, dataset_labels, edges=AQI_BAND_EDGES):
    """Var(eps)_dataset = sum_m p(m) * v(m), reweighting reference noise to the dataset's own p(m).

    Only bands present in BOTH the reference curve (v) and the dataset (p) are used; p is
    renormalised over those shared bands. Returns (var_eps_reweighted, merged_curve_dataframe).
    """
    curve = conditional_noise_curve(per_day, edges)
    p = _label_hist(dataset_labels, edges)
    curve = curve.copy()
    curve["p_dataset"] = curve["band_idx"].map(lambda b: p[b])
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
    var_eps, curve = level_reweighted_var_epsilon(per_day, dataset_labels, edges)
    devs = within_day_deviations(hourly, station_col, time_col, ugm3_col, min_hours, table)
    out = error_ceiling(var_eps, var_labels, deviations=devs, coverage=coverage)
    out.update(bootstrap_r2max_ci(per_day, dataset_labels, var_labels, edges, n_boot, seed))
    out["n_station_days"] = int(len(per_day))
    out["n_stations"] = int(per_day["station"].nunique()) if len(per_day) else 0
    out["min_hours"] = int(min_hours)
    out["curve"] = curve.to_dict(orient="records")
    return out


# ---------------------------------------------------------------------------
# fetching hourly data from OpenAQ (needs a free API key)
# ---------------------------------------------------------------------------
def _get(url, api_key, params=None, timeout=60):
    r = requests.get(url, headers={"X-API-Key": api_key}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def find_pm25_sensors(api_key, n_locations=60, page_limit=100, max_pages=8, max_per_country=6):
    """Return a list of (location_id, sensor_id, country) for PM2.5 sensors worldwide.

    `max_per_country` caps how many stations come from any one country, so the sample is spread
    across regions (region-stratified) rather than dominated by whichever country OpenAQ lists
    first — a more representative within-day variance estimate.
    """
    found, per_country = [], {}
    for page in range(1, max_pages + 1):
        data = _get(f"{OPENAQ_BASE}/locations", api_key,
                    {"parameters_id": PM25_PARAMETER_ID, "limit": page_limit, "page": page})
        for loc in data.get("results", []):
            country = (loc.get("country") or {}).get("code")
            if per_country.get(country, 0) >= max_per_country:
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


def fetch_openaq_hourly(api_key, date_from, date_to, n_locations=40, pause=0.2,
                        max_per_country=6) -> pd.DataFrame:
    """Collect hourly PM2.5 (ug/m3) for a region-stratified sample of stations. Tidy DataFrame.

    Columns: location_id, country, datetime, pm25_ugm3. Feed this to `compute_ceiling` /
    `within_day_aqi_variance`. Free key: https://explore.openaq.org (Account -> API keys).
    """
    sensors = find_pm25_sensors(api_key, n_locations=n_locations, max_per_country=max_per_country)
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
