"""C2 — the unavoidable-error ceiling (paper Sec 3.10).

THE PROBLEM
-----------
Our labels are **daily averages**, but each photo is an **instant**. Pollution genuinely
swings within a day, so even a perfect model that read the exact instantaneous pollution
from a photo would be "wrong" versus a daily-average label. That gap is noise no model can
beat — it puts a hard ceiling on achievable R2.

THE ESTIMATE
------------
Write y for the daily-average label and y* for the instantaneous truth. The residual a
perfect image model still suffers is the within-day deviation, with variance Var(eps).
The best possible R2 on predicting the labels is therefore:

    R2_max = 1 - Var(eps) / Var(y)          (equivalently Var(y*)/(Var(y*)+Var(eps)))

where Var(y) is the variance of the daily-average AQI labels in our dataset, and Var(eps)
is the *average within-day variance* of AQI, estimated from hourly reference measurements.

We get hourly PM2.5 from OpenAQ (which reports µg/m³), convert each hourly reading to AQI,
and for every station-day measure how much the hourly AQI swings about that day's mean.

Exact station matching to PM25Vision is NOT required: we need a defensible estimate of the
typical within-day AQI variance, so a regional sample of stations across pollution levels
suffices.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .aqi import pm25_to_aqi

OPENAQ_BASE = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2   # OpenAQ's parameter id for PM2.5


# ---------------------------------------------------------------------------
# the math (independently testable — no network needed)
# ---------------------------------------------------------------------------
def within_day_aqi_variance(hourly, station_col="location_id", time_col="datetime",
                            ugm3_col="pm25_ugm3", min_hours=6):
    """Average within-day variance of AQI, from hourly µg/m³ readings.

    For each station-day with at least `min_hours` readings: convert each hourly µg/m³ to
    AQI, then take the variance of those hourly AQI values about the day's mean. Average
    those variances across all station-days. Returns (var_epsilon, per_day_dataframe).
    """
    df = hourly.copy()
    df["aqi"] = df[ugm3_col].astype(float).map(pm25_to_aqi)
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


def error_ceiling(var_epsilon: float, var_labels: float) -> dict:
    """Turn the noise + label variance into the R2 ceiling and an honest width floor.

    var_labels : variance of the daily-average AQI labels in the dataset (e.g. df.pm25.var()).
    Returns R2_max, the fraction of variance the noise consumes, and a floor on how narrow
    an honest interval can be (~ the label noise standard deviation).
    """
    r2_max = 1.0 - var_epsilon / var_labels
    return {
        "var_epsilon": var_epsilon,
        "var_labels": var_labels,
        "R2_max": float(np.clip(r2_max, 0.0, 1.0)),
        "noise_fraction": float(var_epsilon / var_labels),
        "interval_width_floor": float(2 * 1.645 * np.sqrt(var_epsilon)),  # ~90% band of the noise
    }


# ---------------------------------------------------------------------------
# fetching hourly data from OpenAQ (needs a free API key)
# ---------------------------------------------------------------------------
def _get(url, api_key, params=None, timeout=60):
    r = requests.get(url, headers={"X-API-Key": api_key}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def find_pm25_sensors(api_key, n_locations=60, page_limit=100, max_pages=5):
    """Return a list of (location_id, sensor_id, country) for PM2.5 sensors worldwide."""
    found = []
    for page in range(1, max_pages + 1):
        data = _get(f"{OPENAQ_BASE}/locations", api_key,
                    {"parameters_id": PM25_PARAMETER_ID, "limit": page_limit, "page": page})
        for loc in data.get("results", []):
            for s in loc.get("sensors", []):
                if s.get("parameter", {}).get("id") == PM25_PARAMETER_ID:
                    found.append((loc["id"], s["id"],
                                  (loc.get("country") or {}).get("code")))
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


def fetch_openaq_hourly(api_key, date_from, date_to, n_locations=40, pause=0.2) -> pd.DataFrame:
    """Collect hourly PM2.5 (µg/m³) for a sample of stations. Returns a tidy DataFrame.

    Columns: location_id, country, datetime, pm25_ugm3. Feed this to
    `within_day_aqi_variance`. Get a free key at https://explore.openaq.org (Account → API keys).
    """
    sensors = find_pm25_sensors(api_key, n_locations=n_locations)
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
