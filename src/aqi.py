"""EPA PM2.5 <-> AQI conversion.

WHY THIS EXISTS
---------------
The PM25Vision label (`pm25`) is already a US-EPA **Air Quality Index** value
(a unitless 0-500+ index), NOT a raw concentration in ug/m3. So the core pipeline
does not need any conversion -- it trains and reports directly in AQI points.

We need this module for contribution **C2 (the error ceiling)**: to estimate how
much pollution naturally swings *within a day*, we download hourly readings from
OpenAQ, which are given in ug/m3. To compare those against the daily-average AQI
labels, we convert the hourly ug/m3 values into AQI using the official EPA
piecewise-linear breakpoint table below.

The mapping is piecewise-linear: each band has its own slope, so you cannot just
multiply -- you find which band a value falls in and apply that band's formula.
"""
from __future__ import annotations

# Official EPA PM2.5 (24-hour) breakpoints:  (C_low, C_high, AQI_low, AQI_high)
# Concentrations in ug/m3. This is the long-standing table WAQI used to build the
# PM25Vision labels over 2014-2025. (EPA revised the lowest bands in 2024; we keep
# the historical table to match how the dataset's labels were produced.)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(conc: float) -> float:
    """Convert a PM2.5 concentration (ug/m3) to an AQI index value.

    Values above the top breakpoint are linearly extrapolated from the last band
    (the dataset contains AQI values up to 530, i.e. slightly beyond the standard
    500 ceiling), so we do not clamp.
    """
    if conc is None or conc < 0:
        return float("nan")
    for c_lo, c_hi, a_lo, a_hi in PM25_BREAKPOINTS:
        if conc <= c_hi:
            return (a_hi - a_lo) / (c_hi - c_lo) * (conc - c_lo) + a_lo
    # above the table: extrapolate using the slope of the last band
    c_lo, c_hi, a_lo, a_hi = PM25_BREAKPOINTS[-1]
    return (a_hi - a_lo) / (c_hi - c_lo) * (conc - c_lo) + a_lo


def aqi_to_pm25(aqi: float) -> float:
    """Convert an AQI index value back to an estimated PM2.5 concentration (ug/m3).

    Inverse of `pm25_to_aqi`. Useful if we ever want concentrations; note it
    introduces distortion near band boundaries (mentioned in the guide).
    """
    if aqi is None or aqi < 0:
        return float("nan")
    for c_lo, c_hi, a_lo, a_hi in PM25_BREAKPOINTS:
        if aqi <= a_hi:
            return (c_hi - c_lo) / (a_hi - a_lo) * (aqi - a_lo) + c_lo
    c_lo, c_hi, a_lo, a_hi = PM25_BREAKPOINTS[-1]
    return (c_hi - c_lo) / (a_hi - a_lo) * (aqi - a_lo) + c_lo


# The six EPA AQI categories, used to report classification-style accuracy
# (paper Sec 3.11) and to describe how wide an interval is in "category" terms.
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 10_000, "Hazardous"),
]


def aqi_category(aqi: float) -> str:
    """Return the EPA category name for an AQI value."""
    for lo, hi, name in AQI_CATEGORIES:
        if aqi <= hi:
            return name
    return "Hazardous"


def aqi_category_index(aqi: float) -> int:
    """Return the EPA category as an integer 0..5 (for classification metrics)."""
    for i, (lo, hi, _name) in enumerate(AQI_CATEGORIES):
        if aqi <= hi:
            return i
    return len(AQI_CATEGORIES) - 1
