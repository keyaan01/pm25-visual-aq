"""Smoke test for C2 (error ceiling) math — CPU, no network, no dataset.

Verifies: 2024 breakpoints selectable + monotonic; within-day variance + min_hours filter;
level-reweighting reproduces a HAND-COMPUTED value; empirical interval floor; error_ceiling as an
upper bound; station cluster-bootstrap CI; and the compute_ceiling one-call path.
Run:  python tests/smoke_ceiling.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ceiling as Cg
from src.aqi import PM25_BREAKPOINTS_2024, pm25_to_aqi


def test_breakpoints_2024():
    # 2024: 9.0 ug/m3 is the AQI-50 boundary; historical table maps 9.0 -> 37.5.
    assert abs(pm25_to_aqi(9.0, PM25_BREAKPOINTS_2024) - 50.0) < 1e-6
    assert abs(pm25_to_aqi(9.0) - 37.5) < 1e-6           # default (historical) differs
    assert abs(pm25_to_aqi(35.4, PM25_BREAKPOINTS_2024) - 100.0) < 1e-6
    # both tables must be monotone non-decreasing in concentration
    for table in (None, PM25_BREAKPOINTS_2024):
        grid = np.linspace(0, 400, 200)
        vals = [pm25_to_aqi(c) if table is None else pm25_to_aqi(c, table) for c in grid]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    print("[aqi]     2024 breakpoints selectable + monotonic OK")


def _synthetic_hourly(n_hours=20, seed=0):
    """Two stations, one day each with n_hours readings (>=18), plus one short day (filtered)."""
    rng = np.random.default_rng(seed)
    rows = []
    specs = [("A", "2024-01-01", 10.0, 3.0, n_hours),   # low band, ug/m3 mean 10 sd 3
             ("B", "2024-01-01", 60.0, 12.0, n_hours),  # higher band, more swing
             ("C", "2024-01-01", 30.0, 5.0, 5)]         # only 5 hours -> dropped by min_hours=18
    for stn, day, mu, sd, nh in specs:
        for h in range(nh):
            rows.append({"location_id": stn, "datetime": f"{day}T{h:02d}:00:00Z",
                         "pm25_ugm3": max(0.1, float(rng.normal(mu, sd)))})
    return pd.DataFrame(rows)


def test_within_day_and_deviations():
    hourly = _synthetic_hourly()
    var_eps, per_day = Cg.within_day_aqi_variance(hourly, min_hours=18)
    assert len(per_day) == 2, f"min_hours filter failed: got {len(per_day)} station-days"
    assert np.isfinite(var_eps) and var_eps > 0
    devs = Cg.within_day_deviations(hourly, min_hours=18)
    assert devs.size == 40 and abs(devs.mean()) < 1e-6   # deviations are centred per day
    print(f"[ceiling] within-day var={var_eps:.1f}, {len(per_day)} station-days, {devs.size} devs OK")


def test_level_reweighting_handcomputed():
    # Hand-built per_day: band 0 has v=100, band 1 has v=400.
    per_day = pd.DataFrame([
        {"station": "A", "date": "d1", "n_hours": 20, "day_mean_aqi": 25.0, "within_day_var": 100.0},
        {"station": "A", "date": "d2", "n_hours": 20, "day_mean_aqi": 25.0, "within_day_var": 100.0},
        {"station": "B", "date": "d1", "n_hours": 20, "day_mean_aqi": 75.0, "within_day_var": 400.0},
        {"station": "B", "date": "d2", "n_hours": 20, "day_mean_aqi": 75.0, "within_day_var": 400.0},
    ])
    # dataset labels: 75% in band 0 (AQI 25), 25% in band 1 (AQI 75)
    labels = [25.0] * 3 + [75.0] * 1
    var_eps, curve = Cg.level_reweighted_var_epsilon(per_day, labels)
    expected = 0.75 * 100.0 + 0.25 * 400.0           # = 175.0
    assert abs(var_eps - expected) < 1e-6, f"reweight got {var_eps}, expected {expected}"
    # unweighted mean would be (100+100+400+400)/4 = 250 -> reweighting must change it
    assert abs(float(per_day["within_day_var"].mean()) - 250.0) < 1e-6
    print(f"[ceiling] level-reweighted Var(eps)={var_eps:.1f} == hand-computed 175.0 OK")


def test_error_ceiling_and_floor():
    devs = np.array([-30, -10, 0, 10, 30, -20, 20, 5, -5, 15])
    out = Cg.error_ceiling(var_epsilon=175.0, var_labels=2500.0, deviations=devs, coverage=0.90)
    assert abs(out["R2_max"] - (1 - 175.0 / 2500.0)) < 1e-9   # 0.93
    assert out["R2_max_is_upper_bound"] is True
    assert abs(out["noise_fraction"] - 0.07) < 1e-9
    assert out["interval_width_floor"] > 0
    # empirical floor = p95 - p5 of the deviations
    assert abs(out["interval_width_floor"] -
               (np.percentile(devs, 95) - np.percentile(devs, 5))) < 1e-9
    print(f"[ceiling] R2_max={out['R2_max']:.3f} (upper bound), floor={out['interval_width_floor']:.1f} OK")


def test_bootstrap_and_compute():
    hourly = _synthetic_hourly(n_hours=20)
    labels = list(np.linspace(5, 120, 200))          # spread across low/mid bands
    _, per_day = Cg.within_day_aqi_variance(hourly, min_hours=18)
    ci = Cg.bootstrap_r2max_ci(per_day, labels, var_labels=2500.0, n_boot=200, seed=1)
    assert 0.0 <= ci["R2_max_lo"] <= ci["R2_max_median"] <= ci["R2_max_hi"] <= 1.0
    full = Cg.compute_ceiling(hourly, labels, var_labels=2500.0, n_boot=100, seed=2)
    for k in ("R2_max", "R2_max_lo", "R2_max_hi", "interval_width_floor", "curve", "n_stations"):
        assert k in full, f"compute_ceiling missing {k}"
    assert 0.0 <= full["R2_max"] <= 1.0 and len(full["curve"]) >= 1
    print(f"[ceiling] bootstrap CI [{ci['R2_max_lo']:.3f},{ci['R2_max_hi']:.3f}] + compute_ceiling OK")


def test_empty_data_clean_error():
    empty = pd.DataFrame(columns=["location_id", "datetime", "pm25_ugm3"])
    _, per_day = Cg.within_day_aqi_variance(empty, min_hours=18)
    curve = Cg.conditional_noise_curve(per_day)       # must NOT raise a KeyError on empty input
    assert len(curve) == 0 and "band_idx" in curve.columns
    try:
        Cg.compute_ceiling(empty, [10.0, 20.0, 30.0], var_labels=2500.0)
        assert False, "expected a ValueError on empty OpenAQ data"
    except ValueError as e:
        assert "station-days" in str(e)
    print("[ceiling] empty-data path raises a clean ValueError (not KeyError) OK")


if __name__ == "__main__":
    test_breakpoints_2024()
    test_within_day_and_deviations()
    test_level_reweighting_handcomputed()
    test_error_ceiling_and_floor()
    test_bootstrap_and_compute()
    test_empty_data_clean_error()
    print("\nALL C2 CEILING SMOKE CHECKS PASSED")
