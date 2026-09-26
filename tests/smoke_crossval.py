"""Smoke test for leakage-safe grouped cross-validation (src/crossval.py), CPU only.

  python tests/_build_fixture.py     # once
  python tests/smoke_crossval.py

Checks: (1) folds are station-disjoint and every station is tested exactly once; (2) summarize_cv
gives a sane mean/CI; (3) run_cv chains end-to-end on the fixture (2 folds, 1 epoch, no pretrained).
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import crossval as CV, data, physics  # noqa: E402
from src.config import load_config  # noqa: E402

FIX = str(Path(__file__).resolve().parent / "fixture_ds")


def test_folds_are_leakage_safe():
    rows = [{"station_id": f"st{s}", "pm25": 40 + s, "_row": s * 3 + j}
            for s in range(30) for j in range(3)]           # 30 stations x 3 photos
    df = pd.DataFrame(rows)
    folds = CV.make_cv_folds(df, n_splits=5, seed=0, station_col="station_id")
    assert len(folds) == 5
    tested = set()
    for sp in folds:
        tr = set(sp.loc[sp.split == "train", "station_id"])
        ca = set(sp.loc[sp.split == "cal", "station_id"])
        te = set(sp.loc[sp.split == "test", "station_id"])
        assert tr.isdisjoint(te) and ca.isdisjoint(te) and tr.isdisjoint(ca), "folds must be station-disjoint"
        assert len(te) > 0 and 0.1 < len(sp[sp.split == "test"]) / len(sp) < 0.3
        tested |= te
    assert tested == {f"st{s}" for s in range(30)}, "every station must be tested exactly once"
    print("[crossval] 5 folds station-disjoint; every station tested exactly once OK")


def test_summarize():
    reports = [{"R2": 0.20, "MAE": 50, "RMSE": 90, "r2_pearson": 0.21, "Spearman": 0.6, "coverage": 0.90},
               {"R2": 0.24, "MAE": 54, "RMSE": 96, "r2_pearson": 0.25, "Spearman": 0.66, "coverage": 0.88}]
    s = CV.summarize_cv(reports)
    assert abs(s["R2"]["mean"] - 0.22) < 1e-9
    assert s["R2"]["ci_lo"] < s["R2"]["mean"] < s["R2"]["ci_hi"]
    print("[crossval] summarize_cv mean=%.3f CI=[%.3f,%.3f] OK"
          % (s["R2"]["mean"], s["R2"]["ci_lo"], s["R2"]["ci_hi"]))


def test_run_cv_fixture():
    cfg = load_config()
    cfg["train"]["batch_size"] = 4
    cfg["model"]["pretrained"] = False          # no network download in the smoke test
    ds, df = data.load_clean(FIX, from_disk=True, seed=42)
    tmp = tempfile.mkdtemp()
    cpath = os.path.join(tmp, "maps.npy")
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cpath, size=224, progress=False)
    cache = physics.load_map_cache(cpath)
    reports, summary = CV.run_cv(ds, df, cache, cfg, device="cpu", out_root=tmp,
                                 n_splits=2, seed=0, num_workers=0, max_epochs=1, verbose=False)
    assert len(reports) == 2 and all(r["straddling_stations"] == 0 for r in reports)
    assert "R2" in summary and summary["R2"]["n"] == 2
    print("[crossval] run_cv 2-fold fixture run OK (R2 mean=%.3f, folds straddling=0)"
          % summary["R2"]["mean"])


if __name__ == "__main__":
    test_folds_are_leakage_safe()
    test_summarize()
    test_run_cv_fixture()
    print("\nALL CROSSVAL SMOKE CHECKS PASSED")
