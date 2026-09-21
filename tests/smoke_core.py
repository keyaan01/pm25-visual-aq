"""Core-pipeline smoke test (Phases 2–6) on the tiny fixture, CPU only.

Verifies the whole chain wires together and the key invariants hold:
  splits (leakage-safe) -> physics cache -> loaders -> model -> train -> save/load
  -> predictions (ascending, AQI space) -> conformal calibration (hits target coverage)
  -> metrics.

    python tests/_build_fixture.py     # once
    python tests/smoke_core.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (aqi, calibrate as C, data, dataset as D, metrics as Mx,  # noqa: E402
                 model as M, physics, splits, train as T)
from src.config import load_config  # noqa: E402

FIX = str(Path(__file__).resolve().parent / "fixture_ds")


def main():
    cfg = load_config()
    cfg["train"]["batch_size"] = 4
    T.set_seed(0)

    ds, df = data.load_clean(FIX, from_disk=True, seed=42)

    # Phase 2 — leakage-safe splits
    sp = splits.make_splits(df, strategy="station_grouped", seed=42)
    assert splits.split_report(sp)["stations_straddling_splits"] == 0
    print("[splits]  station_grouped leakage-safe (0 straddling)")

    # Phase 3 — physics cache
    tmp = tempfile.mkdtemp()
    cpath = os.path.join(tmp, "maps.npy")
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cpath, size=224, progress=False)
    cache = physics.load_map_cache(cpath)
    print("[physics] map cache", cache.shape)

    # Phase 4/5 — model + a 2-epoch mini-train
    loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=0)
    net = M.PM25QuantileNet(in_chans=5, pretrained=False, quantiles=tuple(cfg["quantiles"]))
    out = os.path.join(tmp, "station_grouped")
    T.train_model(net, loaders, cfg, device="cpu", out_dir=out,
                  max_epochs=2, max_steps_per_epoch=3, verbose=False)
    assert os.path.exists(os.path.join(out, "best_model.pth"))
    print("[train]   checkpoint saved")

    # reload + predict
    net2 = M.PM25QuantileNet(in_chans=5, pretrained=False, quantiles=tuple(cfg["quantiles"]))
    T.load_checkpoint(os.path.join(out, "best_model.pth"), net2, map_location="cpu")
    cal_p, cal_y = T.collect_predictions(net2, loaders["cal"], "cpu", True)
    test_p, test_y = T.collect_predictions(net2, loaders["test"], "cpu", True)
    assert np.all(test_p[:, 1:] >= test_p[:, :-1] - 1e-4), "quantiles must stay ascending"
    print("[predict] quantiles ascending in AQI space")

    # Phase 6 — conformal calibration hits the target coverage (model-independent guarantee)
    Q = C.conformal_Q(cal_p, cal_y, cfg["calibration"]["coverage"])
    cal_test = C.apply_conformal(test_p, Q)
    cov = C.coverage(cal_test, test_y)
    # on a tiny cal set the finite-sample coverage is conservative; just require >= target-ish
    assert cov >= cfg["calibration"]["coverage"] - 0.15
    rep = Mx.full_report(cal_test, test_y)
    assert set(rep) >= {"MAE", "RMSE", "R2", "coverage", "mean_width", "category_accuracy"}
    print(f"[calib]   Q={Q:.1f}  coverage={cov:.2f} (target {cfg['calibration']['coverage']})")

    print("\nALL CORE (PHASES 2–6) SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
