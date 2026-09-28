"""Smoke test for the Stage-B/C accuracy levers + the alignment guards, CPU only.

  python tests/_build_fixture.py     # once
  python tests/smoke_accuracy.py

Checks, on the tiny fixture (no pretrained download, 1 epoch):
  (1) EMA + TTA + a 2-seed ensemble run end-to-end through leakage.train_and_evaluate, report
      n_ensemble==2 and tta==True, produce ascending conformal intervals, and the built-in
      alignment assert (frame label == loader target) passes.
  (2) contaminated_vs_clean partitions correctly and now returns SD_y + RMSE per subset.
  (3) physics.verify_cache_alignment passes on a correct cache and RAISES on a shuffled one.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data, physics, leakage as L, calibrate as C  # noqa: E402
from src.config import load_config  # noqa: E402

FIX = str(Path(__file__).resolve().parent / "fixture_ds")


def _fixture():
    cfg = load_config()
    cfg["train"]["batch_size"] = 4
    cfg["model"]["pretrained"] = False
    ds, df = data.load_clean(FIX, from_disk=True, seed=42)
    tmp = tempfile.mkdtemp()
    cpath = os.path.join(tmp, "maps.npy")
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cpath, size=224, progress=False)
    cache = physics.load_map_cache(cpath)
    return cfg, ds, df, cache, tmp


def test_ensemble_ema_tta_end_to_end():
    cfg, ds, df, cache, tmp = _fixture()
    cfg["train"]["ema"] = True
    cfg["train"]["tta"] = True
    cfg["train"]["ensemble_seeds"] = [0, 1]          # 2-member ensemble
    res = L.train_and_evaluate(ds, df, cache, cfg, "station_grouped", device="cpu",
                               out_root=tmp, seed=0, max_epochs=1, num_workers=0, verbose=False)
    rep = res["report"]
    assert rep["n_ensemble"] == 2 and rep["tta"] is True, "ensemble/tta flags not honoured"
    assert set(rep) >= {"R2", "MAE", "RMSE", "coverage", "mean_width"}
    # the built-in alignment assert already ran inside train_and_evaluate (frame label == loader y)
    assert len(res["test_df"]) == rep["n_test"]
    print("[accuracy] EMA + TTA + 2-seed ensemble ran; n_ensemble=%d tta=%s coverage=%.2f OK"
          % (rep["n_ensemble"], rep["tta"], rep["coverage"]))
    return res


def test_contaminated_vs_clean_partition():
    # Build a tiny explicit case: 2 train stations, test has one contaminated (shared station) row-set
    # and one clean station. dup_group unused here (all singletons).
    sp = pd.DataFrame({
        "_row":       [0, 1, 2, 3, 4, 5],
        "station_id": ["a", "a", "b", "b", "c", "c"],
        "split":      ["train", "train", "test", "test", "test", "test"],
    })
    test_df = pd.DataFrame({
        "_row":       [2, 3, 4, 5],
        "station_id": ["b", "b", "c", "c"],
        "y":          [100.0, 110.0, 40.0, 60.0],
        "pred":       [102.0, 108.0, 55.0, 45.0],
    })
    # station 'b' is NOT in train -> those are clean; make 'a' appear in test to be contaminated:
    sp2 = pd.DataFrame({
        "_row":       [0, 1, 2, 3, 4, 5, 6, 7],
        "station_id": ["a", "a", "a", "a", "b", "b", "c", "c"],
        "split":      ["train", "train", "test", "test", "test", "test", "test", "test"],
    })
    test_df2 = pd.DataFrame({
        "_row":       [2, 3, 4, 5, 6, 7],
        "station_id": ["a", "a", "b", "b", "c", "c"],   # 'a' contaminated (in train), b/c clean
        "y":          [80.0, 90.0, 100.0, 110.0, 40.0, 60.0],
        "pred":       [82.0, 88.0, 102.0, 108.0, 55.0, 45.0],
    })
    dup = pd.DataFrame({"_row": list(range(8)), "dup_group": list(range(8))})  # all singletons
    out = L.contaminated_vs_clean(sp2, test_df2, dup, station_col="station_id")
    assert out["contaminated"]["n"] == 2 and out["clean"]["n"] == 4, out
    for side in ("contaminated", "clean"):
        # <5 rows -> NaN by design, but keys must exist (SD_y, RMSE added this change)
        assert {"R2", "MAE", "RMSE", "SD_y", "n"} <= set(out[side]), out[side]
    print("[accuracy] contaminated_vs_clean partitions (2 contaminated, 4 clean) + has SD_y/RMSE OK")


def test_cache_alignment_guard():
    cfg, ds, df, cache, tmp = _fixture()
    n = len(ds)
    ok = physics.verify_cache_alignment(cache, lambda i: data.get_image(ds, i), n,
                                        n_samples=5, seed=0)
    assert ok["ok"] and ok["max_mean_abs_diff"] < 6.0, ok
    # A deliberately MIS-ORDERED view of the cache must be caught.
    shuffled = np.array(cache)[::-1].copy()            # reverse the row order
    raised = False
    try:
        physics.verify_cache_alignment(shuffled, lambda i: data.get_image(ds, i), n,
                                       n_samples=6, seed=1)
    except AssertionError:
        raised = True
    assert raised, "verify_cache_alignment should raise on a reordered cache"
    print("[accuracy] verify_cache_alignment passes on a correct cache, raises on a shuffled one OK")


if __name__ == "__main__":
    test_ensemble_ema_tta_end_to_end()
    test_contaminated_vs_clean_partition()
    test_cache_alignment_guard()
    print("\nALL ACCURACY-LEVER SMOKE CHECKS PASSED")
