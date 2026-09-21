"""Phase-1 smoke test: runs every Phase-1 module on the tiny local fixture.

This does NOT need a GPU or the full 1 GB dataset. Build the fixture first:

    python tests/_build_fixture.py
    python tests/smoke_phase1.py

It checks the *logic* (shapes, ranges, cleaning) so we can trust the code before it
ever runs on the full dataset in Colab.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
from src import aqi, audit, data, physics  # noqa: E402

FIX = str(Path(__file__).resolve().parent / "fixture_ds")


def main():
    # --- data.py: load + clean ---
    ds, df = data.load_clean(FIX, from_disk=True, seed=42)
    assert "camera_angle" not in df.columns and "image" not in df.columns
    assert df["image_id"].is_unique, "de-duplication failed"
    print(f"[data]    pooled={len(ds)} clean={len(df)} cols={list(df.columns)}")

    # de-dup path: inject a duplicate and confirm it is removed
    raw = data.metadata_frame(ds)
    cl = data.clean_metadata(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
    assert cl.attrs["n_duplicates_removed"] == 1 and cl["image_id"].is_unique
    print(f"[data]    dedup removed={cl.attrs['n_duplicates_removed']} (expected 1)")

    img = data.get_image(ds, int(df["_row"].iloc[0]))
    print(f"[data]    sample image {img.size} {img.mode}")

    # --- physics.py ---
    t = physics.transmission_map(img)
    s = physics.inverted_saturation(img)
    assert t.shape == (img.size[1], img.size[0]) and 0 <= t.min() and t.max() <= 1.0001
    assert 0 <= s.min() and s.max() <= 1.0001
    print(f"[physics] transmission {t.shape} [{t.min():.3f},{t.max():.3f}] "
          f"inv-sat [{s.min():.3f},{s.max():.3f}]")

    # --- aqi.py ---
    assert abs(aqi.pm25_to_aqi(35.4) - 100) < 1e-6
    assert abs(aqi.pm25_to_aqi(aqi.aqi_to_pm25(150)) - 150) < 1e-6
    assert aqi.aqi_category(39) == "Good" and aqi.aqi_category(160) == "Unhealthy"
    print("[aqi]     conversions + categories OK")

    # --- audit.py ---
    rep = audit.redundancy_report(ds, df, progress=False)
    assert 0 < rep["distinct_ratio"] <= 1.0
    sps = audit.images_per_station(df)
    cor = audit.transmission_label_correlation(ds, df, sample=None, progress=False)
    print(f"[audit]   distinct_ratio={rep['distinct_ratio']:.3f} "
          f"stations={sps['n_stations']} falsification_r={cor['pearson_r']:.3f}")

    print("\nALL PHASE-1 SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
