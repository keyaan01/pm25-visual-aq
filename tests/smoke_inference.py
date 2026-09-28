"""Smoke test for the demo pipeline (src/inference.py) + collect_features, CPU only.

  python tests/_build_fixture.py     # once
  python tests/smoke_inference.py

Builds a tiny fixture model, saves a real inference bundle (checkpoint + Q + stats + isotonic knots +
Mahalanobis OOD), then checks predict() returns a full UI dict on a real photo, that the abstention
gate is wired (forcing tau_ood low -> abstain, high -> answer), and that a black image scores as OOD.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (abstain as A, calibrate as C, data, dataset as D, inference as I,  # noqa: E402
                 model as M, physics, recalibrate as R, splits, train as T)
from src.config import load_config  # noqa: E402

FIX = str(Path(__file__).resolve().parent / "fixture_ds")


def _build_bundle(tmp, with_maha=True):
    cfg = load_config()
    cfg["train"]["batch_size"] = 4
    cfg["model"]["pretrained"] = False
    T.set_seed(0)
    ds, df = data.load_clean(FIX, from_disk=True, seed=42)
    cpath = os.path.join(tmp, "maps.npy")
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cpath, size=224, progress=False)
    cache = physics.load_map_cache(cpath)
    sp = splits.make_splits(df, strategy="station_grouped", seed=42)
    loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=0)
    out_dir = os.path.join(tmp, "bundle")
    net = M.build_model(cfg)
    T.train_model(net, loaders, cfg, device="cpu", out_dir=out_dir, max_epochs=1,
                  max_steps_per_epoch=3, verbose=False)
    T.load_checkpoint(os.path.join(out_dir, "best_model.pth"), net, map_location="cpu")

    cal = T.collect_outputs(net, loaders["cal"], "cpu")
    cal_feats = T.collect_features(net, loaders["cal"], "cpu")["feats"]
    ymean, ystd = float(net.y_mean), float(net.y_std)
    cal_point = T.point_to_aqi(cal["point_out"], ymean, ystd)
    iso = R.fit_isotonic(cal_point, cal["y_raw"])
    Q = C.conformal_Q(np.exp(cal["q_log"]), cal["y_raw"], cfg["calibration"]["coverage"])
    maha = A.fit_mahalanobis(cal_feats)
    tau_ood = A.ood_threshold(A.mahalanobis_score(maha, cal_feats))
    tau_w = A.width_threshold(A.log_interval_width(cal["q_log"]), 0.9)
    # with_maha=False mirrors the DEPLOYED demo bundle (handcrafted-only OOD gate, no Mahalanobis arrays)
    I.save_bundle(out_dir, net, Q, ymean, ystd, iso, tau_ood, tau_w,
                  maha=(maha if with_maha else None), handcrafted_tau=0.5)
    return I.load_bundle(out_dir, device="cpu"), ds, cal_feats, net


def test_predict_and_gate():
    tmp = tempfile.mkdtemp()
    bundle, ds, cal_feats, net = _build_bundle(tmp)
    img = data.get_image(ds, 0)                       # a real fixture street photo
    r = I.predict(img, bundle)
    need = {"aqi", "low", "high", "epa_category", "epa_colour", "answer", "reason",
            "transmission_map", "inv_sat_map", "ood_score", "log_width"}
    assert need <= set(r), f"missing keys: {need - set(r)}"
    assert r["low"] <= r["high"] and r["aqi"] >= 0
    assert r["transmission_map"].shape == (224, 224) and r["inv_sat_map"].shape == (224, 224)
    assert isinstance(r["answer"], bool)
    print("[inference] predict() -> AQI %.0f [%.0f, %.0f] cat=%s answer=%s OK"
          % (r["aqi"], r["low"], r["high"], r["epa_category"], r["answer"]))

    # gate is wired: force OOD threshold low -> abstain; high -> answer
    bundle["tau_ood"], bundle["tau_width"] = -1e9, 1e9
    assert I.predict(img, bundle)["answer"] is False
    assert "out-of-distribution" in I.predict(img, bundle)["reason"]
    bundle["tau_ood"], bundle["tau_width"] = 1e9, 1e9
    assert I.predict(img, bundle)["answer"] is True
    print("[inference] abstention gate wired (force-abstain + force-answer) OK")


def test_black_image_is_ood():
    tmp = tempfile.mkdtemp()
    bundle, ds, cal_feats, net = _build_bundle(tmp)
    black = Image.new("RGB", (224, 224), (0, 0, 0))
    r = I.predict(black, bundle)
    # a black frame is unusable: either the Mahalanobis score exceeds the calibration mean, or the
    # interpretable handcrafted score flags it outright.
    cal_mean = float(A.mahalanobis_score(bundle["maha"], cal_feats).mean())
    assert r["ood_score"] > cal_mean or r["handcrafted_ood"] > 0.8, (r["ood_score"], cal_mean, r["handcrafted_ood"])
    print("[inference] black image flagged OOD (score %.2f vs cal mean %.2f; handcrafted %.2f) OK"
          % (r["ood_score"], cal_mean, r["handcrafted_ood"]))


def test_deployed_handcrafted_gate():
    """The DEPLOYED config: bundle with maha=None -> predict() runs the handcrafted-only OOD path."""
    tmp = tempfile.mkdtemp()
    bundle, ds, cal_feats, net = _build_bundle(tmp, with_maha=False)
    assert bundle.get("maha") is None, "deployed demo bundle must ship NO Mahalanobis arrays"
    black = Image.new("RGB", (224, 224), (0, 0, 0))
    r = I.predict(black, bundle)
    assert r["answer"] is False and ("dark" in r["reason"] or "flat" in r["reason"]), r["reason"]
    # a bright, detailed fixture photo passes the handcrafted OOD gate (score below the handcrafted tau)
    img = data.get_image(ds, 0)
    r2 = I.predict(img, bundle)
    assert r2["handcrafted_ood"] < bundle["handcrafted_tau"], (r2["handcrafted_ood"], bundle["handcrafted_tau"])
    print("[inference] DEPLOYED handcrafted-only gate: abstains on black (%s), passes good photo OK" % r["reason"])


if __name__ == "__main__":
    test_predict_and_gate()
    test_black_image_is_ood()
    test_deployed_handcrafted_gate()
    print("\nALL INFERENCE/DEMO SMOKE CHECKS PASSED")
