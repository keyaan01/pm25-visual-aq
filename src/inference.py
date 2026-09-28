"""One-photo inference for the demo (paper Phase 10).

`predict(image, bundle)` runs the whole trained pipeline on a single street photo:
    5-channel physics input -> EfficientNet-B0 -> invert log quantiles -> conformal widening (Q)
    -> isotonic-recalibrated point estimate -> abstention gate (OOD first, then uncertainty)
and returns everything the UI shows: the AQI estimate, the low-high interval, the EPA category + colour,
the answer/abstain decision (with a reason), and the two physics maps.

A **bundle** is a self-contained folder produced by `notebooks/08_abstention.ipynb`:
    best_model.pth            the trained checkpoint (carries cfg + y_mean/y_std/quantiles)
    inference_bundle.json     Q, y_mean, y_std, isotonic knots, tau_ood, tau_width, thresholds
    maha_mean.npy / maha_precision.npy   the OOD detector (optional; handcrafted fallback if absent)
so the demo needs no dataset or calibration set at serve time.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import abstain as A
from . import model as M
from . import physics
from . import train as T
from .aqi import AQI_CATEGORIES, aqi_category

# Standard US-EPA AQI category colours (hex), aligned to AQI_CATEGORIES order.
EPA_COLOURS = {
    "Good": "#00e400", "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00", "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023",
}


def apply_isotonic_knots(point, xk, yk):
    """Apply a saved isotonic map (its x/y thresholds) via clipped linear interpolation (>=0)."""
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)
    p = np.clip(np.asarray(point, dtype=float), xk[0], xk[-1])
    return np.maximum(np.interp(p, xk, yk), 0.0)


# ---------------------------------------------------------------------------
# bundle I/O
# ---------------------------------------------------------------------------
def save_bundle(out_dir, net, Q, y_mean, y_std, iso, tau_ood, tau_width, maha=None,
                handcrafted_tau=0.5):
    """Write a self-contained inference bundle. `iso` is a fitted sklearn IsotonicRegression (or None).

    The checkpoint (best_model.pth) must already sit in `out_dir` (training saves it there); here we add
    the scalars + isotonic knots (JSON) and the Mahalanobis arrays (.npy)."""
    os.makedirs(out_dir, exist_ok=True)
    bundle = {"Q": float(Q), "y_mean": float(y_mean), "y_std": float(y_std),
              "tau_ood": float(tau_ood), "tau_width": float(tau_width),
              "handcrafted_tau": float(handcrafted_tau)}
    if iso is not None:
        bundle["iso_x"] = np.asarray(iso.X_thresholds_, dtype=float).tolist()
        bundle["iso_y"] = np.asarray(iso.y_thresholds_, dtype=float).tolist()
    json.dump(bundle, open(os.path.join(out_dir, "inference_bundle.json"), "w"), indent=2)
    if maha is not None:
        A.save_mahalanobis(out_dir, maha)
    return out_dir


def load_bundle(out_dir, device="cpu"):
    """Load a bundle folder into a ready-to-serve dict (builds + loads the model)."""
    b = json.load(open(os.path.join(out_dir, "inference_bundle.json")))
    ckpt = torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device, weights_only=False)
    net = M.build_model(ckpt["cfg"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    b["net"] = net
    b["device"] = device
    b["maha"] = A.load_mahalanobis(out_dir)
    return b


# ---------------------------------------------------------------------------
# the one-photo prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(image, bundle) -> dict:
    """Run the full pipeline on one PIL/array image. Returns a UI-ready dict."""
    net, device = bundle["net"], bundle.get("device", "cpu")
    # 5-channel physics input (+ keep the maps for display)
    rgb01, t01, s01 = physics.compute_maps(image)
    x = physics.assemble_five_channel(rgb01, t01, s01, imagenet_norm=True)
    xt = torch.from_numpy(x)[None].to(device)

    out = net(xt)
    q_log = out["quantiles"].cpu().numpy()                     # (1, 3) log space
    q_aqi = np.exp(q_log)                                       # (1, 3) AQI
    Q = bundle["Q"]
    low = float(max(q_aqi[0, 0] - Q, 0.0))
    high = float(q_aqi[0, 2] + Q)

    # accuracy point: de-standardise the point head, then isotonic-recalibrate
    point = T.point_to_aqi(out["point"].cpu().numpy(), bundle["y_mean"], bundle["y_std"])[0]
    if "iso_x" in bundle:
        point = float(apply_isotonic_knots([point], bundle["iso_x"], bundle["iso_y"])[0])
    aqi = float(max(point, 0.0))

    # abstention: OOD (Mahalanobis if available, else handcrafted) then log-width uncertainty
    log_width = float(q_log[0, 2] - q_log[0, 0])
    hand = A.handcrafted_ood_score(rgb01)
    if bundle.get("maha") is not None:
        feats = net.backbone(xt).cpu().numpy()
        ood_score = float(A.mahalanobis_score(bundle["maha"], feats)[0])
        tau_ood = bundle["tau_ood"]
    else:
        ood_score, tau_ood = hand, bundle.get("handcrafted_tau", 0.5)
    gate = A.decide(ood_score, log_width, tau_ood, bundle["tau_width"])

    cat = aqi_category(aqi)
    return {
        "aqi": round(aqi, 1),
        "low": round(low, 1),
        "high": round(high, 1),
        "epa_category": cat,
        "epa_colour": EPA_COLOURS.get(cat, "#888888"),
        "answer": gate["answer"],
        "reason": gate["reason"],
        "ood_score": round(ood_score, 3),
        "handcrafted_ood": round(hand, 3),
        "log_width": round(log_width, 3),
        "transmission_map": t01,
        "inv_sat_map": s01,
    }
