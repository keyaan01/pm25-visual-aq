"""Physics-guided image features: Dark Channel Prior + inverted saturation.

WHAT / WHY (paper Sec 3.5)
--------------------------
Instead of asking the network to discover the physics of haze from raw pixels, we
hand it two pre-computed maps derived from atmospheric optics, which behave the same
way in every city:

1. **Transmission map (Dark Channel Prior, He et al. 2011).**
   In a clear outdoor photo, almost every small patch has at least one very dark
   pixel in at least one colour channel (a shadow, a dark window, tree bark). Haze
   adds a pale grey veil that lifts those dark pixels. So how dark the darkest pixel
   *still* is tells you how much haze sits between the camera and the scene. The
   result is `transmission` in [0, 1]: LOW where haze is dense, HIGH in clear air.

2. **Inverted-saturation map (Fang et al. 2024).**
   The dark-channel trick fails on sky, because sky has nothing dark in it. Haze
   also washes colour out of a scene (mixes in white), lowering saturation. Sky has
   colour to lose, so an *inverted* saturation map (high where saturation is low)
   stays informative exactly where the dark channel breaks down. The two maps cover
   each other's blind spots.

The final network input (built in Phase 3 / `dataset.py`) is a 5-channel image:
R, G, B, transmission, inverted-saturation.

All functions take an RGB image as a float array in [0, 1] with shape (H, W, 3),
unless noted, and return float maps in [0, 1] with shape (H, W).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def to_float_rgb(img) -> np.ndarray:
    """Accept a PIL image or an array and return float32 RGB in [0, 1], shape (H,W,3)."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:  # grayscale -> 3 channels
            arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


# ---------------------------------------------------------------------------
# Dark Channel Prior -> transmission
# ---------------------------------------------------------------------------
def dark_channel(rgb: np.ndarray, patch: int = 15) -> np.ndarray:
    """Dark channel: per-pixel min over colour channels, then a local min-filter.

    The local min-filter is an erosion with a `patch` x `patch` window.
    """
    min_over_channels = rgb.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_over_channels, kernel)


def atmospheric_light(rgb: np.ndarray, dark: np.ndarray, top_frac: float = 0.001) -> np.ndarray:
    """Estimate the haze/ambient colour A (a 3-vector) -- the "brightest haze".

    Take the brightest `top_frac` of pixels in the dark channel (the haziest
    regions), then among those pick the one that is brightest in the original
    image. This follows He et al. 2011.
    """
    h, w = dark.shape
    n = max(1, int(h * w * top_frac))
    flat_idx = np.argpartition(dark.ravel(), -n)[-n:]  # indices of the n haziest pixels
    candidates = rgb.reshape(-1, 3)[flat_idx]
    brightest = candidates.sum(axis=1).argmax()
    A = candidates[brightest]
    return np.clip(A, 1e-3, 1.0)  # avoid divide-by-zero downstream


def transmission_map(
    img,
    patch: int = 15,
    omega: float = 0.95,
    top_frac: float = 0.001,
    t_min: float = 0.05,
) -> np.ndarray:
    """Estimate the transmission map t(x) in [t_min, 1].

    t = 1 - omega * darkchannel(image / A).  omega < 1 keeps a little haze so
    distant scenes still look natural (He et al.). LOW t = dense haze.
    """
    rgb = to_float_rgb(img)
    dark = dark_channel(rgb, patch)
    A = atmospheric_light(rgb, dark, top_frac)
    normalized = rgb / A[None, None, :]
    t = 1.0 - omega * dark_channel(normalized, patch)
    return np.clip(t, t_min, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# inverted saturation
# ---------------------------------------------------------------------------
def inverted_saturation(img) -> np.ndarray:
    """Inverted HSV saturation in [0, 1]: HIGH where colour is washed out (hazy/sky)."""
    rgb = to_float_rgb(img)
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    return (1.0 - sat).astype(np.float32)


# ---------------------------------------------------------------------------
# scalar summary used by the Phase-1 falsification test (paper Sec 3.3)
# ---------------------------------------------------------------------------
def mean_transmission(img, patch: int = 15, omega: float = 0.95) -> float:
    """A single "how clear is this photo" number = average transmission.

    Physics predicts hazier photos (higher pollution / AQI) have LOWER transmission,
    so across the dataset this scalar should be NEGATIVELY correlated with the AQI
    label. If it is not, images and labels were mis-paired -- a problem we must catch
    before modelling.
    """
    return float(transmission_map(img, patch=patch, omega=omega).mean())
