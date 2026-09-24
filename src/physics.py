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
from tqdm.auto import tqdm

# EfficientNet is pretrained on ImageNet, so its RGB channels expect this normalisation.
# The two physics channels are kept in [0, 1] (a common choice for auxiliary maps).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Rough typical stats of the two physics maps (transmission, inverted-saturation). We standardise
# them so they reach the ImageNet-pretrained stem at ~unit scale like the RGB channels — otherwise
# their small [0,1] variance leaves them heavily down-weighted (the features that should generalise).
PHYS_MEAN = np.array([0.70, 0.30], dtype=np.float32)
PHYS_STD = np.array([0.20, 0.20], dtype=np.float32)


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


# ---------------------------------------------------------------------------
# 5-channel input assembly (paper Sec 3.5) — RGB + transmission + inverted saturation
# ---------------------------------------------------------------------------
def compute_maps(img, size=224, patch=15, omega=0.95, top_frac=0.001, t_min=0.05):
    """Resize the photo to `size`x`size`, then return (rgb01, transmission, inv_sat).

    We resize FIRST so every image — whatever its original aspect ratio — yields maps at
    a fixed, consistent scale (the audit found widths of 1024 but varying heights).
    All three outputs are float in [0, 1]: rgb01 is (size, size, 3); the two maps are
    (size, size).
    """
    rgb = np.asarray(Image.fromarray(
        (to_float_rgb(img) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32) / 255.0
    t = transmission_map(rgb, patch=patch, omega=omega, top_frac=top_frac, t_min=t_min)
    s = inverted_saturation(rgb)
    return rgb, t, s


def assemble_five_channel(rgb01, t01, s01, imagenet_norm=True) -> np.ndarray:
    """Stack into a (5, H, W) float32 tensor: normalised RGB + standardised transmission + inv-sat."""
    phys = np.stack([t01, s01], axis=2)                       # (H, W, 2), values in [0, 1]
    if imagenet_norm:
        rgb = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
        phys = (phys - PHYS_MEAN) / PHYS_STD                  # bring physics maps to ~unit scale
    else:
        rgb = rgb01
    hwc = np.concatenate([rgb, phys], axis=2)
    return np.transpose(hwc, (2, 0, 1)).astype(np.float32)


def five_channel(img, size=224, patch=15, omega=0.95, top_frac=0.001, t_min=0.05,
                 imagenet_norm=True) -> np.ndarray:
    """One-shot: photo -> (5, size, size) model input. Used by inference/demo (no cache)."""
    rgb, t, s = compute_maps(img, size, patch, omega, top_frac, t_min)
    return assemble_five_channel(rgb, t, s, imagenet_norm)


# ---------------------------------------------------------------------------
# disk cache for the two physics maps (paper Sec 3.5: "computed once and cached")
# ---------------------------------------------------------------------------
def build_map_cache(n, get_image, out_path, size=224, patch=15, omega=0.95,
                    top_frac=0.001, t_min=0.05, progress=True) -> str:
    """Precompute transmission + inverted-saturation for `n` images and memmap to disk.

    Stored as uint8 (maps are in [0,1] -> x255) in a single array of shape
    (n, 2, size, size), indexed by dataset position. Recomputing the Dark Channel Prior
    every training epoch would be far too slow, so we do it once. `get_image(i)` must
    return the PIL image for dataset position i. ~1.1 GB for the full dataset.
    """
    cache = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(n, 2, size, size))
    it = tqdm(range(n), desc="caching physics maps") if progress else range(n)
    for i in it:
        _, t, s = compute_maps(get_image(i), size, patch, omega, top_frac, t_min)
        cache[i, 0] = np.clip(t * 255, 0, 255).astype(np.uint8)
        cache[i, 1] = np.clip(s * 255, 0, 255).astype(np.uint8)
    cache.flush()
    return out_path


def load_map_cache(path):
    """Open a cached map array read-only (memory-mapped, so it doesn't load into RAM)."""
    return np.load(path, mmap_mode="r")


def five_channel_cached(img, cache, row, size=224, imagenet_norm=True) -> np.ndarray:
    """Assemble the 5-channel input using cached maps (fast path for training).

    `img` is the PIL photo (RGB is cheap to decode+resize); the two physics maps are read
    from `cache[row]` instead of recomputed.
    """
    rgb = np.asarray(Image.fromarray(
        (to_float_rgb(img) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32) / 255.0
    t = cache[row, 0].astype(np.float32) / 255.0
    s = cache[row, 1].astype(np.float32) / 255.0
    return assemble_five_channel(rgb, t, s, imagenet_norm)
