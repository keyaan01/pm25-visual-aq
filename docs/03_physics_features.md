# Phase 3 — Physics features (the 5-channel input)

Companion to `notebooks/03_physics_features.ipynb`. Maps to **paper §3.5**.

## The idea

A neural network *could* learn the optics of haze from raw pixels, but that wastes data
and invites shortcuts (memorising what a city's buildings look like). Instead we hand the
model two maps that come straight from atmospheric physics and behave the same in every
city. The network input becomes **5 channels**: R, G, B, transmission, inverted-saturation.

### 1. Transmission map (Dark Channel Prior, He et al. 2011)
"Transmission" is the fraction of the scene's light that survives the trip to the camera —
**low where haze is dense**. The trick to estimate it: in a clear outdoor photo, almost
every small patch contains at least one very dark pixel in at least one colour channel (a
shadow, a dark window, tree bark). Haze adds a pale grey veil that lifts those dark pixels.
So *how dark the darkest pixel still is* tells you how much haze sits in front of the scene.

`transmission_map()` computes this: take the per-pixel minimum over R/G/B, erode it with a
15×15 window (the "dark channel"), estimate the haze colour from the haziest pixels, and
turn it into `t = 1 − 0.95 · darkchannel(image / hazecolour)`.

### 2. Inverted-saturation map (Fang et al. 2024)
The Dark Channel Prior fails on **sky**, because sky has nothing dark in it. But haze also
washes colour out of a scene (mixes in white), lowering *saturation* — and sky has colour to
lose. So an **inverted** saturation map (high where saturation is low) stays informative
exactly where the dark channel breaks down. The two maps cover each other's blind spots.

## What the code does

- `compute_maps(img, size)` — resize the photo to 224×224 **first** (so every image, whatever
  its original aspect ratio, gives maps at a consistent scale), then compute both maps. Returns
  `rgb01`, `transmission`, `inverted_saturation`, all in [0, 1].
- `five_channel(img)` — one-shot photo → `(5, 224, 224)` tensor (used by the demo/inference).
- `assemble_five_channel(...)` — stacks normalised RGB + the two maps. RGB is ImageNet-normalised
  (EfficientNet is pretrained on ImageNet); the two physics channels stay in [0, 1].

## Why (and how) we cache

Computing the Dark Channel Prior for every image on *every* training epoch would dominate
training time. So `build_map_cache()` computes both maps **once** for all images and stores
them to Drive as a single memory-mapped `uint8` array of shape `(N, 2, 224, 224)` (~1.1 GB
for the full dataset), indexed by dataset position. During training (Phase 5), the RGB is
decoded fresh (cheap) and the two maps are read straight from the cache
(`five_channel_cached`). Storing as `uint8` costs only rounding error (~0.004), verified
against the on-the-fly computation.

## What to look for in the visualisations

- In the **transmission** map, hazier (higher-AQI) photos look *brighter/less dark* overall —
  less of the scene's light survived, so transmission is uniformly lower... shown as higher
  values where clear, lower where hazy. Compare a clear vs a hazy example side by side.
- In the **inverted-saturation** map, the **sky region lights up**, which is exactly the
  region the Dark Channel Prior can't read.

## What's next

Phase 4 (`04_model`) builds the EfficientNet-B0 backbone, widened from 3 to 5 input
channels, with three monotone quantile heads.
