"""Audit the data BEFORE modelling (paper Sec 3.3).

The audit answers three questions that decide whether any later result is meaningful:

1. **How much redundancy is there?** Consecutive street photos look almost identical.
   If near-duplicates land on both sides of a train/test split, the model can score
   well by *recognising a scene it already saw* instead of reading haze (this is
   "leakage"). We fingerprint every image with a perceptual hash and group
   near-duplicates, then report the ratio of distinct groups to total images.

2. **How are images distributed across stations?** This tells us whether a
   station-grouped split (our leakage-safe protocol) is feasible.

3. **Falsification test:** physics says hazier photos have LOWER transmission and
   HIGHER AQI, so average transmission should be NEGATIVELY correlated with the
   label. If that correlation is absent, images and labels were mis-paired -- a
   fatal problem we want to catch now, not after training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

import imagehash

from . import physics
from .data import get_image


# ---------------------------------------------------------------------------
# 1. near-duplicate detection via perceptual hashing
# ---------------------------------------------------------------------------
def _phash_uint64(pil_img, hash_size: int = 8) -> np.uint64:
    """Compute a 64-bit perceptual hash and pack it into a single uint64.

    Unlike a normal (cryptographic) hash, a perceptual hash gives *similar* images
    *similar* fingerprints, so near-duplicates have a small bit-difference.
    """
    bits = imagehash.phash(pil_img, hash_size=hash_size).hash.flatten()
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def compute_phashes(ds, df, hash_size: int = 8, progress: bool = True) -> np.ndarray:
    """Return a uint64 perceptual hash for every row in `df` (order preserved)."""
    rows = df["_row"].to_numpy()
    it = tqdm(rows, desc="perceptual hashes") if progress else rows
    return np.array([_phash_uint64(get_image(ds, r), hash_size) for r in it], dtype=np.uint64)


def _hamming_to_all(h: np.uint64, arr: np.ndarray) -> np.ndarray:
    """Hamming distance (number of differing bits) between `h` and every hash in `arr`."""
    x = np.bitwise_xor(arr, h).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(x, axis=1).sum(axis=1)


def near_duplicate_groups(hashes: np.ndarray, max_distance: int = 5) -> np.ndarray:
    """Group near-duplicate images with a union-find over Hamming distance.

    Two images are linked if their hashes differ by <= `max_distance` bits.
    Returns an array of group ids (same length as `hashes`); images sharing a group
    id are near-duplicates of each other.
    """
    n = len(hashes)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        dist = _hamming_to_all(hashes[i], hashes[i + 1 :])
        for j in np.nonzero(dist <= max_distance)[0]:
            union(i, i + 1 + int(j))
    return np.array([find(i) for i in range(n)])


def redundancy_report(ds, df, hash_size: int = 8, max_distance: int = 5, progress: bool = True):
    """Full near-duplicate summary. Returns a dict; also attaches group ids to a copy of df."""
    hashes = compute_phashes(ds, df, hash_size=hash_size, progress=progress)
    groups = near_duplicate_groups(hashes, max_distance=max_distance)
    n_total = len(df)
    n_groups = len(np.unique(groups))
    out = df.copy()
    out["dup_group"] = groups
    group_sizes = pd.Series(groups).value_counts()
    return {
        "n_images": n_total,
        "n_distinct_groups": int(n_groups),
        "distinct_ratio": n_groups / n_total,
        "largest_group": int(group_sizes.max()),
        "n_groups_with_dupes": int((group_sizes > 1).sum()),
        "df_with_groups": out,
    }


# ---------------------------------------------------------------------------
# 2. images per station
# ---------------------------------------------------------------------------
def images_per_station(df, station_col: str = "station_id") -> dict:
    """Summary statistics of how many images each monitoring station contributes."""
    counts = df[station_col].value_counts()
    return {
        "n_stations": int(counts.size),
        "min": int(counts.min()),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "max": int(counts.max()),
        "q90": int(counts.quantile(0.90)),
        "q99": int(counts.quantile(0.99)),
        "pct_single_image": float((counts == 1).mean() * 100),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# 3. falsification test: transmission vs label
# ---------------------------------------------------------------------------
def transmission_label_correlation(
    ds, df, target_col: str = "pm25", sample: int | None = 500,
    patch: int = 15, omega: float = 0.95, seed: int = 42, progress: bool = True,
) -> dict:
    """Correlate average transmission with the AQI label.

    Expectation (physics): NEGATIVE correlation -- hazier (higher-AQI) photos have
    lower transmission. `sample` limits how many images we score (transmission is a
    little slow); None uses all rows.
    """
    work = df if sample is None or sample >= len(df) else df.sample(sample, random_state=seed)
    rows = work["_row"].to_numpy()
    it = tqdm(rows, desc="transmission") if progress else rows
    trans = np.array([physics.mean_transmission(get_image(ds, r), patch, omega) for r in it])
    labels = work[target_col].to_numpy(dtype=float)
    pear_r, pear_p = pearsonr(trans, labels)
    spear_r, spear_p = spearmanr(trans, labels)
    return {
        "n": len(rows),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "passes": bool(pear_r < 0),  # physics says it should be negative
        "mean_transmission": trans,
        "labels": labels,
    }
