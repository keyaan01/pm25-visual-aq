"""Leakage-safe train / calibration / test splits (paper Sec 3.4).

WHY THIS MATTERS
----------------
To know whether a model *learned haze* (good) or merely *memorised places* (leakage),
we must control how photos are divided into train / calibration / test. We build FOUR
strategies and compare them:

* **random** — every photo assigned independently. This is the *leaky control*: because
  near-duplicate photos from the same place can land on both sides, scores look better
  than they should. (The dataset's shipped split is already station-disjoint, so we
  construct this leaky control ourselves to measure the inflation.)
* **station_grouped** — every photo from one station goes entirely to one split. No
  location straddles the boundary, so the model is tested on genuinely unseen places.
  **This is our primary protocol.**
* **geographic** — whole geographic regions (coarse lat/lon cells) are held out, a
  stronger test of "did it learn haze or local scenery?".
* **temporal** — train on earlier years, test on later ones (future generalisation).

The **calibration** split is kept strictly separate from train and test because the
conformal guarantee in Phase 6 depends on it never being seen during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "cal", "test")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_split(n: int, fractions, seed: int) -> np.ndarray:
    """Assign n rows to splits by shuffling and slicing (exact proportions)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out = np.empty(n, dtype=object)
    n_train = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    out[order[:n_train]] = "train"
    out[order[n_train:n_train + n_cal]] = "cal"
    out[order[n_train + n_cal:]] = "test"
    return out


def _group_split(groups: pd.Series, fractions, seed: int) -> np.ndarray:
    """Assign whole groups (e.g. stations, regions) to splits, balancing row counts.

    Deficit-greedy: shuffle the groups, then send each group to whichever split is
    currently furthest *below* its target number of rows. Keeps proportions close while
    guaranteeing a group never spans two splits.
    """
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    total = int(sizes.sum())
    targets = {s: f * total for s, f in zip(SPLIT_NAMES, fractions)}
    current = {s: 0 for s in SPLIT_NAMES}
    assign = {}
    for g in order:
        s = max(SPLIT_NAMES, key=lambda k: targets[k] - current[k])
        assign[g] = s
        current[s] += int(sizes[g])
    return groups.map(assign).to_numpy()


def _geo_cell(df, lon_col, lat_col, cell_deg: float) -> pd.Series:
    """Coarse geographic region id = (rounded lat cell, rounded lon cell)."""
    lat_cell = (np.floor(df[lat_col] / cell_deg) * cell_deg).astype(int)
    lon_cell = (np.floor(df[lon_col] / cell_deg) * cell_deg).astype(int)
    return lat_cell.astype(str) + "_" + lon_cell.astype(str)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    strategy: str = "station_grouped",
    fractions=(0.65, 0.15, 0.20),
    seed: int = 42,
    station_col: str = "station_id",
    time_col: str = "captured_at",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    geo_cell_deg: float = 10.0,
) -> pd.DataFrame:
    """Return a copy of `df` with a new `split` column in {train, cal, test}."""
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    out = df.copy()

    if strategy == "random":
        out["split"] = _row_split(len(out), fractions, seed)
    elif strategy == "station_grouped":
        out["split"] = _group_split(out[station_col], fractions, seed)
    elif strategy == "geographic":
        cells = _geo_cell(out, lon_col, lat_col, geo_cell_deg)
        out["split"] = _group_split(cells, fractions, seed)
    elif strategy == "shipped":
        # Reproduce the PM25Vision paper's 80/20 split exactly: test = their shipped test rows.
        # We still need a calibration set, so we carve it (station-grouped) out of their TRAIN rows.
        if "orig_split" not in out.columns:
            raise ValueError("shipped split needs 'orig_split' (load via load_pooled on a DatasetDict)")
        lab = np.where(out["orig_split"].to_numpy() == "test", "test", "train").astype(object)
        train_idx = np.where(lab == "train")[0]
        cal_frac = fractions[1] / (fractions[0] + fractions[1])  # cal share OF the train portion
        sub = out.iloc[train_idx]
        grp = _group_split(sub[station_col], (1 - cal_frac, cal_frac, 0.0), seed)
        for j, g in zip(train_idx, grp):
            lab[j] = "cal" if g == "cal" else "train"
        out["split"] = lab
    elif strategy == "temporal":
        order = out[time_col].astype("datetime64[ns]").argsort(kind="stable").to_numpy()
        n = len(out)
        n_train = int(round(fractions[0] * n))
        n_cal = int(round(fractions[1] * n))
        lab = np.empty(n, dtype=object)
        lab[order[:n_train]] = "train"
        lab[order[n_train:n_train + n_cal]] = "cal"
        lab[order[n_train + n_cal:]] = "test"
        out["split"] = lab
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    out.attrs["split_strategy"] = strategy
    return out


# ---------------------------------------------------------------------------
# reporting / sanity checks
# ---------------------------------------------------------------------------
def split_report(df_split: pd.DataFrame, station_col: str = "station_id") -> dict:
    """Row counts, fractions, and leakage checks for a split assignment."""
    counts = df_split["split"].value_counts().reindex(SPLIT_NAMES).fillna(0).astype(int)
    fracs = (counts / counts.sum()).round(3)
    # how many stations appear in more than one split (0 = leakage-safe by station)
    per_station_splits = df_split.groupby(station_col)["split"].nunique()
    straddling = int((per_station_splits > 1).sum())
    return {
        "strategy": df_split.attrs.get("split_strategy"),
        "counts": counts.to_dict(),
        "fractions": fracs.to_dict(),
        "n_stations": {s: int(df_split.loc[df_split.split == s, station_col].nunique())
                       for s in SPLIT_NAMES},
        "stations_straddling_splits": straddling,
    }


def split_indices(df_split: pd.DataFrame):
    """Convenience: return (train_df, cal_df, test_df) as separate frames."""
    return tuple(df_split[df_split.split == s].copy() for s in SPLIT_NAMES)
