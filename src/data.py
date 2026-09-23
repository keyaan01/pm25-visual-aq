"""Load and clean the PM25Vision dataset (paper Sec 3.2-3.3).

WHAT / WHY
----------
PM25Vision ships as a Hugging Face dataset with the JPEG bytes embedded in an
`image` column and one row per photo. Two things matter before we touch a model:

* The dataset is delivered as a train/test split, but the paper re-splits the data
  itself (by station, region, time). So we **pool both splits into one pool** and do
  our own splitting later (Phase 2).
* Our audit found three issues this module fixes up front:
    - dead columns: `camera_angle` and `quality_score` are null for every row;
      `downloaded_at` is just the scrape date; `filename` duplicates `image_id`.
    - 123 exact-duplicate `image_id` rows -> we drop duplicates.
    - the rows are **ordered** (a contiguous block is all one AQI band), so a naive
      head/tail split would be biased -> we shuffle with a fixed seed.

DESIGN
------
We keep the heavy image bytes inside the Hugging Face `Dataset` object and expose a
lightweight pandas `DataFrame` of *metadata only* (no image bytes) for auditing and
splitting. Every metadata row carries a `_row` index pointing back into the dataset,
so `get_image(ds, row)` fetches the picture on demand. This keeps memory low.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from PIL import Image

# Columns we drop because they carry no usable signal (see audit).
DEAD_COLUMNS = ["camera_angle", "quality_score", "downloaded_at", "filename", "quality"]


# ---------------------------------------------------------------------------
# loading the raw dataset
# ---------------------------------------------------------------------------
def load_pooled(source: str, from_disk: bool = True) -> Dataset:
    """Return a single pooled `Dataset` (all splits concatenated) with a `_row` index.

    Parameters
    ----------
    source     : a `save_to_disk` folder (Colab/Drive) if `from_disk`, else a HF repo id.
    from_disk  : True  -> `load_from_disk(source)` (what Colab uses after 00_setup),
                 False -> `load_dataset(source)` (downloads from the Hub).
    """
    dd = load_from_disk(source) if from_disk else load_dataset(source)
    # `dd` may be a DatasetDict (has splits) or already a single Dataset.
    if hasattr(dd, "keys"):
        parts, orig = [], []
        for k in dd.keys():
            parts.append(dd[k])
            orig += [k] * len(dd[k])          # remember each row's SHIPPED split (train/test)
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    else:
        ds = dd
        orig = ["train"] * len(ds)
    ds = ds.add_column("_row", list(range(len(ds))))
    ds = ds.add_column("orig_split", orig)     # used by the "shipped" split strategy
    return ds


# ---------------------------------------------------------------------------
# metadata frame + cleaning
# ---------------------------------------------------------------------------
def metadata_frame(ds: Dataset) -> pd.DataFrame:
    """Return a metadata DataFrame (everything except the image bytes)."""
    keep = [c for c in ds.column_names if c != "image"]
    return ds.remove_columns(["image"]).select_columns(keep).to_pandas()


def clean_metadata(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Drop dead columns, remove duplicate image_ids, and shuffle (seeded).

    Returns a fresh, re-indexed DataFrame. `_row` is preserved so we can still fetch
    the matching image from the dataset.
    """
    out = df.copy()
    out = out.drop(columns=[c for c in DEAD_COLUMNS if c in out.columns])
    before = len(out)
    out = out.drop_duplicates(subset="image_id", keep="first")
    n_dupes = before - len(out)
    # shuffle so ordering in the file can never bias a split
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.attrs["n_duplicates_removed"] = n_dupes
    return out


def load_clean(source: str, from_disk: bool = True, seed: int = 42):
    """Convenience: return (ds, clean_metadata_df) ready for auditing/splitting."""
    ds = load_pooled(source, from_disk=from_disk)
    df = clean_metadata(metadata_frame(ds), seed=seed)
    return ds, df


# ---------------------------------------------------------------------------
# fetching an image
# ---------------------------------------------------------------------------
def get_image(ds: Dataset, row: int) -> Image.Image:
    """Return the RGB PIL image for dataset position `row` (the `_row` value).

    Robust to how `image` is stored: raw JPEG bytes, a datasets `Image` dict with a
    `bytes` field, or an already-decoded PIL image.
    """
    val = ds[int(row)]["image"]
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict):  # datasets Image feature -> {"bytes": ..., "path": ...}
        if val.get("bytes") is not None:
            return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
        return Image.open(val["path"]).convert("RGB")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val)).convert("RGB")
    raise TypeError(f"Unrecognised image storage type: {type(val)}")


def iter_images(ds: Dataset, rows):
    """Yield (row, PIL image) for a sequence of `_row` indices."""
    for r in rows:
        yield r, get_image(ds, r)
