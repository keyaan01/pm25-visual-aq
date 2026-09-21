"""Generate the Colab notebooks from plain Python (keeps them valid + easy to edit).

Run:  python tools/build_notebooks.py
Each notebook is a list of (kind, source) cells, where kind is "md" or "code".

Every notebook starts with the same SELF-CONTAINED bootstrap cell: it clones the repo
if needed, installs requirements, mounts Drive, and puts the repo on the import path.
That way a notebook works even when opened straight from Colab's GitHub tab (which
otherwise loads only the single .ipynb and triggers `ModuleNotFoundError: No module
named 'src'`).
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)


# The one bootstrap cell reused at the top of every notebook. The user sets REPO_URL
# once. It is safe to re-run and works locally too (no clone if `src/` is already here).
BOOTSTRAP = '''# === Bootstrap — RUN ME FIRST (set REPO_URL to your repo) ===
REPO_URL = "https://github.com/YOUR_USERNAME/pm25-visual-aq.git"   # <-- EDIT THIS

import os, sys, subprocess

def _find_repo_root():
    # Are we already inside the repo (or just above the notebooks/ folder)?
    for cand in (".", "..", "pm25-visual-aq"):
        if os.path.isdir(os.path.join(cand, "src")):
            return os.path.abspath(cand)
    return None

_root = _find_repo_root()
if _root is None:                       # fresh Colab session: clone the code
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "pm25-visual-aq"], check=True)
    _root = os.path.abspath("pm25-visual-aq")
os.chdir(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=False)
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

print("repo root:", _root, "| Colab:", IN_COLAB)'''

BOOTSTRAP_MD = """## Bootstrap — run this first

This one cell makes the notebook self-contained: it grabs the code from GitHub (if it
isn't already here), installs the libraries, connects Google Drive, and makes our `src`
modules importable. **Set `REPO_URL` to your repository's URL.** It's safe to re-run and
also works on a laptop."""


def build(name: str, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, str(NB_DIR / name))
    print("wrote", NB_DIR / name)


# ===========================================================================
# 00_setup.ipynb
# ===========================================================================
SETUP = [
    ("md", """# Phase 0 — Setup

**Goal:** get everything ready to run on a free GPU. By the end you'll have (1) the
code, (2) the libraries, and (3) the dataset saved permanently in your Google Drive.

### The three tools, one sentence each
- **GitHub** stores our *code*; Colab copies ("clones") it.
- **Google Colab** runs the code on Google's computers, *with a free GPU*.
- **Google Drive** is permanent storage; Colab forgets everything when closed, so the *dataset* lives in Drive.

Run each cell with **Shift+Enter**, top to bottom."""),

    ("md", """## Step 1 — Turn on the GPU

Colab menu: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**,
then run the next cell to confirm."""),
    ("code", """import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "— none (set Runtime → T4 GPU)")"""),

    ("md", BOOTSTRAP_MD + "\n\nHere it also downloads the code and connects Drive for the first time."),
    ("code", BOOTSTRAP),

    ("md", """## Step 2 — Load the dataset and save it to Drive (run once)

This downloads PM25Vision (~1 GB) on Google's fast network and saves it to your Drive.
Every later session detects the saved copy and skips the download."""),
    ("code", """from datasets import load_dataset, load_from_disk
from src.config import load_config

cfg = load_config()
drive_path = cfg["data"]["drive_path"]

if os.path.exists(drive_path):
    print("Already saved at", drive_path)
    ds = load_from_disk(drive_path)
else:
    print("Downloading", cfg["data"]["hf_repo"], "…")
    ds = load_dataset(cfg["data"]["hf_repo"])
    ds.save_to_disk(drive_path)
    print("Saved to", drive_path)

print(ds)"""),

    ("md", """## Step 3 — Confirm it worked

You should see ~11,219 rows total, AQI ranging ~1–530, and a street photo."""),
    ("code", """import numpy as np, io
from PIL import Image
import matplotlib.pyplot as plt

train = ds["train"]
pm = np.array(train["pm25"])
print("columns:", train.column_names)
print("rows: train=%d test=%d" % (len(ds["train"]), len(ds["test"])))
print("pm25 (AQI) min/median/max: %.0f / %.0f / %.0f" % (pm.min(), np.median(pm), pm.max()))

row = train[0]
img = row["image"]
img = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img)).convert("RGB")
plt.imshow(img); plt.title("pm25 (AQI) = %.0f" % row["pm25"]); plt.axis("off"); plt.show()"""),

    ("md", """## Done ✓

**Next:** open `notebooks/01_data_audit.ipynb`. Coming back later? Just re-run the
bootstrap cell (the dataset is already in Drive, so Step 2 is instant)."""),
]

# ===========================================================================
# 01_data_audit.ipynb
# ===========================================================================
AUDIT = [
    ("md", """# Phase 1 — Load, clean, and **audit** the data

**Why audit before modelling?** (paper §3.3) A model is only as trustworthy as its
data. Before training we check three things:

1. **Duplicates / redundancy** — street photos taken seconds apart look identical; if
   near-duplicates land on both sides of a split, the model can "cheat" by recognising
   a scene it already saw. We measure how much redundancy exists.
2. **Images per station** — decides whether our leakage-safe split (grouping by
   station) is feasible.
3. **A physics sanity check** — hazier photos should have lower transmission and higher
   AQI; if not, images and labels were mis-paired.

We also **clean** the data: drop dead columns, remove duplicate rows, and shuffle."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("md", "Now import our modules and pick the data source."),
    ("code", """from src.config import load_config
from src import data, audit
cfg = load_config()

# Full dataset (Colab). For a quick laptop test, set SOURCE = "tests/fixture_ds".
SOURCE = cfg["data"]["drive_path"]
print("Loading from:", SOURCE)"""),

    ("md", """## Load + clean

`load_clean` pools the train/test splits into one pool (we make our own splits in
Phase 2), drops columns with no signal, removes duplicate `image_id` rows, and
shuffles with a fixed seed so file ordering can't bias a split."""),
    ("code", """ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
print("rows after cleaning:", len(df))
print("duplicate rows removed:", df.attrs.get("n_duplicates_removed"))
df.head()"""),

    ("md", """## The label: an **AQI index**, not µg/m³

PM25Vision's label is a US-EPA Air Quality Index value (~1–530), a unitless index —
*not* a raw concentration. Every error we report is in **AQI points**. The histogram is
right-skewed, which is why we later train on `log(AQI)`."""),
    ("code", """import matplotlib.pyplot as plt
plt.figure(figsize=(7,3))
plt.hist(df["pm25"], bins=50)
plt.xlabel("pm25 (AQI index)"); plt.ylabel("number of photos")
plt.title("Label distribution — right-skewed"); plt.show()
print(df["pm25"].describe())"""),

    ("md", """## Images per station

Decides whether we can split *by station* (our leakage-safe protocol). With thousands
of stations, most contributing only a couple of images, grouping is easy. The few busy
stations are where near-duplicate risk concentrates."""),
    ("code", """sps = audit.images_per_station(df, station_col=cfg["data"]["station_col"])
print("stations: %d | images/station  median=%.1f  mean=%.2f  max=%d  (%.0f%% have just 1)"
      % (sps["n_stations"], sps["median"], sps["mean"], sps["max"], sps["pct_single_image"]))
plt.figure(figsize=(7,3))
plt.hist(sps["counts"].values, bins=40)
plt.xlabel("images at one station"); plt.ylabel("number of stations")
plt.title("Most stations contribute only a few images"); plt.show()"""),

    ("md", """## Near-duplicate check (perceptual hashing)

A *perceptual hash* is a fingerprint where **similar images get similar fingerprints**.
We fingerprint every image and group ones whose fingerprints differ by only a few bits.
The **distinct-ratio** = groups ÷ images: 1.0 means no near-duplicates; lower means
redundancy we must keep out of the split.

> ⏳ On the full dataset this decodes ~11k images and takes a few minutes."""),
    ("code", """rep = audit.redundancy_report(ds, df, hash_size=8, max_distance=5)
print("images: %d | distinct groups: %d | distinct-ratio: %.3f | largest group: %d"
      % (rep["n_images"], rep["n_distinct_groups"], rep["distinct_ratio"], rep["largest_group"]))"""),

    ("md", """## Physics falsification test

Physics says hazier photos have **lower transmission** and **higher AQI**, so average
transmission should be **negatively** correlated with the label. A clearly negative
correlation means images and labels line up."""),
    ("code", """cor = audit.transmission_label_correlation(
    ds, df, target_col=cfg["data"]["target_col"], sample=1000, seed=cfg["seed"])
print("Pearson r = %.3f (p=%.1e)  |  Spearman r = %.3f  |  negative as expected? %s"
      % (cor["pearson_r"], cor["pearson_p"], cor["spearman_r"], cor["passes"]))
plt.figure(figsize=(5,4))
plt.scatter(cor["mean_transmission"], cor["labels"], s=6, alpha=0.4)
plt.xlabel("average transmission (clearer →)"); plt.ylabel("AQI label")
plt.title("Should slope downward"); plt.show()"""),

    ("md", """## Where in the world are these photos?

Coverage skews to East Asia, Europe, and India, with little in the Americas or Africa.
That's fine, but any "generalises everywhere" claim must be scoped to the regions
actually represented (paper §3.2)."""),
    ("code", """plt.figure(figsize=(8,4))
plt.scatter(df[cfg["data"]["lon_col"]], df[cfg["data"]["lat_col"]], s=4, alpha=0.3)
plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Station geography"); plt.show()"""),

    ("md", """## What we found (summary)

- Label is **AQI (1–530)**, right-skewed → we'll train on `log(AQI)`.
- **3,261 stations**, most with only a few images → station-grouped split is easy.
- Near-duplicate redundancy is concentrated in a handful of busy stations.
- The physics check should be **negative** — confirming images and labels match.
- Geography is skewed → we scope our claims honestly.

**Next:** `02_splits.ipynb` — leakage-safe train/calibration/test splits, and measuring
how much a naive random split inflates results."""),
]

# ===========================================================================
# 02_splits.ipynb
# ===========================================================================
SPLITS = [
    ("md", """# Phase 2 — Leakage-safe splits

**Why (paper §3.4):** to know if the model *learned haze* rather than *memorised
places*, we control how photos are divided into **train / calibration / test**
(65% / 15% / 20%). We compare four strategies:

- **random** — the *leaky control*: near-duplicate photos from one place can land on
  both sides, inflating scores.
- **station_grouped** — all photos from a station go to one split → tested on unseen
  places. **Our primary protocol.**
- **geographic** — whole regions held out (a tougher test).
- **temporal** — train on earlier years, test on later ones.

The **calibration** split is kept separate because Phase 6's guarantee depends on it."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("code", """from src.config import load_config
from src import data, splits
import pandas as pd
cfg = load_config()

SOURCE = cfg["data"]["drive_path"]        # laptop test: "tests/fixture_ds"
ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
print("clean rows:", len(df))"""),

    ("md", """## Compare all four strategies

The key column is **`stations_straddling_splits`**: how many stations have photos in
more than one split. For a leakage-safe split this must be **0**. `random` and
`temporal` will show some straddling — that's the leakage we want to *measure*, not
hide."""),
    ("code", """rows = []
for strat in ["random", "station_grouped", "geographic", "temporal"]:
    s = splits.make_splits(
        df, strategy=strat, fractions=(cfg["split"]["train"], cfg["split"]["calibration"], cfg["split"]["test"]),
        seed=cfg["seed"], station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
        lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
    r = splits.split_report(s, station_col=cfg["data"]["station_col"])
    rows.append({"strategy": strat, **r["counts"],
                 "straddling_stations": r["stations_straddling_splits"]})
pd.DataFrame(rows).set_index("strategy")"""),

    ("md", """## See the geographic hold-out on a map

Colour each photo by which split it landed in under the **geographic** strategy. Whole
regions are one colour — the test regions are places the model never trains on."""),
    ("code", """import matplotlib.pyplot as plt
g = splits.make_splits(df, strategy="geographic", seed=cfg["seed"],
                       lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
colors = {"train": "#4C78A8", "cal": "#F58518", "test": "#E45756"}
plt.figure(figsize=(9,4))
for name, c in colors.items():
    sub = g[g.split == name]
    plt.scatter(sub[cfg["data"]["lon_col"]], sub[cfg["data"]["lat_col"]], s=6, alpha=0.5, c=c, label=name)
plt.legend(); plt.xlabel("longitude"); plt.ylabel("latitude")
plt.title("Geographic split — whole regions held out"); plt.show()"""),

    ("md", """## What this sets up

We'll train and evaluate under **station_grouped** (primary) and report **random**
alongside — the gap between them is an honest measurement of how much near-duplicate
leakage inflates results.

**Next:** `03_physics_features.ipynb` — turning each photo into the 5-channel input
(RGB + transmission + inverted saturation)."""),
]

# ===========================================================================
# 03_physics_features.ipynb
# ===========================================================================
PHYSICS = [
    ("md", """# Phase 3 — Physics features (the 5-channel input)

**Why (paper §3.5):** instead of hoping the network discovers the optics of haze from
raw pixels, we hand it two pre-computed maps that come from physics and behave the same
in every city:

- **Transmission** (Dark Channel Prior): how much of the scene's light survived the trip
  to the camera. **Low where haze is dense.** Clear photos always have *something* dark in
  each small patch; haze lifts those dark pixels.
- **Inverted saturation**: high where colour is washed out. Works on **sky**, exactly
  where the Dark Channel Prior fails (sky has nothing dark).

The model input becomes a **5-channel image**: Red, Green, Blue, transmission, inverted
saturation. We compute the two maps **once and cache them** — recomputing every epoch
would be far too slow."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("code", """from src.config import load_config
from src import data, physics
import numpy as np, matplotlib.pyplot as plt, os
cfg = load_config()

SOURCE = cfg["data"]["drive_path"]        # laptop test: "tests/fixture_ds"
ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
print("rows:", len(df))"""),

    ("md", """## See the maps

For a few photos spanning low → high AQI, we show the RGB image next to its transmission
and inverted-saturation maps. Look for: hazier (higher-AQI) photos tend to be *brighter*
(less dark) in the transmission map, and sky lights up in the inverted-saturation map."""),
    ("code", """qs = df["pm25"].quantile([0.1, 0.5, 0.9]).values
picks = [(df["pm25"] - v).abs().idxmin() for v in qs]

fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
for ax_row, idx in zip(axes, picks):
    row = df.loc[idx]
    rgb, t, s = physics.compute_maps(data.get_image(ds, int(row["_row"])), size=cfg["data"]["image_size"])
    ax_row[0].imshow(rgb); ax_row[0].set_title("RGB  (AQI=%.0f)" % row["pm25"])
    ax_row[1].imshow(t, cmap="viridis", vmin=0, vmax=1); ax_row[1].set_title("transmission")
    ax_row[2].imshow(s, cmap="magma", vmin=0, vmax=1); ax_row[2].set_title("inverted saturation")
    for a in ax_row: a.axis("off")
plt.tight_layout(); plt.show()"""),

    ("md", """## Build the physics-map cache (run once)

We precompute both maps for every image and store them to Drive as one memory-mapped
`uint8` array (~1.1 GB). Phase 5 reads from this cache instead of recomputing the Dark
Channel Prior on every epoch.

> ⏳ On the full dataset this takes ~10–15 minutes. It's idempotent — re-running detects
> the existing cache and skips."""),
    ("code", """cache_dir = cfg["data"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)
cache_path = os.path.join(cache_dir, "physics_maps_%d.npy" % cfg["data"]["image_size"])

if os.path.exists(cache_path):
    cache = physics.load_map_cache(cache_path)
    print("cache already exists:", cache_path, cache.shape)
else:
    physics.build_map_cache(
        len(ds), lambda i: data.get_image(ds, i), cache_path,
        size=cfg["data"]["image_size"], patch=cfg["physics"]["dcp_patch"],
        omega=cfg["physics"]["dcp_omega"], top_frac=cfg["physics"]["atmos_top_frac"],
        t_min=cfg["physics"]["t_min"])
    cache = physics.load_map_cache(cache_path)
    print("built cache:", cache_path, cache.shape)"""),

    ("md", """## Confirm the 5-channel input

We assemble one training tensor to check its shape (5, 224, 224) and channel ranges: the
RGB channels are ImageNet-normalised (roughly centred on 0); the two physics channels
stay in [0, 1]."""),
    ("code", """r = int(df["_row"].iloc[0])
x = physics.five_channel_cached(data.get_image(ds, r), cache, r, size=cfg["data"]["image_size"])
print("input shape:", x.shape)
print("RGB means:", [round(float(x[c].mean()), 2) for c in range(3)])
print("transmission range: [%.3f, %.3f]" % (x[3].min(), x[3].max()))
print("inv-sat range:      [%.3f, %.3f]" % (x[4].min(), x[4].max()))"""),

    ("md", """## What's next

Every photo can now become a 5-channel tensor quickly (RGB decoded on the fly + maps
from cache). **Next:** `04_model.ipynb` — the EfficientNet-B0 backbone widened to 5
input channels, with three monotone quantile heads."""),
]

if __name__ == "__main__":
    build("00_setup.ipynb", SETUP)
    build("01_data_audit.ipynb", AUDIT)
    build("02_splits.ipynb", SPLITS)
    build("03_physics_features.ipynb", PHYSICS)
