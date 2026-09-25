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

# ===========================================================================
# 04_model.ipynb
# ===========================================================================
MODEL = [
    ("md", """# Phase 4 — The model

**Why (paper §3.6):** we use **EfficientNet-B0**, a compact CNN pretrained on ImageNet.
It's small (trains on our budget), its pretrained features help given our modest data,
and it's the strongest published baseline on PM25Vision — so our numbers are directly
comparable. Two tweaks:

1. **5-channel stem.** The first layer normally takes 3 channels (RGB). We widen it to 5
   (RGB + transmission + inverted saturation); `timm` seeds the two new channels from the
   pretrained RGB weights.
2. **Three monotone quantile heads.** We predict the 5th/50th/95th percentiles. To stop a
   "95th below the 50th" nonsense, we build them so each is the previous one **plus a
   strictly positive step** (`softplus`). They can never cross — guaranteed, not checked."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("code", """import torch
from src.config import load_config
from src import model as M
cfg = load_config()

net = M.build_model(cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(device)
print("device:", device)
print("backbone:", cfg["model"]["backbone"], "| input channels:", net.backbone.conv_stem.in_channels)
print("trainable parameters: %.2f M" % (M.count_parameters(net) / 1e6))"""),

    ("md", """## Two outputs: quantiles (for intervals) + a point estimate (for accuracy)

The model returns a dict: **`quantiles`** (the three ordered bounds — for the interval) and
**`point`** (a dedicated estimate trained with Huber loss — for the accuracy numbers; see
Phase 5b). We confirm the quantiles come out **ascending** (q05 ≤ q50 ≤ q95) for any input, by
construction."""),
    ("code", """net.eval()
with torch.no_grad():
    out = net(torch.randn(8, cfg["model"]["in_chans"], cfg["data"]["image_size"], cfg["data"]["image_size"]).to(device))
q = out["quantiles"]
print("output keys:", list(out.keys()))
print("quantiles shape (batch, 3):", tuple(q.shape), "| point shape (batch,):", tuple(out["point"].shape))
print("first sample (q05, q50, q95) in log-space:", [round(v, 3) for v in q[0].tolist()])
print("all samples ascending:", bool(torch.all(q[:, 1:] >= q[:, :-1])))"""),

    ("md", """## What's next

The model outputs three ordered numbers in **log(AQI)** space. Phase 5 trains it with the
**pinball loss** (so each head learns its percentile) on the log target, using only the
physics-safe augmentations. **Next:** `05_train.ipynb`."""),
]

# ===========================================================================
# 05_train.ipynb
# ===========================================================================
TRAIN = [
    ("md", """# Phase 5 — Training

**Why (paper §3.7, §3.12):** we train the three quantile heads with the **pinball loss**
so each head learns its percentile, on the **log(AQI)** target (the label is right-skewed,
so a few extreme days would otherwise dominate). Settings mirror the paper: AdamW
(lr 3e-4), cosine schedule with 2 warm-up epochs, batch 32, ≤40 epochs with early
stopping, mixed precision.

> ⏳ This is the long step — on a Colab T4 GPU, expect ~20–40 minutes. Make sure the GPU
> is on (Runtime → Change runtime type → T4) and the physics cache from Phase 3 exists.

**Accuracy upgrades on by default (Phase 5b):** a dedicated Huber **point-head** for the
accuracy numbers, **class-balanced sampling** (more rare high-AQI photos), and more epochs —
all set in `configs/default.yaml`. These target the extreme-AQI underprediction from the first run."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("code", """import os, torch, matplotlib.pyplot as plt
from src.config import load_config
from src import data, splits, physics, dataset as D, model as M, train as T
cfg = load_config()
T.set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

SOURCE = cfg["data"]["drive_path"]        # laptop test: "tests/fixture_ds"
STRATEGY = cfg["split"]["strategy"]        # station_grouped (primary)

ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
sp = splits.make_splits(df, strategy=STRATEGY, seed=cfg["seed"],
                        station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
                        lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
print("split:", STRATEGY, "->", sp["split"].value_counts().to_dict())"""),

    ("md", """## Load the cached maps and build the data loaders

If the physics cache is missing, run `03_physics_features.ipynb` first."""),
    ("code", """cache_path = os.path.join(cfg["data"]["cache_dir"], "physics_maps_%d.npy" % cfg["data"]["image_size"])
assert os.path.exists(cache_path), "Physics cache missing — run 03_physics_features.ipynb first."
cache = physics.load_map_cache(cache_path)

loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=2)
print("batches:", {k: len(v) for k, v in loaders.items()})"""),

    ("md", "## Train\nThe best model (lowest calibration loss) is saved to your Drive."),
    ("code", """net = M.build_model(cfg).to(device)
out_dir = os.path.join(cfg["data"]["outputs_dir"], STRATEGY)
history = T.train_model(net, loaders, cfg, device=device, out_dir=out_dir)
print("best epoch:", history["best_epoch"] + 1, "| best cal loss:", round(history["best_cal_loss"], 4))
print("checkpoint:", history["checkpoint"])"""),

    ("md", "## Learning curves\nTrain and calibration loss should fall and then flatten; early stopping keeps the best."),
    ("code", """plt.figure(figsize=(7,4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["cal_loss"], label="calibration")
plt.axvline(history["best_epoch"], ls="--", c="grey", label="best")
plt.xlabel("epoch"); plt.ylabel("pinball loss (log-AQI)"); plt.legend()
plt.title("Training curves"); plt.show()"""),

    ("md", """## What's next

We now have a trained model whose three outputs are *ordered* but **not yet guaranteed**
to contain the truth 90% of the time. **Next:** `06_calibrate_evaluate.ipynb` — conformal
calibration to earn that guarantee, then the full metrics."""),
]

# ===========================================================================
# 06_calibrate_evaluate.ipynb
# ===========================================================================
EVAL = [
    ("md", """# Phase 6 — Calibration + Evaluation  (completes the core pipeline)

**Why (paper §3.8, §3.11):** the trained model's three outputs are *ordered* but not yet
*guaranteed* to contain the truth 90% of the time. **Conformal prediction** earns that
guarantee using the held-out calibration set:

1. On calibration, measure how far each truth fell outside the predicted range.
2. Take the 90% mark of those misses → a single number **Q**.
3. Widen every interval by **Q** (clip the lower end at 0).

Then we report the full metrics — all in **AQI points** — and, if you trained more than one
split strategy, compare them side by side to expose the leakage gap."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("code", """import os, json, numpy as np, matplotlib.pyplot as plt
from src.config import load_config
from src import data, splits, physics, dataset as D, model as M, train as T
from src import calibrate as C, metrics as Mx
cfg = load_config()
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

SOURCE = cfg["data"]["drive_path"]        # laptop test: "tests/fixture_ds"
ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
cache = physics.load_map_cache(os.path.join(cfg["data"]["cache_dir"], "physics_maps_%d.npy" % cfg["data"]["image_size"]))
out_root = cfg["data"]["outputs_dir"]; os.makedirs(out_root, exist_ok=True)
print("device:", device)"""),

    ("md", """## Evaluate a trained model

`evaluate_strategy` rebuilds that split, loads its checkpoint, and predicts on calibration +
test. **Accuracy** (MAE/RMSE/R²) comes from the **point head**, de-standardised from its z-score
output back to AQI (it is trained with MSE, so it estimates the conditional mean — what R² rewards);
the **interval** (coverage/width) comes from the conformal-calibrated quantiles. It saves **Q** and
the point head's mean/std next to the checkpoint so the demo (Phase 10) can reuse them."""),
    ("code", """import numpy as np
def evaluate_strategy(strategy):
    sp = splits.make_splits(df, strategy=strategy, seed=cfg["seed"],
            station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
            lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
    loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=2)
    net = M.build_model(cfg).to(device)
    T.load_checkpoint(os.path.join(out_root, strategy, "best_model.pth"), net, map_location=device)
    cal = T.collect_outputs(net, loaders["cal"], device)
    test = T.collect_outputs(net, loaders["test"], device)
    # accuracy: de-standardised point head
    ymean, ystd = float(net.y_mean), float(net.y_std)
    point = T.point_to_aqi(test["point_out"], ymean, ystd)
    # intervals: conformal-calibrated quantiles
    Q = C.conformal_Q(np.exp(cal["q_log"]), cal["y_raw"], cfg["calibration"]["coverage"])
    intervals = C.apply_conformal(np.exp(test["q_log"]), Q)
    y = test["y_raw"]
    json.dump({"Q": Q, "y_mean": ymean, "y_std": ystd},
              open(os.path.join(out_root, strategy, "conformal_Q.json"), "w"))
    return {"strategy": strategy, "Q": Q, "point": point,
            "intervals": intervals, "y": y,
            "raw_coverage": C.coverage(np.exp(test["q_log"]), y),
            "report": Mx.report(point, intervals, y, cfg["calibration"]["coverage"])}

primary = evaluate_strategy(cfg["split"]["strategy"])
print("strategy:", primary["strategy"], "| Q = %.1f AQI" % primary["Q"])
print("coverage: raw %.3f -> calibrated %.3f (target %.2f)"
      % (primary["raw_coverage"], primary["report"]["coverage"], cfg["calibration"]["coverage"]))
{k: round(v,3) for k,v in primary["report"].items()}"""),

    ("md", "## Where is the model weak?\nErrors (of the point estimate) broken down by true-AQI band — watch the high-AQI bands, which Phase 5b targets."),
    ("code", """ebm = Mx.error_by_magnitude(primary["y"], primary["point"])
display(ebm)
plt.figure(figsize=(7,3)); plt.bar(ebm["band"], ebm["MAE"]); plt.ylabel("MAE (AQI)"); plt.xlabel("true AQI band")
plt.title("Error by pollution level"); plt.show()

w = primary["intervals"][:,2]-primary["intervals"][:,0]
plt.figure(figsize=(7,3)); plt.hist(w, bins=40)
plt.xlabel("interval width (AQI)"); plt.ylabel("count"); plt.title("Calibrated interval widths"); plt.show()"""),

    ("md", """## A few example predictions

Each column is a test photo: the dot is the point estimate, the bar is the calibrated 90%
interval, and the ✕ is the truth. Most truths should sit inside their bars."""),
    ("code", """idx = np.argsort(primary["y"])[::max(1, len(primary["y"])//40)][:40]
iv, pt, y = primary["intervals"][idx], primary["point"][idx], primary["y"][idx]
xs = np.arange(len(idx))
plt.figure(figsize=(9,4))
plt.vlines(xs, iv[:,0], iv[:,2], color="#4C78A8", lw=3, alpha=0.5, label="90% interval")
plt.plot(xs, pt, "o", ms=4, color="#4C78A8", label="point estimate")
plt.plot(xs, y, "x", ms=6, color="#E45756", label="truth")
plt.legend(); plt.xlabel("test photos (sorted by true AQI)"); plt.ylabel("AQI"); plt.title("Predictions vs truth"); plt.show()"""),

    ("md", """## Leakage gap: compare every split you trained

If you trained more than one split strategy (e.g. re-ran `05_train` with
`split.strategy: random`), this tabulates them side by side. **`random` should look better
than `station_grouped`** — that difference is leakage inflation, measured not assumed."""),
    ("code", """import pandas as pd
avail = [s for s in ["station_grouped","random","geographic","temporal"]
         if os.path.exists(os.path.join(out_root, s, "best_model.pth"))]
rows = []
for s in avail:
    r = evaluate_strategy(s)
    rows.append({"strategy": s, **{k: round(v,3) for k,v in r["report"].items()}})
table = pd.DataFrame(rows).set_index("strategy")
table.to_csv(os.path.join(out_root, "results_by_split.csv"))
table"""),

    ("md", """## Core pipeline complete ✓

You now have: leakage-safe evaluation, a physics-guided model, and **calibrated intervals
with ~90% coverage**, all in AQI points. Paste me this notebook's numbers and I'll sanity-check
them and write them into `docs/RESULTS.md`.

**Next (contributions):** `07_error_ceiling` (C2 — how much error is unavoidable) and
`08_abstention` (C3 — refusing to answer on unusable inputs)."""),
]

# ===========================================================================
# 07_error_ceiling.ipynb  (C2)
# ===========================================================================
CEILING = [
    ("md", """# Phase 7 — C2: the unavoidable-error ceiling

**Why (paper §3.10):** our labels are **daily averages**, but each photo is an **instant**.
Pollution genuinely swings within a day, so even a *perfect* model reading the exact
instantaneous pollution from a photo would be "wrong" versus a daily-average label. That
gap is noise no model can beat — a hard ceiling on achievable R².

We estimate **R²_max = 1 − Var(ε) / Var(y)** and report it explicitly as an **UPPER bound**
(it accounts for *label noise only* — a real model also loses accuracy to limited visual signal,
imbalance and domain shift, so the honest R² can sit well below R²_max with no bug). Two
best-practice refinements: `Var(y)` is the **station-grouped test split's** label variance (the
ceiling is split-specific), and `Var(ε)` is **level-reweighted** — we measure the within-day noise
*curve* v(m) by pollution level from **hourly** reference data (OpenAQ) and reweight it to
PM25Vision's own label mix p(m). We add a station cluster-**bootstrap 95% CI** and a
**2012-vs-2024 breakpoint** sensitivity check. This tells us how much room really remains above the
honest R²=0.22 (and whether the published R²=0.55 is even physically attainable)."""),

    ("md", BOOTSTRAP_MD),
    ("code", BOOTSTRAP),

    ("md", """## Get a free OpenAQ API key (2 minutes)

1. Go to **https://explore.openaq.org** → sign up (free).
2. Open your **account → API keys** and copy your key.
3. Paste it below. (It's kept only in this session — don't commit it.)"""),
    ("code", """OPENAQ_API_KEY = "PASTE_YOUR_OPENAQ_KEY_HERE"   # <-- from explore.openaq.org
assert OPENAQ_API_KEY != "PASTE_YOUR_OPENAQ_KEY_HERE", "Add your OpenAQ API key first." """),

    ("md", """## Fetch hourly PM2.5 from a region-stratified sample of stations

We pull ~40 stations' hourly readings over a recent 45-day window, **capping stations per country**
so the sample spreads across regions (not dominated by whichever country OpenAQ lists first). Exact
matching to PM25Vision's stations isn't needed — we want a defensible estimate of the within-day AQI
variance *curve*."""),
    ("code", """import datetime as dt
from src import ceiling as CE
to = dt.date.today()
frm = to - dt.timedelta(days=45)
hourly = CE.fetch_openaq_hourly(OPENAQ_API_KEY,
                                date_from=frm.isoformat(), date_to=to.isoformat(),
                                n_locations=40, max_per_country=6)
print("hourly rows fetched:", len(hourly), "| stations:", hourly["location_id"].nunique(),
      "| countries:", hourly["country"].nunique())
hourly.head()"""),

    ("md", "## Compute the ceiling (level-reweighted, split-specific, with a bootstrap CI)"),
    ("code", """from src.config import load_config
from src import data, splits as S
import numpy as np
cfg = load_config()

# The ceiling is SPLIT-SPECIFIC: Var(y) = the label variance of the split we headline
# (station-grouped, leakage-safe). p(m) for level-reweighting = the whole dataset's label mix.
_, df = data.load_clean(cfg["data"]["drive_path"], from_disk=True, seed=cfg["seed"])
sp = S.make_splits(df, strategy="station_grouped", seed=cfg["seed"],
                   station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
                   lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
var_labels = float(np.var(sp.loc[sp["split"] == "test", "pm25"].to_numpy()))
dataset_labels = df["pm25"].to_numpy()

res = CE.compute_ceiling(hourly, dataset_labels, var_labels, min_hours=18, n_boot=1000)
print("Var(eps) level-reweighted:        %.1f  (std ~ %.1f AQI)" % (res["var_epsilon"], res["var_epsilon"]**0.5))
print("Var(y) station-grouped test:      %.1f  (SD  ~ %.1f AQI)" % (var_labels, var_labels**0.5))
print("--------")
print("R2_max (UPPER bound, label noise only): %.3f  [95%% CI %.3f - %.3f]"
      % (res["R2_max"], res["R2_max_lo"], res["R2_max_hi"]))
print("honest R2 we measured (station-grouped): 0.220   |   published baseline: 0.550")
print("honest interval-width floor: ~%.1f AQI (a 90%% range narrower than this over-claims)" % res["interval_width_floor"])
print("based on %d station-days from %d stations" % (res["n_station_days"], res["n_stations"]))"""),

    ("md", "### The within-day noise curve v(m) + a 2012-vs-2024 breakpoint sensitivity check"),
    ("code", """import pandas as pd
from src.aqi import PM25_BREAKPOINTS_2024
print("v(m): within-day AQI variance by pollution level (this is the error-by-band story)")
display(pd.DataFrame(res["curve"])[["band", "n_days", "v", "rmse_within_day", "p_reweighted"]])

res_2024 = CE.compute_ceiling(hourly, dataset_labels, var_labels, min_hours=18,
                              table=PM25_BREAKPOINTS_2024, n_boot=200)
print("R2_max sensitivity to the AQI breakpoint convention:")
print("  historical (the labels' own convention): %.3f" % res["R2_max"])
print("  2024 EPA revision:                        %.3f" % res_2024["R2_max"])"""),

    ("md", """## Save for the paper"""),
    ("code", """import os, json
os.makedirs(cfg["data"]["outputs_dir"], exist_ok=True)
res["R2_max_2024"] = res_2024["R2_max"]
json.dump(res, open(os.path.join(cfg["data"]["outputs_dir"], "error_ceiling.json"), "w"),
          indent=2, default=float)
print("saved error_ceiling.json  (09_leakage draws its ceiling line from this file)")"""),

    ("md", """## Reading the result

- **R²_max is an UPPER bound** (label noise only). The honest gap between our 0.22 and R²_max is the
  room that *better vision* could recover; the gap between R²_max and 1.0 is forever lost to
  daily-average labels.
- **If R²(random) from `09_leakage` (0.759) exceeds R²_max, that split is *provably* leaky** — no
  honest model can beat the label-noise ceiling, so a score above it can only come from leakage. The
  `error_ceiling.json` saved here is what draws the ceiling line on the leakage-gradient figure.
- The **width floor** is a lower bound on honest 90% interval width: no interval should be narrower
  than the pollution's own within-day spread.
- The **95% CI** and the **2012-vs-2024** sensitivity show the estimate is robust, not a single fragile number.

Paste these numbers back and I'll write them into `docs/RESULTS.md`.

**Next:** `08_abstention.ipynb` (C3 — refusing to answer on unusable inputs)."""),
]

# ===========================================================================
# kaggle_pipeline.ipynb  (consolidated single-session runner for Kaggle GPU)
# ===========================================================================
KAGGLE_BOOTSTRAP = '''# === Kaggle bootstrap — RUN ME FIRST ===
# In the right sidebar: Accelerator = GPU T4, Internet = ON (needs a phone-verified account).
REPO_URL = "https://github.com/YOUR_USERNAME/pm25-visual-aq.git"   # <-- EDIT THIS

import os, sys, subprocess
REPO = "/kaggle/working/pm25-visual-aq"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=False)

import torch
print("repo:", os.getcwd())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Accelerator=GPU")'''

KAGGLE = [
    ("md", """# PM2.5 — full pipeline on Kaggle GPU (Stage A: maximize accuracy)

Colab's free GPU is rate-limited, so we run the whole thing here in **one session**: load data →
build physics features → train (with the accuracy upgrades) → calibrate → evaluate. It reuses the
same tested `src/` modules; the per-phase Colab notebooks and `docs/` have the full explanations.

**What "Stage A" does for accuracy (the honest, leakage-safe station-grouped split):**
- reports accuracy from a **standardised point head** (trained to hit the *mean*, not the biased
  log-median) — the main fix for the low R²;
- **class-balanced sampling** so the model sees the rare high-pollution photos;
- **stochastic-depth** regularisation + full convergence;
- everything selected on the **calibration** split; **test is evaluated once**.

**Before running:** right sidebar → Accelerator = **GPU T4**, Internet = **On**. Then set
`REPO_URL` below and **Run All**."""),

    ("md", "## 1 — Bootstrap (clone code, install, check GPU)"),
    ("code", KAGGLE_BOOTSTRAP),

    ("md", """## 2 — Load the dataset from Hugging Face

Kaggle has no Google Drive, so we stream PM25Vision straight from the Hub (~1 GB, a few minutes).
Then we clean it (drop dead columns, de-duplicate, shuffle)."""),
    ("code", """from src.config import load_config
from src import data, splits, physics, dataset as D, model as M, train as T
from src import calibrate as C, metrics as Mx, recalibrate as R
import os, json, numpy as np, matplotlib.pyplot as plt
cfg = load_config()
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

WORK = "/kaggle/working"
cache_path = os.path.join(WORK, "pm25_cache", "physics_maps_%d.npy" % cfg["data"]["image_size"])
out_dir = os.path.join(WORK, "pm25_outputs", cfg["split"]["strategy"])
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

ds, df = data.load_clean(cfg["data"]["hf_repo"], from_disk=False, seed=cfg["seed"])
print("clean rows:", len(df), "| dupes removed:", df.attrs.get("n_duplicates_removed"))"""),

    ("md", """## 3 — Build the physics-map cache (~15 min, once per session)

Dark Channel Prior transmission + inverted-saturation for every image, cached to
`/kaggle/working`. Re-running detects the existing cache and skips."""),
    ("code", """if os.path.exists(cache_path):
    print("cache exists:", cache_path)
else:
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cache_path,
        size=cfg["data"]["image_size"], patch=cfg["physics"]["dcp_patch"],
        omega=cfg["physics"]["dcp_omega"], top_frac=cfg["physics"]["atmos_top_frac"],
        t_min=cfg["physics"]["t_min"])
cache = physics.load_map_cache(cache_path)
print("cache:", cache.shape)"""),

    ("md", """## 4 — Leakage-safe split + train (~30 min on T4)

Station-grouped split (no station spans train/test), balanced loaders, EfficientNet-B0 + the
point head. Best model (lowest **calibration point-MAE**) is saved to `/kaggle/working`."""),
    ("code", """T.set_seed(cfg["seed"])
sp = splits.make_splits(df, strategy=cfg["split"]["strategy"], seed=cfg["seed"],
        station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
        lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
print("split:", sp["split"].value_counts().to_dict())

loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=2)
net = M.build_model(cfg).to(device)
print("backbone:", cfg["model"]["backbone"], "| params %.2fM" % (M.count_parameters(net)/1e6))
history = T.train_model(net, loaders, cfg, device=device, out_dir=out_dir)
print("best epoch:", history["best_epoch"]+1, "| best cal MAE: %.2f AQI" % history["best_cal_point_mae"])"""),

    ("md", "## 5 — Learning curve\nCalibration MAE (AQI) should fall then flatten; early stopping keeps the best."),
    ("code", """plt.figure(figsize=(7,4))
plt.plot(history["cal_point_mae"], label="calibration MAE")
plt.axvline(history["best_epoch"], ls="--", c="grey", label="best")
plt.xlabel("epoch"); plt.ylabel("MAE (AQI)"); plt.legend(); plt.title("Training curve"); plt.show()"""),

    ("md", """## 6 — Calibrate + evaluate (once, on test)

Accuracy from the point head; the 90% interval from the conformal-calibrated quantiles. We also
apply **isotonic recalibration** (fit on the calibration set) to remove the systematic tail bias,
and report **both** R² definitions (strict coefficient-of-determination `R2`, and squared-Pearson
`r2_pearson` that looser papers quote). All in AQI points."""),
    ("code", """# evaluate the BEST checkpoint (early stopping saved it), not the last epoch
T.load_checkpoint(os.path.join(out_dir, "best_model.pth"), net, map_location=device)
cal = T.collect_outputs(net, loaders["cal"], device)
test = T.collect_outputs(net, loaders["test"], device)
ymean, ystd = float(net.y_mean), float(net.y_std)
cal_point = T.point_to_aqi(cal["point_out"], ymean, ystd)
point = T.point_to_aqi(test["point_out"], ymean, ystd)

# isotonic recalibration: fit prediction->truth on CAL, apply to TEST (removes tail bias)
if cfg["train"].get("recalibrate", True):
    iso = R.fit_isotonic(cal_point, cal["y_raw"])
    point_final = R.apply_isotonic(iso, point)
else:
    point_final = point

Q = C.conformal_Q(np.exp(cal["q_log"]), cal["y_raw"], cfg["calibration"]["coverage"])
intervals = C.apply_conformal(np.exp(test["q_log"]), Q)
y = test["y_raw"]
rep_raw = Mx.report(point, intervals, y, cfg["calibration"]["coverage"])
rep = Mx.report(point_final, intervals, y, cfg["calibration"]["coverage"])

keys = ("MAE","RMSE","R2","r2_pearson","Spearman","coverage","mean_width")
print("split:", cfg["split"]["strategy"], "| conformal Q = %.1f AQI" % Q)
print("raw point   :", {k: round(rep_raw[k],3) for k in keys})
print("recalibrated:", {k: round(rep[k],3) for k in keys})
json.dump({"recalibrated": rep, "raw": rep_raw, "Q": Q, "split": cfg["split"]["strategy"]},
          open(os.path.join(out_dir, "results.json"), "w"), indent=2)
{k: round(rep[k],3) for k in keys}"""),

    ("md", "## 7 — Where is the error?\nMAE of the (recalibrated) point estimate by true-AQI band + calibrated interval widths."),
    ("code", """ebm = Mx.error_by_magnitude(y, point_final)
display(ebm)
plt.figure(figsize=(7,3)); plt.bar(ebm["band"], ebm["MAE"]); plt.xlabel("true AQI band"); plt.ylabel("MAE (AQI)")
plt.title("Error by pollution level"); plt.show()

idx = np.argsort(y)[::max(1, len(y)//40)][:40]
xs = np.arange(len(idx))
plt.figure(figsize=(9,4))
plt.vlines(xs, intervals[idx,0], intervals[idx,2], color="#4C78A8", lw=3, alpha=0.5, label="90% interval")
plt.plot(xs, point_final[idx], "o", ms=4, color="#4C78A8", label="point estimate")
plt.plot(xs, y[idx], "x", ms=6, color="#E45756", label="truth")
plt.legend(); plt.xlabel("test photos (sorted by true AQI)"); plt.ylabel("AQI"); plt.title("Predictions vs truth"); plt.show()"""),

    ("md", """## Done — send me these numbers

Paste the **recalibrated** metrics from step 6 (R2, r2_pearson, MAE, RMSE, coverage).

**Leakage check (optional, 1 extra run):** edit `configs/default.yaml` → `split.strategy: random`,
push, Run All again. If R² jumps toward ~0.5, that's the leakage the paper's 0.55 almost certainly
rode on — a headline finding, measured on our own pipeline.

**To keep your model:** **Save Version** (top right) → the checkpoint under
`/kaggle/working/pm25_outputs/…` persists as the notebook Output. Next levers if needed: bigger
backbone (`model.backbone`), EMA/TTA, seed ensemble."""),
]

# ===========================================================================
# 09_leakage.ipynb  (measure the leakage gap — the headline "story" result)
# ===========================================================================
LEAKAGE = [
    ("md", """# The leakage gap — why the benchmark's 0.55 isn't what it seems

**Goal:** show that the honest accuracy depends enormously on *how you split the data*, and that
the benchmark's high R² is consistent with **station-level leakage** (photos from the same monitor —
which share a near-identical daily-average label — landing in both train and test).

We do two things in one session:
1. **A split gradient** — train the *same* model on `random`, `temporal`, `station_grouped`,
   `geographic` (and `shipped` as reference). Only the split changes. Expect
   R²(random) ≫ R²(station_grouped).
2. **A causal check (no extra training)** — inside the `random` run, compare accuracy on
   *contaminated* test photos (same station/near-duplicate also in train) vs *clean* ones. If the
   inflation lives in the contaminated part, leakage is the cause — not the test set being easier.

> ⏳ This trains several models in one go (~2–3 h on a T4). Run it once and leave it. It saves a
> results table you can paste back."""),

    ("md", BOOTSTRAP_MD),
    ("code", KAGGLE_BOOTSTRAP),

    ("code", """import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.config import load_config
from src import data, physics, audit, leakage
cfg = load_config()
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
WORK = "/kaggle/working"
cache_path = os.path.join(WORK, "pm25_cache", "physics_maps_%d.npy" % cfg["data"]["image_size"])
out_root = os.path.join(WORK, "pm25_outputs"); os.makedirs(os.path.dirname(cache_path), exist_ok=True)

ds, df = data.load_clean(cfg["data"]["hf_repo"], from_disk=False, seed=cfg["seed"])
print("rows:", len(df))
if not os.path.exists(cache_path):
    physics.build_map_cache(len(ds), lambda i: data.get_image(ds, i), cache_path,
        size=cfg["data"]["image_size"], patch=cfg["physics"]["dcp_patch"],
        omega=cfg["physics"]["dcp_omega"], top_frac=cfg["physics"]["atmos_top_frac"], t_min=cfg["physics"]["t_min"])
cache = physics.load_map_cache(cache_path)"""),

    ("md", """## Train each split (same model, same seed — only the split differs)

To save time you can trim `STRATEGIES` to `["random","station_grouped"]` (the core comparison)."""),
    ("code", """STRATEGIES = ["random", "temporal", "station_grouped", "geographic", "shipped"]
results, runs = [], {}
for strat in STRATEGIES:
    print("\\n=== training:", strat, "===")
    r = leakage.train_and_evaluate(ds, df, cache, cfg, strat, device, out_root, verbose=False)
    runs[strat] = r
    rep = r["report"]
    results.append({"split": strat, "R2": rep["R2"], "r2_pearson": rep["r2_pearson"],
                    "MAE": rep["MAE"], "RMSE": rep["RMSE"], "SD_y_test": rep["SD_y_test"],
                    "Spearman": rep["Spearman"], "coverage": rep["coverage"],
                    "n_train": rep["n_train"], "n_test": rep["n_test"],
                    "straddling": rep["straddling_stations"]})
table = pd.DataFrame(results).set_index("split").round(3)
table.to_csv(os.path.join(out_root, "leakage_gradient.csv"))
table"""),

    ("md", """## The headline number

`Δ_leak` = how much a leaky random split inflates R² over the honest station-grouped split. We
also show MAE (which, unlike R², doesn't depend on the test-set variance)."""),
    ("code", """r_rand = table.loc["random"]; r_grp = table.loc["station_grouped"]
print("Delta_leak (R2)  = %.3f  (random %.3f  vs  station_grouped %.3f)" % (r_rand.R2 - r_grp.R2, r_rand.R2, r_grp.R2))
print("Delta_leak (MAE) = %.1f AQI (random %.1f vs station_grouped %.1f)" % (r_grp.MAE - r_rand.MAE, r_rand.MAE, r_grp.MAE))
print("published baseline R2 = 0.55")"""),

    ("md", """## Causal check: contaminated vs clean (random split, no extra training)

We fingerprint every image (perceptual hash) and mark a random-split TEST photo as *contaminated*
if its station or its near-duplicate group also appears in the random TRAIN set."""),
    ("code", """rep = audit.redundancy_report(ds, df, hash_size=8, max_distance=5)
dup_df = rep["df_with_groups"][["_row", "dup_group"]]
cc = leakage.contaminated_vs_clean(runs["random"]["sp"], runs["random"]["test_df"], dup_df)
print("contaminated fraction of random test set: %.1f%%" % (100*cc["contaminated_fraction"]))
print("contaminated:", {k: round(v,3) if isinstance(v,float) else v for k,v in cc["contaminated"].items()})
print("clean       :", {k: round(v,3) if isinstance(v,float) else v for k,v in cc["clean"].items()})
print("(expect R2(contaminated) >> R2(clean) ~ R2(station_grouped) - inflation lives in the leaked part)")
json.dump(cc, open(os.path.join(out_root, "contaminated_vs_clean.json"), "w"), indent=2)"""),

    ("md", "## Gradient figure\nR² by split, with the paper's 0.55 and (if available) the C2 error-ceiling line."),
    ("code", """order = [s for s in ["random","temporal","shipped","station_grouped","geographic"] if s in table.index]
vals = table.loc[order, "R2"]
plt.figure(figsize=(8,4)); plt.bar(order, vals, color="#4C78A8")
plt.axhline(0.55, ls="--", c="#E45756", label="paper baseline 0.55")
cj = os.path.join(out_root, "error_ceiling.json")
if os.path.exists(cj):
    r2max = json.load(open(cj)).get("R2_max")
    if r2max is not None: plt.axhline(r2max, ls=":", c="green", label="C2 ceiling R²_max")
plt.ylabel("R² (coefficient of determination)"); plt.title("Leakage gradient: how you split changes the score")
plt.legend(); plt.xticks(rotation=15); plt.tight_layout(); plt.show()"""),

    ("md", """## Done — paste me the table + the two contaminated/clean lines

This is your paper's headline: the honest number, the leaky number, and *proof* the gap is leakage.
Save Version to keep the checkpoints + `leakage_gradient.csv`."""),
]

if __name__ == "__main__":
    build("00_setup.ipynb", SETUP)
    build("01_data_audit.ipynb", AUDIT)
    build("02_splits.ipynb", SPLITS)
    build("03_physics_features.ipynb", PHYSICS)
    build("04_model.ipynb", MODEL)
    build("05_train.ipynb", TRAIN)
    build("06_calibrate_evaluate.ipynb", EVAL)
    build("07_error_ceiling.ipynb", CEILING)
    build("kaggle_pipeline.ipynb", KAGGLE)
    build("09_leakage.ipynb", LEAKAGE)
