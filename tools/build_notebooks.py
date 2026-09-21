"""Generate the Colab notebooks from plain Python (keeps them valid + easy to edit).

Run:  python tools/build_notebooks.py
Each notebook is a list of (kind, source) cells, where kind is "md" or "code".
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)


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
    path = NB_DIR / name
    nbf.write(nb, str(path))
    print("wrote", path)


# ===========================================================================
# 00_setup.ipynb
# ===========================================================================
SETUP = [
    ("md", """# Phase 0 — Setup

**Goal of this notebook:** get everything ready so the rest of the project can run on
a free GPU. By the end you'll have (1) the code, (2) the libraries, and (3) the
dataset saved permanently in your Google Drive.

### The three tools we use, in one sentence each
- **GitHub** — a website that stores our *code*. We copy ("clone") it into Colab.
- **Google Colab** — a free website that runs Python on Google's computers, *including a free GPU* (the fast chip that trains neural networks).
- **Google Drive** — your personal cloud storage. Colab forgets everything when you close it, so we park the *dataset* in Drive so it survives.

Run each cell below with **Shift+Enter**, top to bottom."""),

    ("md", """## Step 1 — Turn on the GPU

In the Colab menu: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
Then run the next cell to confirm the GPU is visible."""),
    ("code", """import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "— none (set Runtime → T4 GPU)")"""),

    ("md", """## Step 2 — Get the code from GitHub

Paste your repository's URL below (it looks like
`https://github.com/YOUR_USERNAME/pm25-visual-aq.git`). Running this clones the code
the first time, and `git pull` grabs the latest version every other time."""),
    ("code", """REPO_URL = "https://github.com/YOUR_USERNAME/pm25-visual-aq.git"   # <-- EDIT THIS

import os
if not os.path.isdir("/content/pm25-visual-aq"):
    !git clone $REPO_URL /content/pm25-visual-aq
%cd /content/pm25-visual-aq
!git pull
import sys; sys.path.insert(0, "/content/pm25-visual-aq")
print("Working in:", os.getcwd())"""),

    ("md", """## Step 3 — Install the libraries

`requirements.txt` lists everything the project needs. On Colab, PyTorch is already
installed with GPU support, so pip will skip it and keep the GPU build."""),
    ("code", """!pip install -q -r requirements.txt
print("Libraries installed.")"""),

    ("md", """## Step 4 — Connect Google Drive

A pop-up will ask you to allow access. This lets us save the dataset somewhere
permanent so we only download it once."""),
    ("code", """from google.colab import drive
drive.mount("/content/drive")
print("Drive connected.")"""),

    ("md", """## Step 5 — Load the dataset and save it to Drive (run once)

This downloads PM25Vision (~1 GB) on Google's fast network and saves it to your
Drive. The next time you run it, it detects the saved copy and skips the download."""),
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

    ("md", """## Step 6 — Confirm it worked

We print the columns and a sample image with its AQI label. You should see ~11,219
rows total and a street photo."""),
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

You now have the code, the libraries, and the dataset in Drive. **Next:** open
`notebooks/01_data_audit.ipynb`. If you close Colab and come back later, just re-run
Steps 1–4 (the dataset is already saved, so Step 5 is instant)."""),
]

# ===========================================================================
# 01_data_audit.ipynb
# ===========================================================================
AUDIT = [
    ("md", """# Phase 1 — Load, clean, and **audit** the data

**Why audit before modelling?** (paper §3.3) A model is only as trustworthy as the
data underneath it. Before training anything we check three things:

1. **Duplicates / redundancy** — street photos taken seconds apart look identical. If
   near-duplicates end up on both sides of a train/test split, the model can "cheat"
   by recognising a scene it already saw. We measure how much redundancy exists.
2. **Images per station** — decides whether our leakage-safe split (grouping by
   station) is feasible.
3. **A physics sanity check** — hazier photos should have lower "transmission" and
   higher AQI. If that relationship is missing, the images and labels were mis-paired,
   and nothing downstream would be meaningful.

We also **clean** the data: drop dead columns, remove duplicate rows, and shuffle."""),

    ("md", "## Bootstrap (connect Drive + make `src` importable)"),
    ("code", """import os, sys
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    if os.path.isdir("/content/pm25-visual-aq"):
        os.chdir("/content/pm25-visual-aq")
except ImportError:
    pass  # running locally, not on Colab
sys.path.insert(0, os.getcwd())

from src.config import load_config
from src import data, audit
cfg = load_config()

# Full dataset (Colab). For a quick laptop test, set SOURCE = "tests/fixture_ds".
SOURCE = cfg["data"]["drive_path"]
print("Loading from:", SOURCE)"""),

    ("md", """## Load + clean

`load_clean` pools the train/test splits into one pool (we make our own splits in
Phase 2), drops columns that carry no signal, removes duplicate `image_id` rows, and
shuffles with a fixed seed so file ordering can never bias a split."""),
    ("code", """ds, df = data.load_clean(SOURCE, from_disk=True, seed=cfg["seed"])
print("rows after cleaning:", len(df))
print("duplicate rows removed:", df.attrs.get("n_duplicates_removed"))
df.head()"""),

    ("md", """## The label: it's an **AQI index**, not µg/m³

PM25Vision's label is a US-EPA Air Quality Index value (roughly 1–530), a unitless
index — *not* a raw concentration. Every error we report later is in **AQI points**.
The histogram is right-skewed (many moderate days, few extreme ones), which is why we
later train on `log(AQI)`."""),
    ("code", """import matplotlib.pyplot as plt
plt.figure(figsize=(7,3))
plt.hist(df["pm25"], bins=50)
plt.xlabel("pm25 (AQI index)"); plt.ylabel("number of photos")
plt.title("Label distribution — right-skewed"); plt.show()
print(df["pm25"].describe())"""),

    ("md", """## Images per station

This decides whether we can split *by station* (our leakage-safe protocol). With
thousands of stations and most contributing only a couple of images, grouping is
easy. The few stations with many images are where near-duplicate risk concentrates."""),
    ("code", """sps = audit.images_per_station(df, station_col=cfg["data"]["station_col"])
print("stations: %d | images/station  median=%.1f  mean=%.2f  max=%d  (%.0f%% have just 1)"
      % (sps["n_stations"], sps["median"], sps["mean"], sps["max"], sps["pct_single_image"]))
plt.figure(figsize=(7,3))
plt.hist(sps["counts"].values, bins=40)
plt.xlabel("images at one station"); plt.ylabel("number of stations")
plt.title("Most stations contribute only a few images"); plt.show()"""),

    ("md", """## Near-duplicate check (perceptual hashing)

A *perceptual hash* is a short fingerprint where **similar-looking images get similar
fingerprints**. We fingerprint every image and group ones whose fingerprints differ
by only a few bits. The **distinct-ratio** = groups ÷ images: 1.0 means no
near-duplicates; lower means redundancy we must keep out of the split.

> ⏳ On the full dataset this decodes ~11k images and takes a few minutes."""),
    ("code", """rep = audit.redundancy_report(ds, df, hash_size=8, max_distance=5)
print("images: %d | distinct groups: %d | distinct-ratio: %.3f | largest group: %d"
      % (rep["n_images"], rep["n_distinct_groups"], rep["distinct_ratio"], rep["largest_group"]))"""),

    ("md", """## Physics falsification test

Physics says hazier photos have **lower transmission** and **higher AQI**, so average
transmission should be **negatively** correlated with the label. A clearly negative
correlation means images and labels line up — the learning problem is real."""),
    ("code", """cor = audit.transmission_label_correlation(
    ds, df, target_col=cfg["data"]["target_col"], sample=1000, seed=cfg["seed"])
print("Pearson r = %.3f (p=%.1e)  |  Spearman r = %.3f  |  negative as expected? %s"
      % (cor["pearson_r"], cor["pearson_p"], cor["spearman_r"], cor["passes"]))
plt.figure(figsize=(5,4))
plt.scatter(cor["mean_transmission"], cor["labels"], s=6, alpha=0.4)
plt.xlabel("average transmission (clearer →)"); plt.ylabel("AQI label")
plt.title("Should slope downward"); plt.show()"""),

    ("md", """## Where in the world are these photos?

Coverage is heavily skewed toward East Asia, Europe, and India, with little in the
Americas or Africa. That's fine, but it means any "generalises everywhere" claim must
be scoped to the regions actually represented (paper §3.2)."""),
    ("code", """plt.figure(figsize=(8,4))
plt.scatter(df[cfg["data"]["lon_col"]], df[cfg["data"]["lat_col"]], s=4, alpha=0.3)
plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Station geography"); plt.show()"""),

    ("md", """## What we found (summary)

- Label is **AQI (1–530)**, right-skewed → we'll train on `log(AQI)`.
- **3,261 stations**, most with only a few images → station-grouped split is easy.
- Near-duplicate redundancy is concentrated in a handful of busy stations.
- The physics check should be **negative** — confirming images and labels match.
- Geography is skewed → we scope our claims honestly.

**Next:** `notebooks/02_splits.ipynb` — building leakage-safe train/calibration/test
splits, and measuring how much a naive random split inflates results."""),
]

if __name__ == "__main__":
    build("00_setup.ipynb", SETUP)
    build("01_data_audit.ipynb", AUDIT)
