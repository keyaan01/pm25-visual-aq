# Physics-Guided Visual Air-Quality Estimation

Estimate PM2.5 air quality (as a US-EPA **AQI index**, 1–530 — *not* µg/m³) from a
single street-level photo, and do it **honestly**:

- output a **calibrated low / median / high range** instead of one number,
- **refuse to answer** on unusable inputs (night, indoor, no sky),
- estimate how much error is **physically unavoidable**,
- and **evaluate without leakage** (splitting by station / region / time).

This repo implements the paper *"Physics-Guided Visual Air Quality Estimation with
Calibrated Prediction Intervals and Inference-Time Abstention."* It is written for a
beginner: every phase has a notebook that explains what/why/how as it runs, mirrored
by a plain-English write-up in [`docs/`](docs/).

## Headline result — the leakage story

The full results ledger is [`docs/RESULTS.md`](docs/RESULTS.md); the essential code behind it is
walked through in [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md); the plain-English story is
[`docs/10_leakage.md`](docs/10_leakage.md).

- **Honest accuracy** (station-grouped, leakage-safe): **R² = 0.220**, MAE 54.2 AQI, Spearman 0.65,
  90% intervals ≈ calibrated (0.87).
- **Measured leakage inflation:** a leaky *random* split scores **R² = 0.759** — above the benchmark's
  reported 0.55, while **no station-disjoint split reaches 0.55**. **Δ_leak(R²) = 0.539.**
- **Proof it's leakage** (not an easier test set): inside the random split, *contaminated* photos score
  R² = 0.76 while *clean* (genuinely unseen) photos score **−0.86**.

The deliverable is the honest number **plus** the measured, causally-demonstrated leakage gap — not
matching a score that only leakage can produce.

## How it fits together

| Folder | What's in it |
|--------|--------------|
| `src/` | Small reusable Python modules (the real logic), imported by the notebooks. |
| `notebooks/` | One Colab notebook per phase — this is where you run things and read the explanations. |
| `docs/` | Standalone Markdown you can read without running anything — including [`RESULTS.md`](docs/RESULTS.md) (results ledger) and [`CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) (the essential code, explained). |
| `configs/default.yaml` | Every hyperparameter in one place. |
| `tests/` | A tiny fixture + smoke-tests so the code can be checked on a laptop (no GPU). |
| `data/`, `outputs/` | Created at run time (git-ignored). |

## The pipeline (maps to the paper)

| Phase | Notebook | Paper § | What it does |
|-------|----------|---------|--------------|
| 0 | `00_setup` | — | Clone repo, install deps, load the dataset into Google Drive. |
| 1 | `01_data_audit` | 3.2–3.3 | Load + clean data; check duplicates, stations, physics sanity. |
| 2 | `02_splits` | 3.4 | Leakage-safe train/calibration/test splits. |
| 3 | `03_physics_features` | 3.5 | Dark Channel Prior + inverted-saturation → 5-channel input. |
| 4 | `04_model` | 3.6 | EfficientNet-B0 with monotone quantile heads. |
| 5 | `05_train` | 3.7, 3.12 | Train with pinball loss on log(AQI). |
| 6 | `06_calibrate_evaluate` | 3.8, 3.11 | Conformal calibration + metrics. **← core pipeline done** |
| A | `09_leakage` | 3.4 | **Leakage measurement** — the split gradient + contaminated-vs-clean proof. ✅ |
| CV | `10_crossval` | 3.11 | Leakage-safe 5-fold (GroupKFold by station) → R² mean ± CI. |
| 7 | `07_error_ceiling` | 3.10 | C2: unavoidable-error ceiling from OpenAQ hourly data. |
| 8 | `08_abstention` | 3.9 | C3: refuse to answer on unusable inputs. |
| 9 | `09_ablations` | 3.11 | Ablations + write-up. |

## Running it

You need a free cloud GPU. This project runs on **[Kaggle Notebooks](https://www.kaggle.com/code)**
(free T4, ~30 h/week) — set Accelerator = GPU T4 and Internet = On, paste your repo URL in the first
cell, and for anything longer than ~30 min use **Save Version → Save & Run All (Commit)** so it runs
headless (it survives you closing the tab). The per-phase notebooks also run on
[Colab](https://colab.research.google.com) when its GPU is available. Start with
**`notebooks/00_setup.ipynb`** and work through them in order.

To check the code on a normal laptop (no GPU, no big download):

```bash
pip install -r requirements.txt
python tests/_build_fixture.py     # ~100 real images, a few MB
python tests/smoke_phase1.py       # runs every Phase-1 module on the fixture
```

## Data & credit

Dataset: [PM25Vision](https://huggingface.co/datasets/DeadCardassian/PM25Vision)
(CC-BY-4.0). Street imagery originates from **Mapillary**; air-quality labels from
the **World Air Quality Index (WAQI)** project. Please credit both; keep any public
demo non-commercial.
