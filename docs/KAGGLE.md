# Running on Kaggle (free GPU)

Colab's free GPU is rate-limited. **Kaggle Notebooks** give a more generous free GPU
(~30 hours/week of T4). Because Kaggle notebooks are *isolated* (no shared Google Drive),
we use **one consolidated notebook** — `notebooks/kaggle_pipeline.ipynb` — that runs the whole
pipeline (load → physics cache → train → calibrate → evaluate) in a single session.

## One-time account setup

1. Create a free account at **[kaggle.com](https://www.kaggle.com)**.
2. **Phone-verify it**: Account → Settings → *Phone Verification*. This is required to unlock the
   **GPU** and **Internet** in notebooks. (Without it, neither works.)

## Run it

1. **kaggle.com → Create → New Notebook.**
2. In the **right sidebar**:
   - **Accelerator → GPU T4 ×2** (or any GPU).
   - **Internet → On.**
3. Get the notebook's code. Two easy ways:
   - **Simplest:** open `notebooks/kaggle_pipeline.ipynb` from your GitHub repo (File → Import
     Notebook → link, or just copy its cells), **or**
   - paste the bootstrap cell and let it clone the repo (it does `git clone` + `pip install`).
4. Set **`REPO_URL`** in the first cell to your repository's `.git` URL.
5. **Run All** (top menu). Expected timing on T4:
   - dataset download from Hugging Face: ~2–4 min,
   - physics cache: ~15 min (once),
   - training: ~30 min,
   - calibration + evaluation: ~1 min.
6. **Paste me the metrics** printed in step 6 (MAE / RMSE / R² / coverage / mean_width).

## Keeping your results

Kaggle wipes the session when you close it. To keep the trained model and outputs:

- Click **Save Version** (top right) → *Save & Run All (Commit)*. Everything under
  `/kaggle/working` (the checkpoint `best_model.pth`, `results.json`) is stored as the notebook's
  **Output**, which you can download later or attach to another notebook (e.g. the demo).

## Notes

- The physics cache (~1.1 GB) and checkpoints live in `/kaggle/working` (up to ~20 GB — plenty).
- If a session times out mid-run, just re-run — the cache step detects the existing cache and skips.
- The per-phase Colab notebooks (`00`–`07`) still work when Colab's GPU frees up; this Kaggle
  notebook is the all-in-one alternative.
