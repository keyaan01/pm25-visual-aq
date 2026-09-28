# Phase 10 — the live demo (Gradio on a Hugging Face Space)

The demo lets anyone upload a street photo and see the honest prediction: an **AQI estimate**, a
**low–high range**, the **EPA category**, an **answer / abstain** badge, and the two **physics maps**.
It reuses the trained model — no retraining. Code: [`app/app.py`](../app/app.py) (UI) +
[`src/inference.py`](../src/inference.py) (the one-photo pipeline).

## What the demo needs (the "bundle")

A small folder produced by `notebooks/08_abstention.ipynb` (its last cell saves it):

```
model/
  best_model.pth            # the trained checkpoint
  inference_bundle.json     # Q, y_mean, y_std, isotonic knots, tau_ood, tau_width
  maha_mean.npy             # the OOD detector
  maha_precision.npy
```

Download that `model/` folder from the Kaggle run's **Output** (or rebuild it from any trained
checkpoint with `inference.save_bundle`).

## Try it locally first (optional, ~1 min)

```bash
pip install gradio
MODEL_DIR=path/to/model python app/app.py
```

Gradio prints a `http://127.0.0.1:7860` link. Upload a photo; a black/indoor photo should **abstain**.

## Deploy it permanently (free Hugging Face Space)

1. Make a free account at **https://huggingface.co**.
2. **New → Space** → give it a name → **SDK: Gradio** → **Hardware: CPU basic (free)** → Create.
3. The Space is a git repo. Put these at its **root** (drag-and-drop via *Files → Add file → Upload*,
   or `git push`):
   - `app.py`  ← copy of [`app/app.py`](../app/app.py)
   - `requirements.txt`  ← copy of [`app/requirements.txt`](../app/requirements.txt)
   - `README.md`  ← copy of [`app/README.md`](../app/README.md) (its top YAML block configures the Space)
   - `src/`  ← the whole [`src/`](../src) folder from this repo (the app imports the pipeline from it)
   - `model/`  ← the bundle folder above
4. The Space rebuilds automatically and goes live at
   `https://huggingface.co/spaces/<your-name>/<space-name>` — a permanent, shareable URL.

> **Tip:** `best_model.pth` is a few MB — fine for the free tier. If a file is >10 MB the HF UI asks you
> to confirm Git-LFS; accept it. Keep the demo **non-commercial** and credit **Mapillary** (imagery) and
> **WAQI** (labels).

## What viewers see

- **AQI ≈ N** and the **90% range** `low – high` (never a bare number — the honesty point).
- The **EPA category** chip in its official colour.
- **✅ ANSWER** or **🚫 ABSTAIN** with the reason (out-of-distribution vs too-uncertain).
- **Transmission** (dark = more haze) and **inverted saturation** (bright = hazy) maps — the physics.
