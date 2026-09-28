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
  inference_bundle.json     # Q, y_mean, y_std, isotonic knots, tau_ood, tau_width, handcrafted_tau
```

The demo's OOD gate uses the **station-invariant handcrafted checks** (too dark / too flat), so the
bundle is just these two files — no Mahalanobis arrays needed. Download that `model/` folder from the
Kaggle run's **Output** (`demo_bundle/`), or rebuild it from any trained checkpoint with
`inference.save_bundle(..., maha=None, handcrafted_tau=...)`.

## Try it locally first (optional, ~1 min)

```bash
pip install gradio
MODEL_DIR=path/to/model python app/app.py
```

Gradio prints a `http://127.0.0.1:7860` link. Upload a photo; a black (night) or blank/featureless photo
should **abstain**. (A well-lit indoor photo is *not* caught by the station-invariant gate — a documented
limitation; see [`docs/09_abstention.md`](09_abstention.md).)

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

The page is built to make the **three honesty guarantees** visible to a viewer (or a professor):

- **AQI ≈ N** and the **calibrated 90% interval** `low – high` — never a bare number (**C1**).
- The **EPA category** chip in its official colour.
- A **C2 reality-check line**: the interval's width vs the **~35 AQI physical floor** (daily-average label
  noise), so a wide interval reads as honesty about a hard task, not a bug.
- **✅ ANSWER** or **🚫 ABSTAIN** with the reason — too dark (night) / too flat (featureless) vs
  too-uncertain (**C3**).
- **Transmission** (dark = more haze) and **inverted saturation** (bright = hazy) maps — the physics.
- Two collapsible panels: **"Why a range, not a single number?"** (the C2 ceiling story) and **"Project
  results at a glance"** (honest 0.22/0.385 vs leaky 0.55/0.76 vs the 0.98 ceiling).

The static panels render even before the model bundle is added, so the page is presentable in "demo mode."
