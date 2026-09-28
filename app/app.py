"""Gradio demo for Physics-Guided Visual Air-Quality Estimation (Phase 10).

Upload a street photo -> the app runs the full trained pipeline (5-channel physics input ->
EfficientNet-B0 -> calibrated low/median/high interval -> isotonic point -> abstention gate) and shows
the AQI estimate, the honest interval, the EPA category, an answer/abstain badge, and the two physics
maps so viewers can SEE the haze signal.

DEPLOY (Hugging Face Space, Gradio SDK): put this file, the repo's `src/` folder, `requirements.txt`,
a `README.md` with the Space header, and a `model/` folder (best_model.pth + inference_bundle.json) at
the Space root. See docs/11_demo.md for click-by-click steps.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Make `from src import ...` work whether src/ sits beside app.py (the Space) or one level up (the repo).
for cand in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(cand, "src")):
        sys.path.insert(0, cand)
        break

import gradio as gr  # noqa: E402
from src import inference as I  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR") or next(
    (d for d in (os.path.join(HERE, "model"), os.path.join(os.path.dirname(HERE), "outputs"),
                 os.path.join(HERE, "outputs")) if os.path.exists(os.path.join(d, "best_model.pth"))),
    os.path.join(HERE, "model"))

_BUNDLE = None
_LOAD_ERROR = None
try:
    _BUNDLE = I.load_bundle(MODEL_DIR, device="cpu")
except Exception as e:                                    # keep the Space up even if the model is missing
    _LOAD_ERROR = f"{type(e).__name__}: {e}"


def _map_to_image(m):
    """A (H,W) float map in [0,1] -> a displayable 8-bit grayscale PIL image."""
    a = np.clip(np.asarray(m, dtype=float) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _badge(ok, reason):
    colour = "#1a7f37" if ok else "#b42318"
    label = "✅ ANSWER" if ok else "🚫 ABSTAIN"
    return (f"<div style='padding:10px 14px;border-radius:8px;background:{colour};color:#fff;"
            f"font-weight:700;display:inline-block'>{label}</div>"
            f"<div style='margin-top:6px;color:#444'>{reason}</div>")


def _chip(cat, colour):
    text = "#000" if cat in ("Good", "Moderate") else "#fff"
    return (f"<span style='padding:4px 12px;border-radius:14px;background:{colour};color:{text};"
            f"font-weight:600'>{cat}</span>")


def run(image):
    if image is None:
        return "", "", "<i>Upload a photo to begin.</i>", None, None
    if _BUNDLE is None:
        return "", "", f"<b style='color:#b42318'>Model not loaded.</b> {_LOAD_ERROR or ''}", None, None
    r = I.predict(image, _BUNDLE)
    aqi_txt = f"## AQI ≈ {r['aqi']:.0f}"
    interval = (f"**Honest 90% range:** {r['low']:.0f} – {r['high']:.0f} AQI &nbsp; "
                f"{_chip(r['epa_category'], r['epa_colour'])}")
    badge = _badge(r["answer"], r["reason"])
    if not r["answer"]:
        aqi_txt += "  \n*(abstained — estimate shown for reference only)*"
    return (aqi_txt, interval, badge,
            _map_to_image(r["transmission_map"]), _map_to_image(r["inv_sat_map"]))


with gr.Blocks(title="Visual Air-Quality Estimation") as demo:
    gr.Markdown("# 🌫️ Air quality from a street photo — *honestly*\n"
                "Upload a daytime street photo. The model returns an **AQI estimate**, an honest "
                "**low–high range**, the **EPA category**, and — crucially — it will **refuse to "
                "answer** on **night** or **featureless** photos it shouldn't guess on. "
                "The two maps show the physics it uses: **transmission** (dark = more haze) and "
                "**inverted saturation** (bright = washed-out/hazy). "
                "*(Indoor scenes are a known limitation of the station-invariant gate — see the writeup.)*")
    if _LOAD_ERROR:
        gr.Markdown(f"> ⚠️ Model bundle not found in `{MODEL_DIR}` ({_LOAD_ERROR}). "
                    "Add `best_model.pth` + `inference_bundle.json` there.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Street photo")
            btn = gr.Button("Estimate air quality", variant="primary")
        with gr.Column():
            aqi_md = gr.Markdown()
            interval_md = gr.Markdown()
            badge_html = gr.HTML()
            with gr.Row():
                tmap = gr.Image(label="Transmission (haze)", height=200)
                smap = gr.Image(label="Inverted saturation", height=200)
    btn.click(run, inputs=inp, outputs=[aqi_md, interval_md, badge_html, tmap, smap])
    inp.change(run, inputs=inp, outputs=[aqi_md, interval_md, badge_html, tmap, smap])

if __name__ == "__main__":
    demo.launch()
