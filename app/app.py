"""Gradio demo for Physics-Guided Visual Air-Quality Estimation (Phase 10).

Upload a street photo -> the app runs the full trained pipeline (5-channel physics input ->
EfficientNet-B0 -> calibrated low/median/high interval -> isotonic point -> abstention gate) and shows
the AQI estimate, the honest interval, the EPA category, an answer/abstain badge, and the two physics
maps so viewers can SEE the haze signal. The page also makes the paper's three honesty guarantees
visible: **C1** (the calibrated conformal interval), **C2** (the measured error ceiling), and **C3**
(inference-time abstention).

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

# ~Irreducible interval width (AQI): even a perfect model can't be tighter, because the label is a
# daily average with within-day swing. This is contribution C2's measured floor (docs/RESULTS.md §7).
IWIDTH_FLOOR = 35.0

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


def _c2_line(low, high):
    """The C2 reality-check: how the shown interval width compares to the physical floor (~35 AQI)."""
    w = max(0.0, high - low)
    if w <= IWIDTH_FLOOR * 1.2:
        note = "near the **physical floor** — about as tight as *any* model could honestly be."
    else:
        note = f"about **{w / IWIDTH_FLOOR:.1f}×** the physical floor (a genuinely hard photo)."
    return (f"**Interval width:** {w:.0f} AQI &nbsp;·&nbsp; irreducible floor **~{IWIDTH_FLOOR:.0f} AQI** "
            f"(daily-average label noise — contribution **C2**): {note}")


def run(image):
    blank = ("", "", "", "<i>Upload a photo to begin.</i>", None, None)
    if image is None:
        return blank
    if _BUNDLE is None:
        return ("", "", "",
                f"<b style='color:#b42318'>Model not loaded.</b> {_LOAD_ERROR or ''}", None, None)
    r = I.predict(image, _BUNDLE)
    aqi_txt = f"## AQI ≈ {r['aqi']:.0f}"
    interval = (f"**Calibrated 90% interval (C1):** {r['low']:.0f} – {r['high']:.0f} AQI &nbsp; "
                f"{_chip(r['epa_category'], r['epa_colour'])}")
    c2 = _c2_line(r["low"], r["high"])
    badge = _badge(r["answer"], r["reason"])
    if not r["answer"]:
        aqi_txt += "  \n*(abstained — estimate shown for reference only)*"
    return (aqi_txt, interval, c2, badge,
            _map_to_image(r["transmission_map"]), _map_to_image(r["inv_sat_map"]))


GUARANTEES_MD = """\
### The three honesty guarantees on this screen
- **C1 — calibrated interval.** The low–high range is a *conformal* 90% prediction interval: across many
  photos the true AQI lands inside it about **90%** of the time — a distribution-free guarantee, not a guess.
- **C2 — error ceiling.** We *measured* that even a **perfect** model tops out near **R² ≈ 0.98** and can't
  give intervals tighter than **~35 AQI**, because the label is a daily average. A wide-ish interval here is
  honesty about a hard task, not a bug.
- **C3 — abstention.** The badge **refuses** night / featureless photos the model shouldn't guess on
  (well-lit indoor scenes are a documented limitation of the station-invariant gate).
"""

WHY_RANGE_MD = """\
Air quality from a single street photo is genuinely hard, and the label we train on is a **daily average**,
which has irreducible within-day swing. We measured this ceiling on **32 stations across 6 countries**: the
best R² *any* model could reach is **≈0.98**, label noise is only **~2%** of the variance, and no honest
interval can be tighter than **~35 AQI**. Our honest model reaches R² **0.22 (single split) / 0.385
(cross-validated)** — far below the ceiling — so the remaining gap is the **difficulty of the visual task**,
not noisy labels. That is why we show a *range* and sometimes *refuse*, instead of a confident single number.
"""

RESULTS_MD = """\
| Setting | R² |
|---|---|
| Honest — station-grouped, single split | **0.22** |
| Honest — station-grouped, 5-fold CV | **0.385** |
| Leaky random split (same model) | 0.76 |
| Published benchmark (leaky) | 0.55 |
| Error ceiling (best any model could do) | **≈0.98** |

The honest-vs-leaky gap is the project's headline finding: a **random** split lets the model recognise
*places* it has already seen (same-station photos share one daily-average label), inflating the score. We
forbid that by splitting on **stations** and report the honest number.
"""


with gr.Blocks(title="Visual Air-Quality Estimation") as demo:
    gr.Markdown("# 🌫️ Air quality from a street photo — *honestly*\n"
                "Upload a daytime street photo. The model returns an **AQI estimate**, a **calibrated "
                "low–high range** (C1), a **reality-check against the error ceiling** (C2), and — crucially "
                "— it will **refuse to answer** on **night** or **featureless** photos it shouldn't guess on "
                "(C3). The two maps show the physics it uses: **transmission** (dark = more haze) and "
                "**inverted saturation** (bright = washed-out/hazy). "
                "*(Indoor scenes are a known limitation of the station-invariant gate — see the writeup.)*")
    if _LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **Demo mode — model bundle not loaded** from `{MODEL_DIR}` ({_LOAD_ERROR}). "
                    "The honesty explanations below still work; add `best_model.pth` + "
                    "`inference_bundle.json` to enable live predictions.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Street photo")
            btn = gr.Button("Estimate air quality", variant="primary")
        with gr.Column():
            aqi_md = gr.Markdown()
            interval_md = gr.Markdown()
            c2_md = gr.Markdown()
            badge_html = gr.HTML()
            with gr.Row():
                tmap = gr.Image(label="Transmission (haze)", height=200)
                smap = gr.Image(label="Inverted saturation", height=200)

    gr.Markdown(GUARANTEES_MD)
    with gr.Accordion("Why a range and not a single exact number? (contribution C2)", open=False):
        gr.Markdown(WHY_RANGE_MD)
    with gr.Accordion("Project results at a glance (honest vs leaky)", open=False):
        gr.Markdown(RESULTS_MD)
    gr.Markdown("*Dataset: PM25Vision (CC-BY-4.0). Imagery: Mapillary · Labels: WAQI. Keep any public "
                "demo non-commercial and credit both.*")

    outs = [aqi_md, interval_md, c2_md, badge_html, tmap, smap]
    btn.click(run, inputs=inp, outputs=outs)
    inp.change(run, inputs=inp, outputs=outs)

if __name__ == "__main__":
    demo.launch()
