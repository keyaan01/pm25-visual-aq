---
title: Visual Air Quality Estimation
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Physics-Guided Visual Air-Quality Estimation — live demo

Upload a daytime street photo and the model estimates the **US-EPA AQI**, an honest **low–high range**,
the **EPA category**, and an **answer / abstain** decision (it refuses on **night** or **featureless**
photos; indoor scenes are a documented limitation of the station-invariant gate). The two maps show the
physics it reads: **transmission** (dark = more haze) and **inverted saturation** (bright = washed-out/hazy).

> This block (the YAML at the top + this README) is what makes a Hugging Face **Space** work. To deploy,
> put `app.py`, the repo's `src/` folder, `requirements.txt`, this `README.md`, and a `model/` folder
> (`best_model.pth` + `inference_bundle.json`) at the Space
> root. Full click-by-click steps: [`docs/11_demo.md`](../docs/11_demo.md).

Dataset: [PM25Vision](https://huggingface.co/datasets/DeadCardassian/PM25Vision) (CC-BY-4.0); imagery
from **Mapillary**, labels from the **WAQI** project — please credit both; keep any public demo
**non-commercial**.
