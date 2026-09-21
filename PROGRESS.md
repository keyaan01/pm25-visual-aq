# PROGRESS — living status log

> Kept current at the end of each phase. If you're an AI assistant picking this up in a
> later session, **read this first**, then `docs/` for details and the approved plan at
> `~/.claude/plans/this-is-the-output-steady-hearth.md`.

**Last updated:** **Phase 5b (accuracy upgrades) built** after the first honest run gave R²=0.153.
Core + C2 + 5b all locally smoke-tested. Awaiting the user's RE-TRAIN on Colab with the upgrades,
then compare R²/MAE/coverage. Next to build: C3 (Phase 8), demo (Phase 10).

**First honest result (station_grouped, pre-5b):** R²=0.153, MAE=55.5, Spearman=0.715,
coverage=0.858, mean_width=159 — good ranking, but severe underprediction of extreme AQI (MAE 257
in the 300+ band). Fix = Phase 5b.

## What this project is (30-second version)

Predict PM2.5 **AQI (index 1–530, NOT µg/m³)** from a single street photo, *honestly*:
calibrated low/median/high interval + refuse-to-answer + an unavoidable-error ceiling +
leakage-safe evaluation. Implements the paper in `../ML_Paper_First_Update (1).pdf`.
Dataset: `DeadCardassian/PM25Vision` (HF, 11,219 rows, 3,261 stations).

## How the pieces fit

- `src/` = the real logic (small, tested modules). `notebooks/` = per-phase Colab
  notebooks that import `src/` and carry the beginner explanations. `docs/` = the same
  explanations, standalone. `configs/default.yaml` = all settings. `tests/` = a 100-image
  fixture + smoke tests so code is verified on a laptop before the full Colab run.
- **Workflow:** Claude writes + smoke-tests code locally (CPU, fixture). User pushes to
  GitHub, runs notebooks in Colab (GPU), pastes output back. **Full dataset always** for
  real runs; the fixture is only Claude's local harness.

## Phase status

| Phase | What | Status |
|-------|------|--------|
| 0 | Setup (repo, deps, Colab, Drive) | ✅ built + locally tested; awaiting user's first Colab run |
| 1 | Data load + clean + audit (§3.2–3.3) | ✅ built + locally tested |
| 2 | Leakage-safe splits (§3.4) | ✅ built + locally tested (4 strategies; grouped/geo verified 0-straddle) |
| 3 | Physics features / 5-channel input (§3.5) | ✅ built + tested (`five_channel`, `build_map_cache` uint8, cache==on-the-fly) |
| 4 | Model — EfficientNet-B0 + monotone quantiles (§3.6) | ✅ built + tested (5-ch stem, monotone by construction, pretrained load OK) |
| 5 | Training — pinball loss, log target (§3.7, §3.12) | ✅ built + tested (CPU mini-run: losses/dataset/train chain) |
| 6 | Conformal calibration + evaluation (§3.8, §3.11) | ✅ built + tested (conformal hits 0.90 on synthetic; full train→eval integration) — **core done** |
| 5b | Accuracy upgrades (point head + Huber + smearing; class-balanced sampling; +epochs) | ✅ built + tested; awaiting Colab re-train |
| 7 | C2 error ceiling via OpenAQ (§3.10) | ✅ built + math tested (needs user's free OpenAQ key to run) |
| 8 | C3 abstention via ExDark/DTD/Indoor (§3.9) | TODO (after core) |
| 9 | Ablations (§3.11) | TODO |
| 10 | Frontend demo (Gradio) | TODO (needs trained model) |

## Key decisions

- Label is AQI, not µg/m³ → report in AQI points; `src/aqi.py` converts only for C2.
- Re-split the data ourselves (pool HF train+test). The shipped split is already
  station-disjoint, so the "leaky" random control is one we construct (Phase 2).
- Train on `log(AQI)`; calibrate/report in AQI units.
- 5-channel input = RGB + DCP transmission + inverted saturation; physics maps cached.
- Every notebook self-clones/install/mounts Drive (fixes `No module named 'src'`).

## Known issues / open items

- **Not yet run on real data in Colab** — need the user's Phase-0/1 output to confirm
  real-dataset numbers (especially the falsification test's sign).
- Demo hosting (temporary Colab link vs permanent HF Space) — decide at Phase 10.
- OpenAQ API key (Phase 7) and Kaggle account for MIT Indoor (Phase 8) — needed later.

## Immediate next step

Core is built. On the user's side: run the full Colab pipeline (03 cache → 05 train on GPU →
06 evaluate) and paste numbers. On Claude's side: build **Phase 7 (C2 error ceiling via
OpenAQ)**, **Phase 8 (C3 abstention via ExDark/DTD/Indoor)**, then **Phase 10 (Gradio demo)**.
Write real Colab numbers into `docs/RESULTS.md` when they arrive.

## Test commands (laptop, no GPU)
`python tests/_build_fixture.py` then `python tests/smoke_phase1.py` and `python tests/smoke_core.py`.
