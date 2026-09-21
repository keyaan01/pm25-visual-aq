# PROGRESS — living status log

> Kept current at the end of each phase. If you're an AI assistant picking this up in a
> later session, **read this first**, then `docs/` for details and the approved plan at
> `~/.claude/plans/this-is-the-output-steady-hearth.md`.

**Last updated:** Phases 0–2 complete; notebook bootstrap fixed (self-cloning). Next:
Phase 3 (physics features / 5-channel caching).

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
| 3 | Physics features / 5-channel input (§3.5) | core physics fns exist in `src/physics.py`; caching + notebook TODO |
| 4 | Model — EfficientNet-B0 + monotone quantiles (§3.6) | TODO |
| 5 | Training — pinball loss, log target (§3.7, §3.12) | TODO |
| 6 | Conformal calibration + evaluation (§3.8, §3.11) | TODO — **completes core** |
| 7 | C2 error ceiling via OpenAQ (§3.10) | TODO (after core) |
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

Build Phase 3: `src/physics.py` already has transmission + inverted-saturation; add a
`five_channel_tensor(img, size)` + a disk cache of the two maps, `03_physics_features.ipynb`
(visualise maps), `docs/03_physics_features.md`. Then Phase 4 (model).
