# PROGRESS — living status log

> Kept current at the end of each phase. If you're an AI assistant picking this up in a
> later session, **read this first**, then `docs/` for details and the approved plan at
> `~/.claude/plans/this-is-the-output-steady-hearth.md`.

**Last updated:** **Stage A (max-accuracy) + Kaggle pivot built.** Colab free GPU hit its limit →
switched to Kaggle. Standardized point head (Huber on z-scored AQI — robust, no smearing blow-up),
early-stop on cal point-MAE, class-balanced sampling, drop_path. All locally smoke-tested incl. a
headless end-to-end run of `kaggle_pipeline.ipynb`. **Awaiting user's Kaggle run** (Stage A) to
compare R²/MAE vs the 0.153 baseline. Then Stage B (backbone/EMA/TTA), Stage C (ensemble), C3, demo.

**Key design note:** point head predicts a STANDARDIZED target (mean/std stored as model buffers,
travel in the checkpoint). Earlier raw-direct underpredicted (slow to reach scale) and log+smearing
was numerically UNSTABLE (smear exploded) — standardization fixed both. `point_to_aqi(out, y_mean,
y_std)` de-standardizes; no smearing anywhere now.

**Result history (station_grouped, honest):**
- pre-5b: R²=0.153, MAE=55.5, Spearman=0.715.
- Stage A (Kaggle, standardized point head + balanced + drop_path): R²=0.164, MAE=53.5,
  Spearman=0.665, coverage=0.881, mean_width=164. **Barely moved** — because the bottleneck is NOT
  the objective, it's the **300+ AQI band (MAE 287, n=173)** which likely hits a visual ceiling
  (a photo can't tell AQI 300 from 500). Model is actually decent for AQI<200 (MAE 22–47).
- Open question: gap to published 0.55 (RGB-only baseline). Hypotheses to TEST: (a) our physics
  channels may be hurting — added an **RGB-only ablation toggle** (`model.in_chans: 3`); (b) our
  test split may be harder. Also need C2 ceiling to know how much is irreducible.

**Experiment results so far (station_grouped honest, unless noted):**
- physics 5-ch, Stage A: R²=0.164, MAE=53.5.
- RGB-only (in_chans=3) + shipped 80/20 split (reproduce paper): **R²=0.132** — WORSE than physics,
  nowhere near their 0.55. So physics isn't hurting and split choice isn't it.
- Diagnosis (research-backed): (a) their 0.55 almost certainly used a RANDOM (station-leaky) split;
  (b) our model underpredicts the rare high-AQI tail (deep imbalanced regression).

**Accuracy fix implemented (this session):** `balance_power` 0.5→1.0 (strong tail sampling);
**isotonic recalibration** of the point estimate on cal (`src/recalibrate.py` — removes systematic
tail bias, no retrain); metrics now report BOTH `R2` (coeff. of determination, strict/field-standard)
and `r2_pearson` (squared correlation, what loose papers quote). Config reset to honest primary
(in_chans=5, station_grouped, recalibrate=true). Kaggle notebook reports raw vs recalibrated.

**Next:** user runs kaggle_pipeline (honest, improved) → paste recalibrated R2/r2_pearson/MAE. Then
optional `split.strategy: random` run to measure the leakage gap (explains the paper's 0.55). Then
C2 ceiling; Stage B (backbone/EMA/TTA); C3 abstention.

**PM25Vision paper (arXiv 2509.16519) baseline facts:** EfficientNet-B0 R²=0.550/MAE=36.6/RMSE=54.6
on an **80/20 split** (= the shipped station-disjoint split). Paper gives NO target/loss/preproc/
augmentation details — it's a plain RGB regressor, no physics, no calibration holdout. So our 0.16
differs by (a) physics channels and (b) our own re-split. **EXPERIMENT 1 (set up now):** reproduce
their baseline → `split.strategy: shipped` (test = their exact 2921 test rows; cal carved from their
train) + `model.in_chans: 3` (RGB-only). Added `shipped` split strategy (uses `orig_split` from the
DatasetDict) and RGB-only toggle. If this reproduces ~0.55, our pipeline is correct and 0.16 is just
the honest harder split; if not, pipeline issue. Both toggles revert to 5-ch/station_grouped after.

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
| 5b | Accuracy upgrades (point head; class-balanced sampling; +epochs) | ✅ built + tested |
| 5c | Stage A max-accuracy (standardized point head, cal-MAE early stop, drop_path) + Kaggle notebook | ✅ built + tested; awaiting Kaggle run |
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
