# PROGRESS — living status log

> Kept current at the end of each phase. If you're an AI assistant picking this up in a
> later session, **read this first**, then `docs/` for details and the approved plan at
> `~/.claude/plans/this-is-the-output-steady-hearth.md`.

**Last updated (2026-09-25):** **The leakage story is done, verified, and documented.** The headline
experiment ran as one Kaggle *commit* (headless) job and produced the project's central result:

- Honest **station-grouped R² = 0.220** (MAE 54.2, Spearman 0.649, coverage 0.872).
- Leaky **random R² = 0.759** → **Δ_leak(R²) = 0.539**. The random split even exceeds the paper's 0.55;
  no station-disjoint split reaches 0.55 (shipped 0.378, geographic −0.079).
- Causal clincher: within the random test set, **contaminated R² = 0.762 vs clean R² = −0.86** — the
  inflation lives entirely in the leaked photos.

A **3-agent read-only correctness audit** (data/physics, model/loss/training, splits/calibration/
metrics/leakage) found **no result-affecting bug**: leakage prediction↔row ordering correct, physics
cache↔image alignment correct, splits station-disjoint by construction, conformal coverage math correct
(sim 0.9006), point head genuinely mean-seeking, every results-table row reconciles R²≈1−(RMSE/SD)².
Minor cleanups from the audit have been applied (see "Hardening" below).

**Documentation written this session:** `docs/RESULTS.md` (the full results ledger / story arc),
`docs/10_leakage.md` (beginner narrative of the leakage experiment), `docs/CODE_WALKTHROUGH.md`
(the essential code embedded + explained, teacher-facing).

**Method note (the honest pipeline):** 5-channel physics input → EfficientNet-B0 → monotone quantile
heads (calibrated intervals) **plus** a dedicated **MSE point head on a standardized target** (estimates
the conditional mean → good R²). Intervals via Conformalized Quantile Regression (~90% coverage); point
estimate de-standardized then isotonic-recalibrated on the calibration split. Primary evaluation split is
**station-grouped** (leakage-safe); `random` is the constructed leaky control.

**Hardening applied this session (low-risk, no rerun needed):** fixed stale docstrings/comments that
described the point head as "Huber-trained + smearing" (it is MSE + standardize + isotonic); removed
unused functions (`calibrate.smearing_factor`/`apply_point`, `train.evaluate_loss`/`collect_predictions`);
added defensive asserts (`splits.make_splits` guards NaN station/lat/lon; `physics.five_channel_cached`
guards cache bounds/size). All local smoke tests still pass.

**C2 (error ceiling) BUILT + smoke-tested (2026-09-25).** `src/aqi.py` now has a parameterized
breakpoint table (historical default + `PM25_BREAKPOINTS_2024`). `src/ceiling.py` upgraded:
level-reweighted within-day noise curve v(m)→p(m) (`level_reweighted_var_epsilon`), split-specific
Var(y), R²_max reported as an UPPER bound, empirical 5–95 percentile interval floor
(`within_day_deviations`/`empirical_interval_floor`), min_hours=18, region-stratified OpenAQ sampling
(`max_per_country`), station cluster-bootstrap CI (`bootstrap_r2max_ci`), and a one-call
`compute_ceiling` that saves everything to `outputs/error_ceiling.json` (so `09_leakage` draws the
ceiling line). `tests/smoke_ceiling.py` verifies it all incl. a hand-computed reweighting value
(=175.0); 2024 breakpoints selectable + monotonic. `notebooks/07_error_ceiling.ipynb` + `docs/08_error_ceiling.md`
updated and notebooks regenerated. **Next: user runs `07_error_ceiling` with a free OpenAQ key**
(explore.openaq.org, no GPU) → paste R²_max + CI + the 2012/2024 band. Then **C3** (abstention:
`src/abstain.py` + `notebooks/08_abstention.ipynb`) → Phase 10 Gradio demo.

## What this project is (30-second version)

Predict PM2.5 **AQI (index 1–530, NOT µg/m³)** from a single street photo, *honestly*:
calibrated low/median/high interval + refuse-to-answer + an unavoidable-error ceiling +
leakage-safe evaluation. Implements the paper in `../ML_Paper_First_Update (1).pdf`.
Dataset: `DeadCardassian/PM25Vision` (HF, 11,096 rows after dedup, 3,259 stations).

## How the pieces fit

- `src/` = the real logic (small, tested modules). `notebooks/` = per-phase notebooks that import
  `src/` and carry the beginner explanations. `docs/` = the same explanations, standalone, plus
  `RESULTS.md` (results ledger) and `CODE_WALKTHROUGH.md` (essential code). `configs/default.yaml` =
  all settings. `tests/` = a 100-image fixture + smoke tests so code is verified on a laptop first.
- **Workflow:** Claude writes + smoke-tests code locally (CPU, fixture). User pushes to GitHub, runs
  notebooks on Kaggle (GPU) as **Save & Run All (Commit)** jobs (headless — survives closing the tab),
  pastes output back. **Full dataset always** for real runs; the fixture is only Claude's local harness.

## Phase status

| Phase | What | Status |
|-------|------|--------|
| 0 | Setup (repo, deps, Colab/Kaggle) | ✅ built + run |
| 1 | Data load + clean + audit (§3.2–3.3) | ✅ built + run on full data |
| 2 | Leakage-safe splits (§3.4) | ✅ built + run (5 strategies; grouped/geo/shipped verified 0-straddle) |
| 3 | Physics features / 5-channel input (§3.5) | ✅ built + tested (cache verified aligned to image) |
| 4 | Model — EfficientNet-B0 + monotone quantiles (§3.6) | ✅ built + tested (monotone by construction) |
| 5 | Training — pinball + MSE point head, log-target quantiles (§3.7, §3.12) | ✅ built + tested |
| 6 | Conformal calibration + evaluation (§3.8, §3.11) | ✅ built + tested (coverage ≈ 0.90) |
| A | Leakage measurement (the "story") | ✅ **run + verified + documented** (Δ_leak=0.539) |
| 7 / C2 | Error ceiling via OpenAQ (§3.10) | ✅ **built + smoke-tested**; needs user's free OpenAQ key to run (CPU) |
| 8 / C3 | Abstention via ExDark/DTD/Indoor (§3.9) | ⏳ after C2 |
| 9 | Ablations (§3.11) | ⏳ optional |
| 10 | Frontend demo (Gradio) | ⏳ needs trained model |

## Key decisions

- Label is AQI, not µg/m³ → report in AQI points; `src/aqi.py` converts only for C2.
- Re-split the data ourselves. The shipped split is already station-disjoint, so the "leaky" random
  control is one we construct (Phase 2).
- Point head trained with **MSE on a standardized target** (mean-seeking → good R²); intervals from
  log-space monotone quantiles + conformal; isotonic recalibration on cal.
- 5-channel input = RGB + DCP transmission + inverted saturation; physics maps cached (indexed by `_row`).
- Every notebook self-clones/installs so `from src import ...` works standalone.
- A bigger backbone can be trained later via `model.backbone` (config change + retrain); its result gets
  a new row in `RESULTS.md` beside the B0 baseline.

## Known issues / open items

- C2 needs a free **OpenAQ API key** (explore.openaq.org) to fetch hourly data — user provides at run.
- C3 needs external image sets (ExDark, DTD, MIT-Indoor) — user downloads when we reach it.
- Demo hosting (temporary Colab/Kaggle link vs permanent HF Space) — decide at Phase 10.

## Immediate next step

Build **C2 (error ceiling)**: upgrade `src/ceiling.py` + `src/aqi.py` (see "Next" above), smoke-test on
synthetic hourly data, then hand off to the user to run with an OpenAQ key. Append the R²_max + sensitivity
band to `docs/RESULTS.md` when it arrives.

## Test commands (laptop, no GPU)
`python tests/_build_fixture.py` then `python tests/smoke_phase1.py` and `python tests/smoke_core.py`.
