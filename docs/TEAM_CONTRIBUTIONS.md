# Team contributions — the 4-person work split

*Four balanced, coherent workstreams (roughly equal in code + concept). Each member owns a slice they can
present and defend standalone; together they cover the whole project. Files are grouped by owner so there
is no overlap in "who explains what."*

---

## Member 1 — Keyaan — Data, leakage & integration (project lead)

**Owns:** `src/data.py`, `src/audit.py`, `src/splits.py`, `src/leakage.py`,
`notebooks/09_leakage.ipynb`, `notebooks/10_crossval.ipynb` (co-owned with Lamiya), `docs/RESULTS.md`,
`docs/10_leakage.md`, overall repo integration + the results ledger.

**What this covers:**
- Loading and cleaning PM25Vision: dropping dead columns, removing the 123 duplicate image IDs,
  deterministic shuffle, and the `_row` index that keeps every photo aligned to its label throughout.
- The data audit: perceptual-hash near-duplicate grouping and the **physics falsification test**
  (transmission↔AQI correlation ≈ −0.14 confirms correct pairing).
- The **leakage-safe split family** (station-grouped, random, geographic, temporal, shipped) and why the
  calibration set is held separate.
- The **headline experiment**: the leakage gradient (same model, only the split changes) and the
  **causal contaminated-vs-clean** proof — the project's central finding.
- Integration: keeping `docs/RESULTS.md` the single source of truth for every number.

**Be ready to explain:** what data leakage is and why *this* dataset leaks; why station-grouped is the
honest protocol; the full gradient table and Δ_leak (lead with the MAE gap); why contaminated ≫ clean
proves leakage; why the −0.86 is variance-amplified.

**Likely professor questions:** *How did you prove leakage rather than assume it? · Couldn't the random
test set just be easier? · Do you accuse the benchmark authors of cheating?* (Answers in
`MASTER_EXPLAINER` §3 and §9.2.)

---

## Member 2 — Anup — Physics features, model architecture & training

**Owns:** `src/physics.py`, `src/model.py`, `src/losses.py`, `src/dataset.py`, `src/train.py`,
`notebooks/03_physics_features.ipynb`, `notebooks/04_model.ipynb`, `notebooks/05_train.ipynb`,
`docs/03_physics_features.md`, `docs/04_model.md`, `docs/05_training.md`.

**What this covers:**
- The **physics features**: Dark Channel Prior → transmission map, inverted-saturation map, the
  5-channel input, and the disk cache.
- The **model**: EfficientNet-B0 with the 3→5 channel stem widening; the three **monotone quantile heads**
  (cumulative softplus, non-crossing by construction); the **mean-seeking MSE point head** (why the mean,
  not the median, for R²) with standardisation buffers.
- The **losses**: pinball (asymmetric quantile) loss + MSE point loss.
- **Training**: log-target quantiles, AdamW + cosine warm-up, early stopping on calibration point-MAE,
  class-balanced sampling, drop-path, and the optional accuracy levers (EMA / TTA / seed ensemble).

**Be ready to explain:** how the Dark Channel Prior measures haze and why sky needs the inverted-saturation
map; why the quantiles can't cross; the mean-vs-median point about R²; how the pinball loss's asymmetry
teaches a percentile (τ=0.95 penalises under-prediction 19×).

**Likely professor questions:** *Why physics channels instead of raw pixels? · Why a separate point head?
· Why train on log(AQI)? · How do you guarantee the interval bounds don't cross?* (Answers in
`MASTER_EXPLAINER` §4.)

---

## Member 3 — Lamiya — Uncertainty, calibration & evaluation rigor

**Owns:** `src/calibrate.py`, `src/recalibrate.py`, `src/metrics.py`, `src/crossval.py`,
`notebooks/06_calibrate_evaluate.ipynb`, `notebooks/10_crossval.ipynb` (co-owned with Keyaan),
`docs/06_calibration.md`, `docs/07_evaluation.md`, `docs/11_crossval.md`, the correctness-audit
methodology.

**What this covers:**
- **Conformalized Quantile Regression (CQR)**: the nonconformity score, the finite-sample quantile Q, the
  coverage guarantee, and why it holds for any model under exchangeability.
- **Isotonic recalibration**: a monotone prediction→truth map fit on calibration only, to remove the
  systematic tail bias that depresses R².
- **Metrics**: MAE, RMSE, the strict R² *and* squared-Pearson, Spearman, coverage/width, per-band error,
  EPA-category accuracy — and the R² ≈ 1 − (RMSE/SD)² self-check.
- **Leakage-safe cross-validation** (GroupKFold by station): the 0.385 ± 0.19 result, the coverage
  validation (0.895), the split-dependence finding, and the correlated-fold caveat + Nadeau–Bengio.
- The **two 3-agent correctness audits** and what they verified.

**Be ready to explain:** why conformal gives a *guarantee* and what exchangeability means; why coverage
0.87 on one split but 0.895 in CV is expected; the difference between R² and squared-Pearson; why plain
k-fold would leak and GroupKFold doesn't; why the CV CI is mildly optimistic.

**Likely professor questions:** *Why does the conformal guarantee hold? · Coverage is 0.87, is it broken?
· How do you know the numbers aren't a bug? · Why report a CI, and is it rigorous?* (Answers in
`MASTER_EXPLAINER` §5–§6.)

---

## Member 4 — Asif — Contributions C2/C3 & the live demo

**Owns:** `src/ceiling.py`, `src/aqi.py`, `src/abstain.py`, `src/inference.py`, `app/app.py`,
`src/train.py::collect_features` (the feature extractor for the OOD gate),
`notebooks/07_error_ceiling.ipynb`, `notebooks/08_abstention.ipynb`, `docs/08_error_ceiling.md`,
`docs/09_abstention.md`, `docs/11_demo.md`.

**What this covers:**
- **C2 — the error ceiling**: R²_max = 1 − Var(ε)/Var(y); OpenAQ hourly data; InstantCast conversion;
  level-reweighting v(m)→p(m); split-specific Var(y); the measured **≈0.98** result and its honest
  caveats (optimistic bias, high-AQI under-coverage, split-specificity).
- **C3 — abstention**: the two gates (OOD via Mahalanobis + handcrafted checks, then log-width
  uncertainty), calibration-only thresholds with a 5% false-refusal budget, and the evaluation
  (risk–coverage/AURC + OOD AUROC/FPR95 with ExDark/DTD/MIT-Indoor, near vs far OOD).
- **The demo**: the one-photo inference pipeline and the Gradio app deployed as a permanent Hugging Face
  Space (AQI + interval + EPA category + answer/abstain badge + physics maps).

**Be ready to explain:** why daily-average labels create a ceiling and why it's an *upper bound* (~0.98);
what the ceiling means (task-limited, not label-limited); why OOD is checked before width and why width is
not used for OOD; how the demo turns the pipeline into a live "answer/abstain" experience.

**Likely professor questions:** *What is the error ceiling and is 0.98 trustworthy? · How does the model
decide to refuse? · Why not use interval width for out-of-distribution detection?* (Answers in
`MASTER_EXPLAINER` §7–§8.)

---

## Shared / joint

Everyone should know the **five-minute story** and the **top-level thesis** (honest AQI-from-photo + a
measured, causally-proven leakage effect), so any member can field a cross-cutting question. The
`MASTER_EXPLAINER.pdf` is the common study document; each member owns their section of it.
