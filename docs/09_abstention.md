# Phase 8 — C3: knowing when to refuse (inference-time abstention)

Companion to `notebooks/08_abstention.ipynb` and `src/abstain.py`. Maps to **paper §3.9** (contribution
C3). An honest system must sometimes say **"I can't answer this."**

## Two reasons to refuse — checked in order

A photo can be un-answerable for two very different reasons, so we use **two independent gates**, and we
check them in this order:

**Gate 1 — OOD / unusable (checked FIRST): "is this even a valid daytime street photo?"**
Night shots, indoor scenes, and blank textures are *out of distribution* — the model was never trained
on anything like them, so any number it produces is meaningless. Our gate uses **interpretable,
station-invariant checks** — **too dark** (night) or **too flat / featureless** (`handcrafted_ood_score`
= `max(dark, flat)` on brightness and detail-variance). These are the right primary signal because a
valid daytime street photo from *any* station is bright and detailed. They catch **night** and
**featureless** photos; a well-lit **indoor** scene is bright and detailed, so these checks do **not**
catch it — indoor is a documented near-OOD limitation (see the OOD-detection section below), the price we
pay for a gate that is robust across unseen stations.

> **An honest finding we document.** We *also* tried a shrinkage **Mahalanobis distance** in the
> backbone feature space (`fit_mahalanobis` / `mahalanobis_score`) and found it **over-refuses valid
> photos from unseen stations** — it confuses "new station" (benign) with "unusable photo" — because in
> ~1280-dim feature space, unseen-station photos are genuinely far from the calibration photos. So a raw
> feature-distance detector is the wrong tool for a station-generalisation task; we report its AUROC for
> comparison but use the station-invariant handcrafted gate. (We deliberately do **NOT** use interval
> width for OOD either — the conformal guarantee only holds in-distribution, so a width test is invalid
> off-distribution.)

**Gate 2 — uncertainty (checked second): "the photo is valid, but is the model too unsure?"**
Among valid photos, we refuse the ones with the widest **log-space** interval (`log_interval_width` =
`q95_log − q05_log`). Log space matters: because we train on `log(AQI)`, the raw-AQI width grows with
pollution level, so a raw-width rule would just refuse every high-AQI photo — the log width is the
level-fair uncertainty signal.

> **Honest finding.** On this hard task the model's **log-width is a weak predictor of its point error**
> (its risk–coverage AURC is close to random) — the interval width and the point-head error aren't
> strongly coupled. The **raw** AQI width does better, but only because it refuses high-AQI photos, which
> genuinely are the hard ones. We therefore report the **deployed** signal — **log width** — as the
> headline selective-prediction result (it is what the gate and the demo actually use), and show the raw
> width and a random baseline only as **labelled diagnostics**, so a lower-AURC diagnostic can never be
> mistaken for the deployed behaviour. **The OOD gate (catching night / featureless photos) is C3's main
> practical value**, not the uncertainty gate.

## Setting the thresholds honestly (calibration only)

Thresholds are chosen on **good images only** — never on labels, and never on the OOD sets:

- `τ_ood` fixes an explicit **5% false-refusal budget** on the handcrafted score (`ood_threshold`), and
  we set it and *report* it on **two disjoint sets of unseen-station good photos**: a `good_fit` slice to
  choose the threshold and a separate `good_eval` slice to measure the achieved refuse-rate and to serve
  as the in-distribution reference in the OOD-detection metrics. Fitting and reporting on the *same* set
  would make "≈5% refused" true by construction; splitting them makes it a real held-out check. We
  calibrate on unseen-station photos because a threshold set on the calibration stations does not transfer
  to new stations. No AQI labels are used — this is a deployment-safety budget, not accuracy tuning.
  `ood_threshold` uses an exact **top-k** rule (refuse the k = ⌊budget·n⌋ highest scores) rather than a
  raw percentile, because the handcrafted score is spiked at exactly 0 for most good photos and a plain
  percentile could collapse to 0 and refuse arbitrary mildly-dim photos.
- `τ_width` = the calibration log-width at the target coverage (`width_threshold`), so on good inputs we
  answer about 90% of the time.

Both the handcrafted score used to set `τ_ood` and the score computed at serve time in the demo are taken
at the **same image resolution** (224 px, the size `physics.compute_maps` produces), so the deployed
operating point matches the calibrated one — `detail_variance` is resolution-dependent, so a size mismatch
would shift the `flat` term's threshold.

## What we measure

**Selective prediction** (does refusing the low-confidence photos actually improve accuracy?):
- **risk–coverage curves** — accept the most-confident fraction, plot the error as you accept more
  (`risk_coverage_curve`);
- **AURC** (area under it), **excess-AURC** vs an oracle, and **AUGRC** (`aurc` / `excess_aurc` /
  `augrc`);
- **selective MAE / RMSE / category-accuracy / interval-coverage at 90/80/70% coverage**
  (`selective_metrics_at_coverage`); baselines: **random** (no skill) and **oracle** (perfect).

**OOD detection** (does the OOD gate actually catch unusable photos?), using three external
"should-refuse" sets pushed through the *identical* `physics.five_channel` pipeline:
- **AUROC, AUPR, FPR@95TPR** (`ood_metrics`) for the deployed handcrafted score, with the Mahalanobis
  score reported alongside for comparison;
- **near-OOD (MIT-Indoor)** vs **far-OOD (ExDark night, DTD textures)** — near-OOD is harder. We *expect*
  the handcrafted gate to score **near-chance on MIT-Indoor**: a lit indoor photo is bright and detailed,
  so `max(dark, flat)` does not fire. Indoor is a **documented limitation** of the station-invariant gate;
  the reported Mahalanobis catches indoor better but over-refuses valid unseen-station outdoor photos, so
  neither is free — we deploy the handcrafted gate and are explicit that it targets **night / featureless**,
  not indoor.
- a **refuse-rate table** over {ExDark, DTD, MIT-Indoor, PM25Vision-good}: high on the far-OOD sets, near
  the ~5% budget on the held-out `good_eval` photos, and (honestly) low on MIT-Indoor for the handcrafted
  gate.

> **A clean Mahalanobis comparison.** To show the Mahalanobis over-refusal is genuine *station shift* and
> not just in-sample optimism, we fit it on one calibration slice and set its threshold on a **held-out
> same-station** slice, then measure its refuse-rate on unseen-station test. The *excess* refusal over the
> same-station holdout is the part attributable to unseen stations — the confound (any held-out set scores
> higher than the fitted one in ~1280-dim space) is removed from the claim.

## How to run

`notebooks/08_abstention.ipynb` on Kaggle (uses the trained model, **no retrain**). Download the three
OOD sets — **ExDark** (night), **DTD** textures, **MIT-Indoor** (indoor) — then run it. It prints the
risk–coverage/AURC and OOD tables and saves the **inference bundle** the demo needs
(`docs/11_demo.md`). Paste the numbers back and they go into `docs/RESULTS.md`.

## The payoff

The same gate powers the demo's **answer / abstain** badge — so the honesty isn't just a table in the
paper, it's visible: upload a night photo and the system *tells you it won't guess*.
