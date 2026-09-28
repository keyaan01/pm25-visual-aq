# Phase 8 — C3: knowing when to refuse (inference-time abstention)

Companion to `notebooks/08_abstention.ipynb` and `src/abstain.py`. Maps to **paper §3.9** (contribution
C3). An honest system must sometimes say **"I can't answer this."**

## Two reasons to refuse — checked in order

A photo can be un-answerable for two very different reasons, so we use **two independent gates**, and we
check them in this order:

**Gate 1 — OOD / unusable (checked FIRST): "is this even a valid daytime street photo?"**
Night shots, indoor scenes, and blank textures are *out of distribution* — the model was never trained
on anything like them, so any number it produces is meaningless. We score how far a photo is from the
training photos **in the network's own feature space**, using a shrinkage **Mahalanobis distance** on
the backbone features (`fit_mahalanobis` / `mahalanobis_score`). As an interpretable backup we also
compute plain **brightness**, **detail variance**, and **sky fraction** (`handcrafted_ood_score`).
**We deliberately do NOT use the interval width here** — the conformal guarantee only holds on
in-distribution data, so a width-based test is invalid off-distribution.

**Gate 2 — uncertainty (checked second): "the photo is valid, but is the model too unsure?"**
Among valid photos, we refuse the ones with the widest **log-space** interval (`log_interval_width` =
`q95_log − q05_log`). Log space matters: because we train on `log(AQI)`, the raw-AQI width grows with
pollution level, so a raw-width rule would just refuse every high-AQI photo — the log width is the
level-fair uncertainty signal.

## Setting the thresholds honestly (calibration only)

Both thresholds are chosen on the **calibration** split — never on the test set or the OOD sets:

- `τ_ood` = the 95th percentile of the calibration **good-image** OOD scores (`ood_threshold`). This
  fixes an explicit **5% false-refusal budget** using good images alone — so we never tune the OOD
  detector to the very OOD data we then evaluate on.
- `τ_width` = the calibration log-width at the target coverage (`width_threshold`), so on good inputs we
  answer about 90% of the time.

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
- **AUROC, AUPR, FPR@95TPR** (`ood_metrics`);
- **near-OOD (MIT-Indoor)** vs **far-OOD (ExDark night, DTD textures)** — near-OOD is harder;
- a **refuse-rate table** over {ExDark, DTD, MIT-Indoor, PM25Vision-good}: high on the unusable sets,
  ~5% on good photos (the false-refusal budget).

## How to run

`notebooks/08_abstention.ipynb` on Kaggle (uses the trained model, **no retrain**). Download the three
OOD sets — **ExDark** (night), **DTD** textures, **MIT-Indoor** (indoor) — then run it. It prints the
risk–coverage/AURC and OOD tables and saves the **inference bundle** the demo needs
(`docs/11_demo.md`). Paste the numbers back and they go into `docs/RESULTS.md`.

## The payoff

The same gate powers the demo's **answer / abstain** badge — so the honesty isn't just a table in the
paper, it's visible: upload a night photo and the system *tells you it won't guess*.
