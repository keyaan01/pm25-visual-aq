# Phase 2 — Leakage-safe splits

Companion to `notebooks/02_splits.ipynb`. Maps to **paper §3.4**.

## The problem

To measure whether the model genuinely reads haze, we split the photos into three
groups:

- **train** (65%) — what the model learns from.
- **calibration** (15%) — a held-out slice used *only* to size the conformal
  correction in Phase 6 (never for training).
- **test** (20%) — untouched until the final evaluation.

*How* we split matters enormously. If two near-identical photos of the same street end
up one in train and one in test, the model can score well by recognising the scene
rather than by estimating pollution. That inflation is **data leakage**.

## The four strategies

| Strategy | How it assigns photos | What it's for |
|----------|----------------------|---------------|
| **random** | each photo independently | the **leaky control** — measures how much leakage inflates scores |
| **station_grouped** | all photos from a station → one split | **primary** — tests on genuinely unseen places |
| **geographic** | whole lat/lon regions (10° cells) → one split | tougher test of spatial generalisation |
| **temporal** | earliest years → train, latest → test | tests generalisation to the future |

**Why we build our own `random` control.** The proposal assumed the *shipped* split
was naively random and therefore leaky. Our audit showed the shipped split is already
station-disjoint, so it isn't leaky. To still *measure* leakage inflation honestly, we
construct the random control ourselves (pool everything, reshuffle), then compare it
against the grouped split. The gap between them is the finding.

## How the code guarantees no leakage

`make_splits` uses a **deficit-greedy group assignment** (`_group_split`): it shuffles
the groups (stations, or geographic cells), then sends each group to whichever split is
currently furthest below its target size. This keeps the 65/15/20 proportions close
*while guaranteeing a group is never split across two partitions*.

`split_report` then verifies this with **`stations_straddling_splits`** — the number of
stations that appear in more than one split. On the fixture:

| strategy | straddling stations |
|----------|--------------------|
| random | 6 &nbsp;← leakage (expected, it's the control) |
| station_grouped | **0** ← leakage-safe |
| geographic | **0** ← leakage-safe |
| temporal | 1 &nbsp;← a station spanning the time cutoff (temporal isn't station-safe by design) |

## Honest caveats

- **Geographic proportions can drift** from 65/15/20 because whole regions are
  indivisible — a big region either is or isn't in test. That's the price of a
  region-level hold-out, and we report the actual proportions.
- **Temporal** deliberately allows a station to appear on both sides of the time cutoff
  (same place, different years); it tests *time* generalisation, not *place*.

## What this sets up

Later phases train and evaluate under **station_grouped** (primary) and report
**random** alongside, so the leakage gap is measured, not assumed.

**Next:** Phase 3 (`03_physics_features`) — turning each photo into the 5-channel input.
