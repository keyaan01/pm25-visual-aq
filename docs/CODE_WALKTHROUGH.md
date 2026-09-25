# Essential code walkthrough

> A self-contained tour of the code that produced this project's two headline results, written so it
> reads on its own. The key functions are **embedded here as text** (a frozen snapshot), so this
> document stays accurate even after the live `src/` files evolve (e.g. when a larger model is trained
> later). Full modules live in `src/`; the numbers they produce are in [`RESULTS.md`](RESULTS.md).
>
> - **Part 1 — the honest model** that produced **R² = 0.220** (station-grouped, leakage-safe).
> - **Part 2 — the leakage test** that produced **Δ_leak(R²) = 0.539** and the contaminated-vs-clean proof.
>
> *Snapshot taken 2026-09-25, after the correctness audit. Label = US-EPA AQI index (not µg/m³).*

---

# Part 1 — the honest model (R² = 0.220)

The pipeline turns one photo into a **calibrated air-quality estimate**: a mean prediction (for
accuracy) plus a 90% low–high interval (for honesty). Six pieces, in the order the data flows.

## 1.1 Physics input — turn a photo into 5 channels  · `src/physics.py`

We hand the network two physics-derived maps alongside RGB, so it doesn't have to rediscover the optics
of haze from scratch. **Transmission** (Dark Channel Prior, He et al. 2011) is low where haze is dense;
**inverted saturation** covers the sky, where the dark-channel trick fails.

```python
def transmission_map(img, patch=15, omega=0.95, top_frac=0.001, t_min=0.05):
    """Estimate transmission t(x) in [t_min, 1]: LOW where haze is dense, HIGH in clear air."""
    rgb = to_float_rgb(img)
    dark = dark_channel(rgb, patch)                    # darkest pixel in any channel, locally
    A = atmospheric_light(rgb, dark, top_frac)         # the pale "haze colour"
    normalized = rgb / A[None, None, :]
    t = 1.0 - omega * dark_channel(normalized, patch)  # more haze -> lower transmission
    return np.clip(t, t_min, 1.0).astype(np.float32)

def inverted_saturation(img):
    """HIGH where colour is washed out (hazy / sky) — covers the dark channel's blind spot."""
    rgb = to_float_rgb(img)
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    return (1.0 - hsv[:, :, 1] / 255.0).astype(np.float32)
```

The three sources are stacked into the model's input tensor. RGB gets ImageNet normalisation (the
pretrained backbone expects it); the two physics maps get their own standardisation so they reach the
network at a comparable scale instead of being drowned out:

```python
def assemble_five_channel(rgb01, t01, s01, imagenet_norm=True):
    """Stack into (5, H, W): normalised RGB + standardised transmission + inverted-saturation."""
    phys = np.stack([t01, s01], axis=2)                    # (H, W, 2) in [0,1]
    if imagenet_norm:
        rgb = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
        phys = (phys - PHYS_MEAN) / PHYS_STD              # bring physics maps to ~unit scale
    else:
        rgb = rgb01
    hwc = np.concatenate([rgb, phys], axis=2)             # order: R,G,B,transmission,inv-sat
    return np.transpose(hwc, (2, 0, 1)).astype(np.float32)
```

> These maps are **cached to disk once** (`build_map_cache`) and read back per image by `_row` index
> (`five_channel_cached`) — recomputing the Dark Channel Prior every epoch would be far too slow. The
> audit verified the cache row always matches the correct source image.

## 1.2 The model — one backbone, two kinds of head  · `src/model.py`

The **interval heads** must never cross (a 95th percentile below the 50th would be nonsense). We
guarantee that *by construction* (paper Eq. 1): predict the lowest quantile freely, then add
**strictly-positive** softplus increments. Because every step is positive, the outputs can only rise.

```python
class MonotoneQuantiles(nn.Module):
    """Map features to n strictly-increasing quantiles (cumulative softplus)."""
    def forward(self, feats):
        z = self.fc(feats)                        # (B, n)
        base = z[:, :1]                           # lowest quantile, unconstrained
        increments = F.softplus(z[:, 1:])         # (B, n-1), all > 0
        steps = torch.cumsum(increments, dim=1)   # running total
        return torch.cat([base, base + steps], dim=1)   # ascending, guaranteed
```

The **point head** exists because the *median* quantile minimises absolute error, which on a
right-skewed target sits below the mean and drags R² down. So we add a dedicated head that estimates the
**mean** (see the loss in §1.3). It predicts a *standardised* target; the train-set mean/std travel with
the checkpoint as buffers so we can convert back to AQI at inference.

```python
class PM25QuantileNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", in_chans=5, ..., point_head=True, drop_path_rate=0.0):
        self.backbone = timm.create_model(backbone, pretrained=..., in_chans=in_chans,
                                          num_classes=0, drop_path_rate=drop_path_rate)
        self.head = MonotoneQuantiles(self.backbone.num_features, len(quantiles))
        self.point_head = nn.Linear(self.backbone.num_features, 1) if point_head else None
        self.register_buffer("y_mean", torch.tensor(0.0))   # train-set AQI mean (de-standardise)
        self.register_buffer("y_std",  torch.tensor(1.0))

    def forward(self, x):
        feats = self.backbone(x)
        out = {"quantiles": self.head(feats)}
        if self.point_head is not None:
            out["point"] = self.point_head(feats).squeeze(-1)   # (B,) standardised
        return out
```

## 1.3 The loss — mean for accuracy, quantiles for the interval  · `src/losses.py`

The quantile heads train with **pinball loss** (asymmetric: for τ=0.95, being too low is punished 19×
more than being too high). The point head trains with **MSE on the standardised target**, which is what
makes it estimate the conditional **mean** — the correct objective for R². (Using a robust/median loss
here would *depress* R² on this skewed target — so MSE is deliberate.)

```python
def combined_loss(out, y_raw, quantiles, point_weight=1.0, huber_delta=1.0,
                  y_mean=0.0, y_std=1.0, point_loss="mse"):
    y_log = torch.log(y_raw.clamp_min(1e-6))
    loss = pinball_loss(out["quantiles"], y_log, quantiles)     # intervals: log space
    if "point" in out and point_weight > 0:
        pt_target = (y_raw - y_mean) / y_std                    # standardised raw AQI
        if point_loss == "huber":
            point_term = F.huber_loss(out["point"], pt_target, delta=huber_delta)
        else:
            point_term = F.mse_loss(out["point"], pt_target)    # MEAN-seeking (good R²)
        loss = loss + point_weight * point_term
    return loss
```

## 1.4 The data & the leakage-safe split  · `src/dataset.py`, `src/splits.py`

Two choices keep the evaluation honest. First, **class-balanced oversampling is applied to the *train*
loader only** — the rare extreme-AQI photos are shown more often during training, but calibration and
test keep the natural distribution (otherwise the conformal guarantee and the reported metrics would be
distorted):

```python
train_sampler = None
if balanced:
    w = balanced_weights(train_df[target_col].to_numpy(), power=power)   # ∝ 1/band_frequency
    train_sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
# cal / test loaders use shuffle=False and NO sampler -> natural order & distribution
```

Second, the **station-grouped split** sends every photo of a station to exactly one side, so no station
straddles the train/test boundary — the model is tested on genuinely unseen places:

```python
def _group_split(groups, fractions, seed):
    """Assign whole groups (stations) to splits — a group never spans two splits."""
    sizes = groups.value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    targets = {s: f * sizes.sum() for s, f in zip(SPLIT_NAMES, fractions)}
    current = {s: 0 for s in SPLIT_NAMES}; assign = {}
    for g in order:                                   # send each station to the most "owed" split
        s = max(SPLIT_NAMES, key=lambda k: targets[k] - current[k])
        assign[g] = s; current[s] += int(sizes[g])
    return groups.map(assign).to_numpy()
```

## 1.5 Training loop  · `src/train.py`

AdamW + cosine schedule, mixed precision, fixed seeds. Two honesty-relevant details: the point head's
standardisation stats are computed from **train only**, and **early stopping tracks calibration
point-MAE** (the accuracy metric we care about), saving the *best* checkpoint — including the stats
buffers.

```python
y_mean, y_std = target_stats(loaders["train"])          # TRAIN only — no leakage
net.set_target_stats(y_mean, y_std)
...
cal_mae = point_mae_from_outputs(collect_outputs(net, loaders["cal"], device), y_mean, y_std)
if cal_mae < best_mae - 1e-4:                            # keep the best-on-calibration model
    best_mae, best_epoch, bad = cal_mae, epoch, 0
    torch.save({"model": net.state_dict(), ..., "y_mean": y_mean, "y_std": y_std}, ckpt_path)
```

`collect_outputs` runs the model over a loader (with `shuffle=False`, so outputs stay in dataframe
order — this is what lets the leakage analysis in Part 2 line predictions up with the right photo), and
`point_to_aqi` converts the standardised head output back to AQI:

```python
def point_to_aqi(point_out, y_mean, y_std):
    """De-standardise the point head's z-score back to AQI (clip at 0)."""
    return np.maximum(np.asarray(point_out) * y_std + y_mean, 0.0)
```

## 1.6 Calibrated intervals + bias removal  · `src/calibrate.py`, `src/recalibrate.py`

**Conformalized Quantile Regression** turns the raw quantiles into intervals with a *guaranteed* ~90%
coverage, using only the calibration set: measure how far each truth fell outside its predicted band,
take the 90th percentile of those misses, and widen every interval by that one number `Q`.

```python
def conformal_Q(preds_cal, y_cal, coverage=0.90):
    lo, hi = preds_cal[:, 0], preds_cal[:, 2]
    E = np.maximum(lo - y_cal, y_cal - hi)                    # nonconformity: how far outside
    level = min(1.0, np.ceil((len(E) + 1) * coverage) / len(E))
    return float(np.quantile(E, level, method="higher"))

def apply_conformal(preds, Q):
    out = preds.astype(float).copy()
    out[:, 0] = np.maximum(out[:, 0] - Q, 0.0)               # widen low end (AQI can't be < 0)
    out[:, -1] = out[:, -1] + Q                              # widen high end
    return out
```

Finally, **isotonic recalibration** fits a monotone pred→truth map on the *calibration* set and applies
it to test — removing systematic bias (the model's tendency to under-predict extremes) while preserving
ranking. Fit on calibration, never on test:

```python
def fit_isotonic(cal_point, cal_true):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    iso.fit(cal_point, cal_true)      # "when the model says 260, truth averages 340" -> map it
    return iso
```

**Result of Part 1 (station-grouped, honest): R² = 0.220, MAE = 54.2, Spearman = 0.649, coverage =
0.872.** See [`RESULTS.md`](RESULTS.md) §2.

---

# Part 2 — the leakage test (Δ_leak = 0.539)

This is the project's central finding: measure how much a *leaky* split inflates the score, and prove
the inflation is caused by leakage.

## 2.1 One split, one trained-and-evaluated model  · `src/leakage.py`

`train_and_evaluate` runs the **entire** Part-1 pipeline for a *single* split strategy and returns the
metrics plus per-test-row predictions. Everything except `strategy` is held constant (same config, seed,
cache), so comparing strategies is apples-to-apples.

```python
def train_and_evaluate(ds, df, cache, cfg, strategy, device, out_root, seed=None, ...):
    seed = seed if seed is not None else cfg["seed"]
    T.set_seed(seed)
    sp = S.make_splits(df, strategy=strategy, seed=seed, ...)   # ONLY this changes between runs
    loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=num_workers)
    net = M.build_model(cfg).to(device)
    hist = T.train_model(net, loaders, cfg, device=device, out_dir=os.path.join(out_root, strategy), ...)
    T.load_checkpoint(os.path.join(out_root, strategy, "best_model.pth"), net, map_location=device)

    cal  = T.collect_outputs(net, loaders["cal"],  device)
    test = T.collect_outputs(net, loaders["test"], device)
    ymean, ystd = float(net.y_mean), float(net.y_std)
    point = T.point_to_aqi(test["point_out"], ymean, ystd)
    if cfg["train"].get("recalibrate", True):                   # isotonic, fit on cal
        iso = R.fit_isotonic(T.point_to_aqi(cal["point_out"], ymean, ystd), cal["y_raw"])
        point = R.apply_isotonic(iso, point)
    Q = C.conformal_Q(np.exp(cal["q_log"]), cal["y_raw"], cfg["calibration"]["coverage"])
    intervals = C.apply_conformal(np.exp(test["q_log"]), Q)
    y = test["y_raw"]

    rep = Mx.report(point, intervals, y, cfg["calibration"]["coverage"])
    rep["SD_y_test"] = float(np.std(y)); rep["straddling_stations"] = S.split_report(sp, ...)[...]
    # per-test-row frame — loader used shuffle=False, so row i of test_df == prediction i
    test_df = sp[sp["split"] == "test"].copy().reset_index(drop=True)
    test_df["pred"] = point; test_df["y"] = y
    return {"strategy": strategy, "report": rep, "sp": sp, "test_df": test_df}
```

The notebook (`09_leakage.ipynb`) simply calls this once per split and tabulates the results:

```python
STRATEGIES = ["random", "temporal", "station_grouped", "geographic", "shipped"]
for strat in STRATEGIES:
    r = leakage.train_and_evaluate(ds, df, cache, cfg, strat, device, out_root)
    # collect R2, MAE, RMSE, SD_y_test, Spearman, coverage, straddling ... into one table
```

## 2.2 The causal check — contaminated vs clean  · `src/leakage.py`

No extra training. Inside the **random** model's test set, a photo is **contaminated** if its station —
or a perceptual-hash near-duplicate (`dup_group` from `audit.redundancy_report`) — also appears in that
model's *training* split. We then score contaminated vs clean separately.

```python
def contaminated_vs_clean(sp_random, test_df, dup_df, station_col="station_id"):
    train = sp_random[sp_random["split"] == "train"]
    train_stations = set(train[station_col])
    train_groups   = set(dup_df.merge(train[["_row"]], on="_row")["dup_group"])

    t = test_df.merge(dup_df[["_row", "dup_group"]], on="_row", how="left")
    contaminated = t[station_col].isin(train_stations) | t["dup_group"].isin(train_groups)

    def _score(mask):
        sub = t[mask]
        return {"n": int(len(sub)),
                "R2":  float(Mx.r2_score(sub["y"], sub["pred"])),
                "MAE": float(Mx.mean_absolute_error(sub["y"], sub["pred"]))}
    return {"contaminated": _score(contaminated), "clean": _score(~contaminated),
            "contaminated_fraction": float(contaminated.mean())}
```

**Result of Part 2:** random R² = 0.759 vs station_grouped 0.220 → **Δ_leak = 0.539**. Within the
random test set, **contaminated R² = 0.762** vs **clean R² = −0.86** — the score lives entirely in the
leaked photos. See [`RESULTS.md`](RESULTS.md) §4–§5 and [`10_leakage.md`](10_leakage.md).

---

## Where each result comes from (quick map)

| Result | Notebook | Core code |
|---|---|---|
| Honest R² = 0.220 (station-grouped) | `09_leakage.ipynb` (station_grouped arm) / `kaggle_pipeline.ipynb` | `physics.py`, `model.py`, `losses.py`, `dataset.py`, `splits.py`, `train.py`, `calibrate.py`, `recalibrate.py`, `metrics.py` |
| Leakage gradient + Δ_leak = 0.539 | `09_leakage.ipynb` | `leakage.train_and_evaluate` |
| Contaminated vs clean proof | `09_leakage.ipynb` | `leakage.contaminated_vs_clean`, `audit.redundancy_report` |

*A larger-backbone model can be trained later by changing `model.backbone` in `configs/default.yaml`
and re-running — no code change here. Its result will be added to `RESULTS.md` as a new row.*
