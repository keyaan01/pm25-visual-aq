"""Measure data-leakage inflation across splitting strategies (the "story" phase, part A).

Two things, per the research-backed design:

1. **Split gradient** — train the SAME model/config on each split strategy (random, temporal,
   station_grouped, geographic, shipped) and tabulate metrics. The honest number is
   station_grouped; `random` is the leaky control. We always report SD(y_test) + MAE next to R²
   because R² depends on the test-set variance (a key confounder).

2. **Causal clincher (no extra training)** — on the `random` model, split its test set into
   *contaminated* (a photo whose station OR perceptual-hash duplicate-group also appears in the
   random TRAIN set) vs *clean*, and compare accuracy. If R²(contaminated) >> R²(clean) ≈
   R²(station_grouped), the inflation demonstrably comes from same-place leakage, not from the
   test set merely being easier.

All heavy lifting reuses the existing modules; this file is just orchestration so the notebook
stays thin and the logic is unit-testable on the fixture.
"""
from __future__ import annotations

import os

import numpy as np

from . import calibrate as C
from . import dataset as D
from . import metrics as Mx
from . import model as M
from . import recalibrate as R
from . import splits as S
from . import train as T


def train_and_evaluate(ds, df, cache, cfg, strategy, device, out_root, seed=None,
                       max_epochs=None, num_workers=2, verbose=False, sp=None):
    """Train on one split and evaluate; return metrics + per-test-row predictions.

    Everything except the split is held constant (same cfg, same seed, same cache) so comparisons
    are apples-to-apples. `strategy` names the split (and the checkpoint sub-dir). Pass a pre-built
    `sp` (a df with a 'split' column) to evaluate an arbitrary split — e.g. a cross-validation fold;
    otherwise the split is built by `make_splits(strategy)`.

    ACCURACY LEVERS (config, honest — everything is selected on cal, test is evaluated once):
      * `train.tta`: average each prediction over the image + its horizontal flip, on BOTH cal and
        test, so the conformal Q and isotonic map stay valid.
      * `train.ensemble_seeds`: a list of seeds; when it has >1 entry we train one model per seed on
        the SAME split, average the (log-space, still-ascending) quantiles and the standardised point
        outputs across members, then fit isotonic + conformal ONCE on the averaged cal outputs. A
        single-entry list reproduces the plain single-model run exactly.
    """
    base_seed = seed if seed is not None else cfg["seed"]
    if sp is None:
        T.set_seed(base_seed)
        sp = S.make_splits(df, strategy=strategy, seed=base_seed,
                           station_col=cfg["data"]["station_col"], time_col=cfg["data"]["time_col"],
                           lon_col=cfg["data"]["lon_col"], lat_col=cfg["data"]["lat_col"])
    # The split is fixed; all ensemble members share it (only their init/sampling/augmentation seed
    # differs, which is the source of ensemble diversity).
    loaders = D.make_dataloaders(ds, sp, cache, cfg, num_workers=num_workers)
    seeds = list(cfg["train"].get("ensemble_seeds", [base_seed])) or [base_seed]
    tta = bool(cfg["train"].get("tta", False))

    cal_qlogs, cal_pouts, test_qlogs, test_pouts = [], [], [], []
    best_epochs, ymean, ystd, cal_y, test_y = [], None, None, None, None
    for s in seeds:
        T.set_seed(s)
        net = M.build_model(cfg).to(device)
        out_dir = os.path.join(out_root, strategy if len(seeds) == 1 else f"{strategy}/seed{s}")
        hist = T.train_model(net, loaders, cfg, device=device, out_dir=out_dir,
                             max_epochs=max_epochs, verbose=verbose)
        T.load_checkpoint(os.path.join(out_dir, "best_model.pth"), net, map_location=device)
        cal = T.collect_outputs(net, loaders["cal"], device, tta=tta)
        test = T.collect_outputs(net, loaders["test"], device, tta=tta)
        ymean, ystd = float(net.y_mean), float(net.y_std)   # identical across members (same train split)
        cal_y, test_y = cal["y_raw"], test["y_raw"]         # identical order across members (shuffle=False)
        cal_qlogs.append(cal["q_log"]); cal_pouts.append(cal["point_out"])
        test_qlogs.append(test["q_log"]); test_pouts.append(test["point_out"])
        best_epochs.append(hist["best_epoch"] + 1)

    # Average across ensemble members (a single member is a no-op average).
    cal_qlog = np.mean(cal_qlogs, axis=0); cal_pout = np.mean(cal_pouts, axis=0)
    test_qlog = np.mean(test_qlogs, axis=0); test_pout = np.mean(test_pouts, axis=0)

    point = T.point_to_aqi(test_pout, ymean, ystd)
    if cfg["train"].get("recalibrate", True):
        iso = R.fit_isotonic(T.point_to_aqi(cal_pout, ymean, ystd), cal_y)
        point = R.apply_isotonic(iso, point)
    Q = C.conformal_Q(np.exp(cal_qlog), cal_y, cfg["calibration"]["coverage"])
    intervals = C.apply_conformal(np.exp(test_qlog), Q)
    y = test_y

    rep = Mx.report(point, intervals, y, cfg["calibration"]["coverage"])
    rep["SD_y_test"] = float(np.std(y))
    rep["n_train"] = int((sp["split"] == "train").sum())
    rep["n_test"] = int(len(y))
    rep["straddling_stations"] = S.split_report(sp, cfg["data"]["station_col"])["stations_straddling_splits"]
    rep["best_epoch"] = int(round(float(np.mean(best_epochs))))
    rep["n_ensemble"] = len(seeds)
    rep["tta"] = tta

    # per-test-row frame (loader used shuffle=False, so order matches sp[test] row order)
    test_df = sp[sp["split"] == "test"].copy().reset_index(drop=True)
    test_df["pred"] = point
    test_df["y"] = y
    # End-to-end alignment guard: the label the frame carries (target_col, e.g. pm25) MUST equal the
    # target the image loader produced (y). If a reorder ever slipped in between the metadata frame
    # and the loader, this fails loudly instead of silently mispairing predictions with labels.
    _label = test_df[cfg["data"]["target_col"]].to_numpy(dtype=float)
    assert np.allclose(_label, np.asarray(test_df["y"], dtype=float)), (
        "alignment broken: the frame's label column != the loader's target on the test split")
    return {"strategy": strategy, "report": rep, "sp": sp, "test_df": test_df}


def contaminated_vs_clean(sp_random, test_df, dup_df, station_col="station_id"):
    """Split the RANDOM test set into contaminated vs clean and score each.

    contaminated = test photo whose `station_id` OR perceptual-hash `dup_group` also appears in the
    random TRAIN split. `dup_df` must have columns [_row, dup_group] from audit.redundancy_report.
    """
    train = sp_random[sp_random["split"] == "train"]
    train_stations = set(train[station_col])
    train_groups = set(dup_df.merge(train[["_row"]], on="_row")["dup_group"])

    t = test_df.merge(dup_df[["_row", "dup_group"]], on="_row", how="left")
    contaminated = t[station_col].isin(train_stations) | t["dup_group"].isin(train_groups)

    def _score(mask):
        sub = t[mask]
        if len(sub) < 5:
            return {"n": int(len(sub)), "R2": float("nan"), "MAE": float("nan"),
                    "RMSE": float("nan"), "SD_y": float("nan")}
        # Report SD_y and RMSE beside R2: on a low-variance subset (e.g. the small "clean" set of
        # single-image stations) R2 = 1-(RMSE/SD)^2 goes sharply negative on even a modest RMSE, so
        # R2's magnitude is variance-amplified. The MAE (and RMSE) are the honest, variance-free
        # effect sizes; lead the interpretation with those.
        return {"n": int(len(sub)),
                "R2": float(Mx.r2_score(sub["y"], sub["pred"])),
                "MAE": float(Mx.mean_absolute_error(sub["y"], sub["pred"])),
                "RMSE": float(np.sqrt(Mx.mean_squared_error(sub["y"], sub["pred"]))),
                "SD_y": float(np.std(sub["y"]))}

    return {"contaminated": _score(contaminated), "clean": _score(~contaminated),
            "contaminated_fraction": float(contaminated.mean())}
