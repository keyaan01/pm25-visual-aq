"""Training loop (paper Sec 3.7, 3.12).

Trains the quantile network with pinball loss on the log-AQI target. Settings mirror the
paper: AdamW (lr 3e-4, weight decay 1e-4), cosine-annealed learning rate with a 2-epoch
warm-up, batch 32, up to 40 epochs with early stopping, mixed precision, fixed seeds.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch

from .losses import combined_loss


def set_seed(seed: int = 42):
    """Fix all random seeds so runs are reproducible (paper Sec 3.12)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cosine_warmup(epoch: int, warmup: int, max_epochs: int) -> float:
    """LR multiplier: linear warm-up for `warmup` epochs, then cosine decay to 0."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, max_epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_model(net, loaders, cfg, device, out_dir="outputs",
                max_epochs=None, max_steps_per_epoch=None, verbose=True):
    """Train `net`; save the best checkpoint (lowest calibration loss). Returns history dict.

    `max_epochs` / `max_steps_per_epoch` override the config for quick smoke tests.
    """
    os.makedirs(out_dir, exist_ok=True)
    quantiles = tuple(cfg["quantiles"])
    max_epochs = max_epochs or cfg["train"]["max_epochs"]
    warmup = cfg["train"]["warmup_epochs"]
    patience = cfg["train"]["early_stop_patience"]
    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"
    point_weight = cfg["train"].get("point_loss_weight", 1.0)
    huber_delta = cfg["train"].get("huber_delta", 1.0)
    point_loss = cfg["train"].get("point_loss", "mse")   # "mse" (mean, good R²) or "huber"

    # Standardise the point head's target using the TRAIN split's AQI mean/std (stored on the
    # model so it travels with the checkpoint and is used to de-standardise at inference).
    y_mean, y_std = target_stats(loaders["train"])
    if hasattr(net, "set_target_stats"):
        net.set_target_stats(y_mean, y_std)

    # Optional weight EMA (Stage B): keep an exponential moving average of the weights. We evaluate
    # the cal metric AND checkpoint the EMA weights, since the averaged model is usually a little more
    # accurate and more stable. use_buffers=True so BN running stats are averaged too; the constant
    # y_mean/y_std buffers are unchanged by averaging. Created AFTER set_target_stats so the buffers
    # are already correct when the EMA copy is initialised.
    use_ema = bool(cfg["train"].get("ema", False))
    ema = None
    if use_ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_decay = float(cfg["train"].get("ema_decay", 0.999))
        ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: _cosine_warmup(e, warmup, max_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(out_dir, "best_model.pth")
    history = {"train_loss": [], "cal_point_mae": [], "lr": []}
    # Early stopping tracks calibration point-MAE (in AQI) — the accuracy metric we care about.
    best_mae, best_epoch, bad = float("inf"), -1, 0

    for epoch in range(max_epochs):
        net.train()
        epoch_lr = opt.param_groups[0]["lr"]      # the LR actually used this epoch
        running, seen = 0.0, 0
        for step, (x, y) in enumerate(loaders["train"]):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
                loss = combined_loss(net(x), y, quantiles, point_weight, huber_delta,
                                     y_mean, y_std, point_loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(net)
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()

        train_loss = running / max(1, seen)
        # Evaluate (and later checkpoint) the EMA weights when EMA is on, else the live weights.
        eval_model = ema.module if ema is not None else net
        cal_mae = point_mae_from_outputs(collect_outputs(eval_model, loaders["cal"], device), y_mean, y_std)
        history["train_loss"].append(train_loss)
        history["cal_point_mae"].append(cal_mae)
        history["lr"].append(epoch_lr)
        if verbose:
            print(f"epoch {epoch+1:02d}/{max_epochs}  train={train_loss:.4f}  "
                  f"cal_MAE={cal_mae:.2f} AQI  lr={epoch_lr:.2e}")

        if cal_mae < best_mae - 1e-4:
            best_mae, best_epoch, bad = cal_mae, epoch, 0
            torch.save({"model": eval_model.state_dict(), "quantiles": quantiles,
                        "cfg": dict(cfg), "epoch": epoch, "cal_point_mae": cal_mae,
                        "y_mean": y_mean, "y_std": y_std}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
                break

    history["best_epoch"] = best_epoch
    history["best_cal_point_mae"] = best_mae
    history["checkpoint"] = ckpt_path
    with open(os.path.join(out_dir, "train_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    return history


@torch.no_grad()
def collect_outputs(net, loader, device, tta=False):
    """Run the model over a loader and return raw arrays.

    Returns a dict of numpy arrays:
      "q_log":     (N, Q) quantile predictions (always log space),
      "point_out": (N,)  point-head predictions (in the point head's training space),
      "y_raw":     (N,)  targets in AQI.

    `tta` (test-time augmentation): when True, average each prediction over the image and its
    horizontal flip. The flip is GEOMETRIC (safe for all 5 channels), and averaging two ascending
    quantile vectors stays ascending, so monotonicity is preserved. TTA must be applied to BOTH the
    calibration and test sets (never test only) so the conformal Q and isotonic map stay valid.
    """
    net.eval()
    qs, ps, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = net(x)
        q = out["quantiles"]
        p = out["point"] if "point" in out else out["quantiles"][:, 1]
        if tta:
            out_f = net(torch.flip(x, dims=[-1]))          # horizontal flip (last dim = width)
            q = 0.5 * (q + out_f["quantiles"])
            p_f = out_f["point"] if "point" in out_f else out_f["quantiles"][:, 1]
            p = 0.5 * (p + p_f)
        qs.append(q.cpu().numpy())
        ps.append(p.cpu().numpy())
        ys.append(y.numpy())
    return {"q_log": np.concatenate(qs), "point_out": np.concatenate(ps),
            "y_raw": np.concatenate(ys)}


@torch.no_grad()
def collect_features(net, loader, device):
    """Backbone POOLED features for every item in a loader (order matches the loader; shuffle=False).

    Returns {"feats": (N, D), "y_raw": (N,)}. `collect_outputs` returns predictions; this returns the
    penultimate features `net.backbone(x)` instead — needed by the abstention OOD gate, which fits a
    feature-space detector on calibration good-images and scores test / out-of-distribution images.
    """
    net.eval()
    fs, ys = [], []
    for x, y in loader:
        feats = net.backbone(x.to(device))
        fs.append(feats.cpu().numpy())
        ys.append(y.numpy())
    return {"feats": np.concatenate(fs), "y_raw": np.concatenate(ys)}


def point_to_aqi(point_out, y_mean, y_std):
    """De-standardise the point head's z-score output back to AQI (clip at 0)."""
    return np.maximum(np.asarray(point_out) * y_std + y_mean, 0.0)


def point_mae_from_outputs(out, y_mean, y_std):
    """Calibration point-MAE in AQI. Drives early stopping (Stage A)."""
    pt = point_to_aqi(out["point_out"], y_mean, y_std)
    return float(np.mean(np.abs(pt - out["y_raw"])))


def target_stats(loader):
    """Mean/std of the raw AQI targets in a loader's dataset (for standardising the point head)."""
    y = np.asarray(loader.dataset.targets, dtype=float)
    return float(y.mean()), float(y.std() + 1e-6)


def load_checkpoint(path, net, map_location="cpu"):
    """Load a saved checkpoint's weights into `net`; return the checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    net.load_state_dict(ckpt["model"])
    return ckpt
