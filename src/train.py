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


@torch.no_grad()
def evaluate_loss(net, loader, quantiles, device, point_weight=1.0, huber_delta=1.0) -> float:
    """Mean combined loss (pinball + Huber) over a loader (no gradients)."""
    net.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = combined_loss(net(x), y, quantiles, point_weight, huber_delta)
        total += loss.item() * len(y)
        n += len(y)
    return total / max(1, n)


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

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: _cosine_warmup(e, warmup, max_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(out_dir, "best_model.pth")
    history = {"train_loss": [], "cal_loss": [], "lr": []}
    best_cal, best_epoch, bad = float("inf"), -1, 0

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
                loss = combined_loss(net(x), y, quantiles, point_weight, huber_delta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()

        train_loss = running / max(1, seen)
        cal_loss = evaluate_loss(net, loaders["cal"], quantiles, device, point_weight, huber_delta)
        history["train_loss"].append(train_loss)
        history["cal_loss"].append(cal_loss)
        history["lr"].append(epoch_lr)
        if verbose:
            print(f"epoch {epoch+1:02d}/{max_epochs}  train={train_loss:.4f}  "
                  f"cal={cal_loss:.4f}  lr={epoch_lr:.2e}")

        if cal_loss < best_cal - 1e-5:
            best_cal, best_epoch, bad = cal_loss, epoch, 0
            torch.save({"model": net.state_dict(), "quantiles": quantiles,
                        "cfg": dict(cfg), "epoch": epoch, "cal_loss": cal_loss}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
                break

    history["best_epoch"] = best_epoch
    history["best_cal_loss"] = best_cal
    history["checkpoint"] = ckpt_path
    with open(os.path.join(out_dir, "train_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    return history


@torch.no_grad()
def collect_outputs(net, loader, device):
    """Run the model over a loader and return raw outputs in the (log) target space.

    Returns a dict of numpy arrays:
      "q_log":     (N, Q) quantile predictions,
      "point_log": (N,)  point-head predictions (falls back to the median if no point head),
      "y_log":     (N,)  targets.
    Kept in log space so the eval step can apply the smearing correction before exponentiating.
    """
    net.eval()
    qs, ps, ys = [], [], []
    for x, y in loader:
        out = net(x.to(device))
        qs.append(out["quantiles"].cpu().numpy())
        ps.append((out["point"] if "point" in out else out["quantiles"][:, 1]).cpu().numpy())
        ys.append(y.numpy())
    return {"q_log": np.concatenate(qs), "point_log": np.concatenate(ps),
            "y_log": np.concatenate(ys)}


@torch.no_grad()
def collect_predictions(net, loader, device, log_target=True):
    """Back-compat: return (quantiles_AQI (N,Q), targets_AQI (N,)) using the quantile heads."""
    o = collect_outputs(net, loader, device)
    q, y = o["q_log"], o["y_log"]
    return (np.exp(q), np.exp(y)) if log_target else (q, y)


def load_checkpoint(path, net, map_location="cpu"):
    """Load a saved checkpoint's weights into `net`; return the checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    net.load_state_dict(ckpt["model"])
    return ckpt
