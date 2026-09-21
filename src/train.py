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

from .losses import pinball_loss


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
def evaluate_loss(net, loader, quantiles, device) -> float:
    """Mean pinball loss over a loader (no gradients)."""
    net.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = pinball_loss(net(x), y, quantiles)
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
                loss = pinball_loss(net(x), y, quantiles)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()

        train_loss = running / max(1, seen)
        cal_loss = evaluate_loss(net, loaders["cal"], quantiles, device)
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
def collect_predictions(net, loader, device, log_target=True):
    """Run the model over a loader and return (preds_AQI (N,Q), targets_AQI (N,)).

    Converts out of log space back to AQI so downstream calibration/metrics work in the
    original, interpretable units.
    """
    net.eval()
    preds, ys = [], []
    for x, y in loader:
        out = net(x.to(device)).cpu().numpy()
        preds.append(out)
        ys.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    if log_target:
        preds = np.exp(preds)
        ys = np.exp(ys)
    return preds, ys


def load_checkpoint(path, net, map_location="cpu"):
    """Load a saved checkpoint's weights into `net`; return the checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    net.load_state_dict(ckpt["model"])
    return ckpt
