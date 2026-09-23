"""PyTorch Dataset + DataLoaders producing the 5-channel input (paper Sec 3.5, 3.12).

Each item is a `(5, 224, 224)` tensor (RGB + transmission + inverted saturation) and a
target. Two important, deliberate choices from the paper:

* **Train on log(AQI).** The label is right-skewed (many moderate days, few extreme ones);
  a handful of extreme days would otherwise dominate the loss. We train on `log(AQI)` and
  convert back at prediction time. Ordering is preserved, so the monotonicity/guarantee
  still hold.

* **Geometric augmentation ONLY, applied identically to all 5 channels.** Flips / small
  rotations / mild crops are fine — they move the colour and physics channels together.
  **Photometric** augmentation (brightness / contrast / saturation / hue jitter) is
  *excluded entirely*, because it would alter exactly the image statistics that encode
  particle concentration — corrupting the training signal while leaving the label unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

# EPA AQI band edges used to balance the classes when sampling.
AQI_BAND_EDGES = (0, 50, 100, 150, 200, 300, 10_000)


def balanced_weights(targets_aqi, power=0.5, edges=AQI_BAND_EDGES):
    """Per-sample sampling weights that oversample rare (extreme-AQI) bands.

    weight ∝ (1 / band_frequency) ** power. `power=0` = no balancing (natural),
    `power=1` = full inverse-frequency (aggressive), `power=0.5` = a gentle middle ground
    (the default) that shows the model more high-AQI photos without over-flooding.
    """
    targets_aqi = np.asarray(targets_aqi, dtype=float)
    band = np.digitize(targets_aqi, edges[1:-1])          # 0..len(edges)-2
    counts = np.bincount(band, minlength=len(edges) - 1).astype(float)
    freq = counts[band]
    return (1.0 / np.maximum(freq, 1.0)) ** power

from . import physics
from .data import get_image


class PM25Dataset(Dataset):
    def __init__(self, ds, df_split, cache, size=224, target_col="pm25",
                 log_target=True, augment=False):
        self.ds = ds
        self.cache = cache
        self.size = size
        self.log_target = log_target
        self.rows = df_split["_row"].to_numpy()
        self.targets = df_split[target_col].to_numpy(dtype=np.float32)
        # geometric-only augmentation (train split); acts on all 5 channels together
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=8),
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        ]) if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        if self.cache is not None:
            x = physics.five_channel_cached(get_image(self.ds, row), self.cache, row, self.size)
        else:  # fallback: compute on the fly (slower)
            x = physics.five_channel(get_image(self.ds, row), size=self.size)
        x = torch.from_numpy(x)
        if self.aug is not None:
            x = self.aug(x)
        # Return the RAW AQI target. The loss applies log for the quantile heads and the
        # configured transform for the point head; eval inverts them. Keeping raw here lets the
        # point head train in raw space (best for R²) or log space, chosen by config.
        return x, torch.tensor(float(self.targets[i]), dtype=torch.float32)


def make_dataloaders(ds, df_with_split, cache, cfg, num_workers=2):
    """Build train / cal / test DataLoaders from a split-annotated metadata frame.

    Class-balanced sampling (if enabled in the config) is applied to the **train loader only**,
    so calibration and test keep the natural distribution — that's what makes the conformal
    guarantee and the reported metrics honest.
    """
    size = cfg["data"]["image_size"]
    target_col = cfg["data"]["target_col"]
    log_target = cfg["train"]["log_target"]
    batch = cfg["train"]["batch_size"]
    balanced = cfg["train"].get("balanced_sampling", False)
    power = cfg["train"].get("balance_power", 0.5)

    def loader(split, augment, shuffle, sampler=None):
        sub = df_with_split[df_with_split["split"] == split]
        dset = PM25Dataset(ds, sub, cache, size=size, target_col=target_col,
                           log_target=log_target, augment=augment)
        return DataLoader(dset, batch_size=batch, shuffle=shuffle, sampler=sampler,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=False)

    train_sampler = None
    if balanced:
        train_df = df_with_split[df_with_split["split"] == "train"]
        w = balanced_weights(train_df[target_col].to_numpy(), power=power)
        train_sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                              num_samples=len(w), replacement=True)

    return {
        # a sampler and shuffle are mutually exclusive: shuffle=False when sampling
        "train": loader("train", augment=True, shuffle=(train_sampler is None), sampler=train_sampler),
        "cal": loader("cal", augment=False, shuffle=False),
        "test": loader("test", augment=False, shuffle=False),
    }
