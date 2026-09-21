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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

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
        y = float(self.targets[i])
        if self.log_target:
            y = np.log(max(y, 1e-6))          # AQI >= 1, so log is well-defined
        return x, torch.tensor(y, dtype=torch.float32)


def make_dataloaders(ds, df_with_split, cache, cfg, num_workers=2):
    """Build train / cal / test DataLoaders from a split-annotated metadata frame."""
    size = cfg["data"]["image_size"]
    target_col = cfg["data"]["target_col"]
    log_target = cfg["train"]["log_target"]
    batch = cfg["train"]["batch_size"]

    def loader(split, augment, shuffle):
        sub = df_with_split[df_with_split["split"] == split]
        dset = PM25Dataset(ds, sub, cache, size=size, target_col=target_col,
                           log_target=log_target, augment=augment)
        return DataLoader(dset, batch_size=batch, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=False)

    return {
        "train": loader("train", augment=True, shuffle=True),
        "cal": loader("cal", augment=False, shuffle=False),
        "test": loader("test", augment=False, shuffle=False),
    }
