"""Physics-guided visual air-quality estimation -- reusable helper modules.

Each module maps to a phase of the project (and a section of the paper):

    aqi          EPA PM2.5 <-> AQI conversion (used by the C2 error-ceiling work)
    data         load + clean the PM25Vision dataset            (paper Sec 3.2-3.3)
    audit        near-duplicate, per-station, falsification test (paper Sec 3.3)
    physics      Dark Channel Prior + inverted saturation maps   (paper Sec 3.5)
    splits       leakage-safe train/calibration/test splits      (paper Sec 3.4)
    dataset      PyTorch Dataset producing 5-channel tensors      (paper Sec 3.5)
    model        EfficientNet-B0 with monotone quantile heads     (paper Sec 3.6)
    losses       pinball / quantile loss                          (paper Sec 3.7)
    train        training loop                                    (paper Sec 3.12)
    calibrate    conformal prediction                             (paper Sec 3.8)
    recalibrate  isotonic recalibration of the point estimate     (paper Sec 3.11)
    metrics      MAE / RMSE / R2 / coverage / width               (paper Sec 3.11)
    ceiling      C2 unavoidable-error ceiling (OpenAQ)             (paper Sec 3.10)
    leakage      the leakage-gradient experiment (the "story")    (paper Sec 3.4)
    crossval     leakage-safe grouped K-fold cross-validation     (paper Sec 3.11)
    config       load configs/default.yaml
"""

__all__ = [
    "aqi",
    "data",
    "audit",
    "physics",
    "splits",
    "dataset",
    "model",
    "losses",
    "train",
    "calibrate",
    "recalibrate",
    "metrics",
    "ceiling",
    "leakage",
    "crossval",
    "config",
]
