"""Load `configs/default.yaml` from anywhere (repo root, a notebook, or Colab).

Notebooks run from different working directories, so instead of hard-coding a path
we search upward for the `configs/default.yaml` file. Access values with normal
dict/attribute syntax, e.g. `cfg["train"]["lr"]` or `cfg.train.lr`.
"""
from __future__ import annotations

from pathlib import Path

import yaml


class Config(dict):
    """A dict that also allows attribute access and nests recursively."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up until we see configs/default.yaml."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "configs" / "default.yaml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | Path | None = None) -> Config:
    """Return the parsed configuration as a `Config` (dict with attribute access)."""
    if path is None:
        path = repo_root() / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
