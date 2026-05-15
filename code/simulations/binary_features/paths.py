"""Path helpers for binary_features simulation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

BINARY_FEATURES_DIR = Path(__file__).resolve().parent


def setup_binary_features_path() -> Path:
    """Insert binary_features on sys.path so local modules import cleanly."""
    root = str(BINARY_FEATURES_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return BINARY_FEATURES_DIR


def repo_root() -> Path:
    for candidate in (BINARY_FEATURES_DIR, *BINARY_FEATURES_DIR.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (missing pyproject.toml).")
