"""Common repository path helpers.

This module avoids fragile relative-path depth assumptions by discovering
the repository root from the current file location.
"""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``pyproject.toml``."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (missing pyproject.toml).")


def code_dir() -> Path:
    return repo_root() / "code"


def code_src_dir() -> Path:
    return code_dir() / "src"


def code_simulations_dir() -> Path:
    return code_dir() / "simulations"


def binary_features_dir() -> Path:
    return code_simulations_dir() / "binary_features"


def binary_features_dgp_dir() -> Path:
    return binary_features_dir() / "dgp"


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def outputs_simulations_dir() -> Path:
    return outputs_dir() / "simulations"


def binary_features_outputs_dir() -> Path:
    return outputs_simulations_dir() / "binary_features"
