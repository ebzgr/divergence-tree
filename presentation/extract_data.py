#!/usr/bin/env python3
"""
Convert all_simulations_results.pkl to static/simulations.json for the static HTML presentation.

Reads the aggregated simulation DataFrame and exports a compact JSON array with
factor columns and per-method metrics (flat column names matching the DataFrame).

Usage (from repository root, where ``outputs/`` lives):
  python presentation/extract_data.py
  python presentation/extract_data.py --pkl outputs/simulations/binary_features/aggregated/lambda_twostep_comparison/all_simulations_results.pkl

Use an absolute path if the pickle is outside the repo (must start with ``/`` on Linux).

Environment:
  SIMULATIONS_PKL  overrides default pickle path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _standard_pickle_relative() -> Path:
    """Path segments under repo root: outputs/simulations/.../all_simulations_results.pkl"""
    return (
        Path("outputs")
        / "simulations"
        / "binary_features"
        / "aggregated"
        / "lambda_twostep_comparison"
        / "all_simulations_results.pkl"
    )


def _find_default_pickle(repo_root: Path) -> Optional[Path]:
    """Resolve pickle path: env, then repo-relative standard location."""
    env = os.environ.get("SIMULATIONS_PKL")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    candidates = [
        repo_root / _standard_pickle_relative(),
        Path.cwd() / _standard_pickle_relative(),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _resolve_explicit_pickle(user_path: Path, repo_root: Path) -> tuple[Optional[Path], list[Path]]:
    """
    Resolve --pkl to an existing file. Tries several bases so relative paths work from repo root.

    Also fixes a common mistake: ``data/ebrahim/...`` (missing leading ``/``) -> ``/data/ebrahim/...``.
    """
    tried: list[Path] = []
    p = user_path.expanduser()

    def try_path(candidate: Path) -> Optional[Path]:
        r = candidate.resolve()
        tried.append(r)
        return r if r.is_file() else None

    if p.is_file():
        return p.resolve(), tried

    if p.is_absolute():
        found = try_path(p)
        return (found, tried)

    # Relative path: try cwd, then repo root (so running from repo root works)
    for base in (Path.cwd(), repo_root):
        found = try_path(base / p)
        if found is not None:
            return found, tried

    # Typo: user wrote "data/ebrahim/..." instead of "/data/ebrahim/..."
    parts = p.parts
    if parts and parts[0] == "data" and len(parts) > 1:
        fixed = Path("/") / p
        found = try_path(fixed)
        if found is not None:
            return found, tried

    return None, tried


# Method prefixes in v3 aggregated results (order for charts)
METHOD_PREFIXES = [
    "divtree_lambda0",
    "divtree_lambda1_region2",
    "divtree_lambda2_region2",
    "divtree_lambda4_region2",
    "divtree_lambda8_region2",
    "twostep_tuned",
    "twostep_recall",
]

# Metrics to export per method (cpu_time uses total for twostep when available)
METRICS = [
    "accuracy",
    "recall_region_2",
    "precision_region_2",
    "fnr_region_2",
    "f1_region_2",
    "n_leaves",
    "cpu_time",
]

FACTOR_COLS = [
    "simulation_id",
    "noise",
    "data_size",
    "sparsity",
    "rareness",
    "intensity",
    "combo_share_real_region2",  # dispersion proxy: share of combos in Region 2
]


def build_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select and derive columns; align cpu_time for TwoStep with total_cpu_time."""
    out = pd.DataFrame()
    for c in FACTOR_COLS:
        if c in df.columns:
            out[c] = df[c]
        elif c == "combo_share_real_region2" and c not in df.columns:
            out[c] = np.nan

    for prefix in METHOD_PREFIXES:
        for metric in METRICS:
            col = f"{prefix}_{metric}"
            if metric == "cpu_time" and prefix in ("twostep_tuned", "twostep_recall"):
                total_col = f"{prefix}_total_cpu_time"
                if total_col in df.columns:
                    out[col] = df[total_col]
                elif col in df.columns:
                    out[col] = df[col]
                else:
                    out[col] = np.nan
            else:
                if col in df.columns:
                    out[col] = df[col]
                else:
                    out[col] = np.nan

    return out


def dataframe_to_records(df: pd.DataFrame):
    """Round floats for smaller JSON; NaN -> None."""
    records = []
    for _, row in df.iterrows():
        rec: dict = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = None
            elif isinstance(v, (np.floating, float)):
                rec[k] = float(round(v, 5))
            elif isinstance(v, (np.integer, int)):
                rec[k] = int(v)
            else:
                rec[k] = v
        records.append(rec)
    return records


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Export simulations pickle to JSON for the HTML presentation.",
        epilog=(
            "Examples (run from repository root):\n"
            "  %(prog)s\n"
            "  %(prog)s --pkl outputs/simulations/binary_features/aggregated/"
            "lambda_twostep_comparison/all_simulations_results.pkl\n"
            "Absolute path (note the leading /):\n"
            "  %(prog)s --pkl /path/to/repo/outputs/"
            "simulations/binary_features/aggregated/lambda_twostep_comparison/"
            "all_simulations_results.pkl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help=(
            "Path to all_simulations_results.pkl. "
            "If omitted, uses SIMULATIONS_PKL or the standard path under repo outputs/. "
            "Relative paths are resolved from the current directory and from the repo root."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: presentation/static/simulations.json)",
    )
    args = parser.parse_args()

    out_path = args.output or (script_dir / "static" / "simulations.json")

    if args.pkl is None:
        pkl_path = _find_default_pickle(repo_root)
        tried_msg = ""
    else:
        pkl_path, tried = _resolve_explicit_pickle(args.pkl, repo_root)
        tried_msg = "\nPaths checked:\n  " + "\n  ".join(str(t) for t in tried)

    if pkl_path is None or not pkl_path.is_file():
        std = repo_root / _standard_pickle_relative()
        raise SystemExit(
            "Could not find all_simulations_results.pkl.\n\n"
            "Fix one of the following:\n"
            "  • Run this script from the repository root, or pass --pkl explicitly.\n"
            "  • Use the path relative to repo root:\n"
            f"      --pkl {_standard_pickle_relative()}\n"
            "  • Use a full absolute path (must start with / on Linux), e.g.:\n"
            f"      --pkl {std}\n"
            "  • Set environment variable SIMULATIONS_PKL to the pickle file.\n"
            f"{tried_msg}\n\n"
            f"Expected file at (under your clone): {std}"
        )

    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame):
        raise SystemExit(f"Expected DataFrame, got {type(df)}")

    export_df = build_export_dataframe(df)
    records = dataframe_to_records(export_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "source_pickle": str(pkl_path),
                    "n_rows": len(records),
                    "method_prefixes": METHOD_PREFIXES,
                    "metrics": METRICS,
                },
                "rows": records,
            },
            f,
            ensure_ascii=False,
        )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({len(records)} rows, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
