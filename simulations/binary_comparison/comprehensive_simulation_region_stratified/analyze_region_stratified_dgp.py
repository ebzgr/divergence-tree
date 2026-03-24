"""
DGP diagnostics for region-stratified simulation.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_hist(series: pd.Series, title: str, out_file: str, log_x: bool = False) -> None:
    vals = series.dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    bins = 30 if len(vals) > 100 else 15
    plt.hist(vals, bins=bins, alpha=0.85, edgecolor="black")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_dgp(results_file: str, output_dir: str, max_per_sim_plots: int = 200) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_pickle(results_file)

    # Global aspect distributions
    aspect_cols = ["noise", "data_size", "sparsity", "dispersion", "rareness"]
    for col in aspect_cols:
        if col in df.columns:
            _save_hist(
                df[col],
                title=f"Distribution of {col}",
                out_file=os.path.join(output_dir, f"aspect_distribution_{col}.png"),
                log_x=(col in ("noise", "data_size")),
            )

    # Realized region distributions over simulations
    for prefix in ["dispersion_real_region", "rareness_real_region"]:
        for r in [1, 2, 3, 4]:
            col = f"{prefix}{r}"
            if col in df.columns:
                _save_hist(
                    df[col],
                    title=f"Distribution of {col}",
                    out_file=os.path.join(output_dir, f"{col}.png"),
                    log_x=False,
                )

    # Target vs realized scatter for region 2
    pairs = [
        ("dispersion_target_region2", "dispersion_real_region2"),
        ("rareness_target_region2", "rareness_real_region2"),
    ]
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        d = df[[x_col, y_col]].dropna()
        if len(d) == 0:
            continue
        plt.figure(figsize=(6, 6))
        if len(d) > max_per_sim_plots:
            d = d.sample(max_per_sim_plots, random_state=0)
        plt.scatter(d[x_col], d[y_col], alpha=0.5, s=15)
        lo = min(d[x_col].min(), d[y_col].min())
        hi = max(d[x_col].max(), d[y_col].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{y_col}_vs_{x_col}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Save numeric summaries
    summary_cols = [c for c in df.columns if c in aspect_cols or c.startswith("dispersion_") or c.startswith("rareness_")]
    if summary_cols:
        desc = df[summary_cols].describe(include="all").T
        desc.to_csv(os.path.join(output_dir, "dgp_summary_statistics.csv"))

    # Per-simulation generated-data summaries from stored train/val/test files.
    base_output_dir = os.path.dirname(os.path.dirname(os.path.dirname(results_file)))
    data_root = os.path.join(base_output_dir, "data", "region_stratified_lambda_twostep_comparison")
    per_sim_rows = []
    if os.path.isdir(data_root):
        for sim_id in sorted(df["simulation_id"].dropna().astype(int).tolist()):
            sim_dir = os.path.join(data_root, f"simulation_{sim_id:06d}")
            train_f = os.path.join(sim_dir, "train_data.pkl")
            val_f = os.path.join(sim_dir, "val_data.pkl")
            test_f = os.path.join(sim_dir, "test_data.pkl")
            if not (os.path.exists(train_f) and os.path.exists(val_f) and os.path.exists(test_f)):
                continue
            train_df = pd.read_pickle(train_f)
            val_df = pd.read_pickle(val_f)
            test_df = pd.read_pickle(test_f)
            merged = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
            row = {"simulation_id": sim_id, "n_total": int(len(merged))}
            if "region_type_true" in merged.columns:
                for r in [1, 2, 3, 4]:
                    row[f"region{r}_obs_share_generated"] = float((merged["region_type_true"] == r).mean())
            per_sim_rows.append(row)

    if per_sim_rows:
        per_sim_df = pd.DataFrame(per_sim_rows).sort_values("simulation_id")
        per_sim_df.to_csv(os.path.join(output_dir, "per_sim_generated_data_summary.csv"), index=False)

        # Plot generated observation share distributions per region across simulations.
        for r in [1, 2, 3, 4]:
            col = f"region{r}_obs_share_generated"
            if col in per_sim_df.columns:
                _save_hist(
                    per_sim_df[col],
                    title=f"Generated observation share distribution (region {r})",
                    out_file=os.path.join(output_dir, f"generated_obs_share_region{r}.png"),
                    log_x=False,
                )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "output")
    results_file = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "analysis", "dgp_diagnostics")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    analyze_dgp(results_file, output_dir)

