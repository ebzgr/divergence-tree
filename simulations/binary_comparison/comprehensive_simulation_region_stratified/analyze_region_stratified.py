"""
Analysis for region-stratified lambda + TwoStep simulation results.
"""

import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(BINARY_COMPARISON_DIR, "comprehensive_simulation"))
import utils


def _method_configs():
    return [
        (0, "divtree_lambda0", "DivTree λ=0"),
        (1, "divtree_lambda1_region2", "DivTree λ=1"),
        (2, "divtree_lambda2_region2", "DivTree λ=2"),
        (4, "divtree_lambda4_region2", "DivTree λ=4"),
        (6, "divtree_lambda6_region2", "DivTree λ=6"),
        (8, "divtree_lambda8_region2", "DivTree λ=8"),
        (None, "twostep_tuned", "TwoStep (tuned)"),
        (None, "twostep_recall", "TwoStep (recall)"),
        (None, "twostep_cap120", "TwoStep (cap120)"),
    ]


def create_method_comparison_plots(df: pd.DataFrame, output_dir: str, metrics: List[str]) -> None:
    utils.safe_makedirs(output_dir)
    labels = {
        "accuracy": "Accuracy",
        "f1_region_2": "F1 Score (Region 2)",
        "fnr_region_2": "False Negative Rate (Region 2)",
        "precision_region_2": "Precision (Region 2)",
        "recall_region_2": "Recall (Region 2)",
        "n_leaves": "Number of Leaves",
        "runtime": "Runtime (seconds)",
        "cpu_time": "CPU Time (seconds)",
    }
    for metric in metrics:
        data = []
        names = []
        for _, prefix, label in _method_configs():
            if metric == "cpu_time" and prefix.startswith("twostep_"):
                col = f"{prefix}_total_cpu_time"
            else:
                col = f"{prefix}_{metric}"
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            data.append((vals.mean(), vals.std()))
            names.append(label)
        if not data:
            continue
        x = np.arange(len(names))
        means = [d[0] for d in data]
        stds = [d[1] for d in data]
        plt.figure(figsize=(12, 6))
        plt.bar(x, means, yerr=stds, capsize=4, edgecolor="black")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel(labels.get(metric, metric))
        plt.title(f"{labels.get(metric, metric)} by Method")
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"method_comparison_{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def create_factor_comparison_plots(df: pd.DataFrame, output_dir: str) -> None:
    utils.safe_makedirs(output_dir)
    factors = {
        "noise": ("noise", True),
        "data_size": ("data_size", True),
        "sparsity": ("sparsity", False),
        "dispersion": ("dispersion", True),
        "rareness": ("rareness", True),
    }
    metrics = ["accuracy", "recall_region_2", "fnr_region_2", "precision_region_2"]
    methods = [
        ("divtree_lambda0", "λ=0"),
        ("divtree_lambda2_region2", "λ=2"),
        ("divtree_lambda4_region2", "λ=4"),
        ("twostep_tuned", "TwoStep (tuned)"),
        ("twostep_recall", "TwoStep (recall)"),
        ("twostep_cap120", "TwoStep (cap120)"),
    ]
    for factor_name, (factor_col, log_x) in factors.items():
        if factor_col not in df.columns:
            continue
        vals_unique = sorted(df[factor_col].dropna().unique())
        if len(vals_unique) < 2:
            continue
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            plotted = False
            for prefix, label in methods:
                col = f"{prefix}_{metric}"
                if col not in df.columns:
                    continue
                means_x = []
                means_y = []
                for v in vals_unique:
                    subset = df[df[factor_col] == v][col].dropna()
                    if len(subset) == 0:
                        continue
                    means_x.append(v)
                    means_y.append(subset.mean())
                if len(means_x) < 2:
                    continue
                plotted = True
                plt.plot(means_x, means_y, marker="o", linewidth=2, label=label)
            if not plotted:
                plt.close()
                continue
            if log_x and factor_col in ("noise", "data_size"):
                plt.xscale("log")
            plt.xlabel(factor_col)
            plt.ylabel(metric)
            plt.title(f"{metric} across {factor_col}")
            plt.legend()
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"factor_comparison_{factor_name}_{metric}.png"), dpi=300, bbox_inches="tight")
            plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str, metrics: List[str]) -> None:
    utils.safe_makedirs(output_dir)
    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for _, prefix, label in _method_configs():
            col = f"{prefix}_total_cpu_time" if metric == "cpu_time" and prefix.startswith("twostep_") else f"{prefix}_{metric}"
            if col in df.columns and df[col].notna().any():
                vals = df[col].dropna().values
                row[f"{label} (mean)"] = float(np.mean(vals))
                row[f"{label} (std)"] = float(np.std(vals))
            else:
                row[f"{label} (mean)"] = np.nan
                row[f"{label} (std)"] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "method_comparison_summary.csv"), index=False)


def analyze_region_stratified(results_file: str, output_dir: str, metrics: Optional[List[str]] = None) -> None:
    if metrics is None:
        metrics = ["accuracy", "f1_region_2", "fnr_region_2", "precision_region_2", "recall_region_2", "n_leaves", "runtime", "cpu_time"]
    df = pd.read_pickle(results_file)
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    create_method_comparison_plots(df, plots_dir, metrics)
    create_factor_comparison_plots(df, plots_dir)
    create_summary_table(df, tables_dir, metrics)


if __name__ == "__main__":
    base_dir = os.path.join(SCRIPT_DIR, "output")
    results_file = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "region_stratified_lambda_twostep_comparison", "analysis")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found.")
        sys.exit(1)
    analyze_region_stratified(results_file, output_dir)

