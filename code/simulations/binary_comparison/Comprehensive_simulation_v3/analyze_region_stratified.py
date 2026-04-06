"""
Analysis for region-stratified lambda + TwoStep simulation results (v3).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = next(p for p in Path(SCRIPT_DIR).resolve().parents if (p / "pyproject.toml").exists())
sys.path.append(os.path.join(BINARY_COMPARISON_DIR, "comprehensive_simulation"))
import utils


def _method_configs():
    return [
        (0, "divtree_lambda0", "DivTree λ=0"),
        (1, "divtree_lambda1_region2", "DivTree λ=1"),
        (2, "divtree_lambda2_region2", "DivTree λ=2"),
        (4, "divtree_lambda4_region2", "DivTree λ=4"),
        (8, "divtree_lambda8_region2", "DivTree λ=8"),
        (None, "twostep_tuned", "TwoStep (tuned)"),
        (None, "twostep_recall", "TwoStep (recall)"),
    ]


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _round5(x: object) -> object:
    """Round floats for JSON readability; keep ints/None as-is."""
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return x
    if not np.isfinite(xf):
        return None
    return float(f"{xf:.5f}")


def _metric_bad_threshold(metric: str) -> Tuple[str, float]:
    if metric in ("accuracy", "precision_region_2", "recall_region_2", "f1_region_2"):
        return "low", 0.55
    if metric == "fnr_region_2":
        return "high", 0.45
    if metric in ("runtime", "cpu_time"):
        return "high", float("nan")
    if metric == "n_leaves":
        return "high", float("nan")
    return "low", float("nan")


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    qs = np.quantile(values, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    return {
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q10": float(qs[2]),
        "q50": float(qs[3]),
        "q90": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
    }


def create_diagnostics_tables(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str],
    *,
    verbose: bool = True,
) -> None:
    utils.safe_makedirs(output_dir)

    tail_rows: List[Dict[str, object]] = []
    for _, prefix, method_label in _method_configs():
        for metric in metrics:
            col = (
                f"{prefix}_total_cpu_time"
                if metric == "cpu_time" and prefix.startswith("twostep_")
                else f"{prefix}_{metric}"
            )
            if col not in df.columns:
                continue
            vals = df[col].dropna().to_numpy()
            if vals.size < 10:
                continue

            direction, thr = _metric_bad_threshold(metric)
            if metric in ("runtime", "cpu_time", "n_leaves") or (isinstance(thr, float) and np.isnan(thr)):
                thr = float(np.quantile(vals, 0.95))
                direction = "high"

            bad_rate = float(np.mean(vals <= thr)) if direction == "low" else float(np.mean(vals >= thr))
            q = _quantiles(vals)
            tail_rows.append(
                {
                    "method": method_label,
                    "prefix": prefix,
                    "metric": metric,
                    "n": int(vals.size),
                    "bad_direction": direction,
                    "bad_threshold": float(thr),
                    "bad_rate": bad_rate,
                    **q,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            )

    tail_csv = os.path.join(output_dir, "diagnostics_tail_rates_quantiles.csv")
    pd.DataFrame(tail_rows).sort_values(["metric", "method"]).to_csv(tail_csv, index=False)
    _log(f"  [diagnostics] wrote {tail_csv}", verbose)

    factor_defs = {
        "noise": {"column": "noise", "is_discrete": False, "log_x": True},
        "data_size": {"column": "data_size", "is_discrete": False, "log_x": True},
        "sparsity": {"column": "sparsity", "is_discrete": True, "log_x": False},
        "rareness": {"column": "rareness", "is_discrete": False, "log_x": False},
        "intensity": {"column": "intensity", "is_discrete": False, "log_x": True},
    }
    method_plot_configs = [
        ("divtree_lambda0", "λ=0"),
        ("divtree_lambda2_region2", "λ=2"),
        ("divtree_lambda4_region2", "λ=4"),
        ("divtree_lambda8_region2", "λ=8"),
        ("twostep_tuned", "TwoStep (tuned)"),
        ("twostep_recall", "TwoStep (recall)"),
    ]
    metrics_01 = {"accuracy", "recall_region_2", "fnr_region_2", "precision_region_2", "f1_region_2"}
    variability_rows: List[Dict[str, object]] = []

    for factor_name, finfo in factor_defs.items():
        factor_col = finfo["column"]
        if factor_col not in df.columns:
            continue
        is_discrete = finfo["is_discrete"]
        df_plot = df[[factor_col]].copy()
        if is_discrete:
            df_plot["_factor_bin"] = df[factor_col]
            bin_values = sorted(df_plot["_factor_bin"].dropna().unique())
        else:
            n_unique = len(df[factor_col].dropna().unique())
            if n_unique < 2:
                continue
            n_bins = min(20, max(10, n_unique // 10))
            factor_min = float(df[factor_col].min())
            factor_max = float(df[factor_col].max())
            if finfo.get("log_x"):
                lo = max(factor_min, 1e-12)
                hi = max(factor_max, lo * (1.0 + 1e-9))
                edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
            else:
                edges = np.linspace(factor_min, factor_max, n_bins + 1)
            mid = (edges[:-1] + edges[1:]) / 2
            df_plot["_factor_bin"] = pd.cut(df[factor_col], bins=edges, labels=mid, include_lowest=True)
            bin_values = list(mid)

        for metric in metrics:
            if metric not in metrics_01:
                continue
            for prefix, method_label in method_plot_configs:
                col = f"{prefix}_{metric}"
                if col not in df.columns:
                    continue
                for b in bin_values:
                    if is_discrete:
                        mask = df_plot["_factor_bin"] == b
                    else:
                        bin_numeric = pd.to_numeric(df_plot["_factor_bin"], errors="coerce")
                        mask = np.abs(bin_numeric - float(b)) < 1e-10
                    vals = df.loc[mask, col].dropna().to_numpy()
                    if vals.size < 25:
                        continue
                    q25, q75 = np.quantile(vals, [0.25, 0.75])
                    variability_rows.append(
                        {
                            "factor": factor_name,
                            "factor_bin": float(b) if not is_discrete else int(b),
                            "metric": metric,
                            "method": method_label,
                            "prefix": prefix,
                            "n": int(vals.size),
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "iqr": float(q75 - q25),
                            "q05": float(np.quantile(vals, 0.05)),
                            "q95": float(np.quantile(vals, 0.95)),
                        }
                    )

    var_csv = os.path.join(output_dir, "diagnostics_factor_bin_variability.csv")
    var_df = pd.DataFrame(variability_rows)
    if not var_df.empty:
        var_df["std_p90"] = var_df.groupby(["factor", "metric", "method"])["std"].transform(
            lambda s: float(np.quantile(s, 0.9))
        )
        var_df["unpredictable"] = var_df["std"] >= var_df["std_p90"]
        var_df = var_df.sort_values(["unpredictable", "std"], ascending=[False, False])
    else:
        # Keep a stable schema even when there are not enough rows per factor/bin.
        var_df = pd.DataFrame(
            columns=[
                "factor",
                "factor_bin",
                "metric",
                "method",
                "prefix",
                "n",
                "mean",
                "std",
                "iqr",
                "q05",
                "q95",
                "std_p90",
                "unpredictable",
            ]
        )
    var_df.to_csv(var_csv, index=False)
    _log(f"  [diagnostics] wrote {var_csv}", verbose)


def create_method_comparison_plots(
    df: pd.DataFrame, output_dir: str, metrics: List[str], *, verbose: bool = True
) -> None:
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
    n_metrics = len(metrics)
    for i, metric in enumerate(metrics, start=1):
        _log(f"  [method plots] {i}/{n_metrics}: {metric} ...", verbose)
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
        out_path = os.path.join(output_dir, f"method_comparison_{metric}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        _log(f"  [method plots]     -> wrote {out_path}", verbose)


def create_factor_comparison_plots(df: pd.DataFrame, output_dir: str, *, verbose: bool = True) -> None:
    utils.safe_makedirs(output_dir)
    json_dir = os.path.join(os.path.dirname(output_dir), "json")
    utils.safe_makedirs(json_dir)

    method_configs = [
        ("divtree_lambda0", "λ=0", "#1f77b4", "o"),
        ("divtree_lambda2_region2", "λ=2", "#ff7f0e", "s"),
        ("divtree_lambda4_region2", "λ=4", "#2ca02c", "^"),
        ("divtree_lambda8_region2", "λ=8", "#17becf", "X"),
        ("twostep_tuned", "TwoStep (tuned)", "#9467bd", "D"),
        ("twostep_recall", "TwoStep (recall)", "#d62728", "v"),
    ]
    metrics = ["accuracy", "recall_region_2", "fnr_region_2", "precision_region_2"]
    metric_labels = {
        "accuracy": "Accuracy",
        "recall_region_2": "Recall (Region 2)",
        "fnr_region_2": "False Negative Rate (Region 2)",
        "precision_region_2": "Precision (Region 2)",
    }

    factors = {
        "noise": {
            "column": "noise",
            "label": "Outcome Noise",
            "xlabel": "Outcome Noise Std",
            "is_discrete": False,
            "log_x": True,
        },
        "data_size": {
            "column": "data_size",
            "label": "Data Size (n_users_train)",
            "xlabel": "Number of Training Observations",
            "is_discrete": False,
            "log_x": True,
        },
        "sparsity": {
            "column": "sparsity",
            "label": "Sparsity (k)",
            "xlabel": "Number of Categorical Variables",
            "is_discrete": True,
            "log_x": False,
        },
        "rareness": {
            "column": "rareness",
            "label": "Region-2 Share (rareness)",
            "xlabel": "Region-2 Share",
            "is_discrete": False,
            "log_x": False,
        },
        "intensity": {
            "column": "intensity",
            "label": "Treatment Effect Intensity",
            "xlabel": "Intensity",
            "is_discrete": False,
            "log_x": True,
        },
    }

    factor_list = [(k, v) for k, v in factors.items() if v["column"] in df.columns]
    total_factor_plots = sum(
        1
        for _, finfo in factor_list
        for _ in metrics
        if len(df[finfo["column"]].dropna().unique()) >= 2
    )
    done_factor = 0

    for factor_name, factor_info in factor_list:
        factor_col = factor_info["column"]
        if len(df[factor_col].dropna().unique()) < 2:
            continue

        is_discrete = factor_info["is_discrete"]
        if is_discrete:
            factor_values = sorted(df[factor_col].dropna().unique())
            df_plot = df.copy()
            df_plot["_factor_bin"] = df_plot[factor_col]
            bin_edges = None
            bin_midpoints = None
        else:
            n_bins = min(20, max(10, len(df[factor_col].dropna().unique()) // 10))
            factor_min = df[factor_col].min()
            factor_max = df[factor_col].max()
            if factor_info.get("log_x"):
                lo = max(float(factor_min), 1e-12)
                hi = max(float(factor_max), lo * (1.0 + 1e-9))
                bin_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
            else:
                bin_edges = np.linspace(factor_min, factor_max, n_bins + 1)
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            df_plot = df.copy()
            df_plot["_factor_bin"] = pd.cut(
                df_plot[factor_col],
                bins=bin_edges,
                labels=bin_midpoints,
                include_lowest=True,
            )
            factor_values = sorted(bin_midpoints)

        if len(factor_values) == 0:
            continue

        # JSON payload: compact chart data (shared x array + per-method series).
        factor_json = {
            "factor": factor_name,
            "xLabel": factor_info["xlabel"],
            "logX": bool(factor_info.get("log_x", False)),
            "x": [int(v) if is_discrete else _round5(v) for v in factor_values],
            # Provide bin edges only for continuous factors (for tooltips if needed).
            "binEdges": None if is_discrete else [_round5(x) for x in bin_edges],
            "metrics": {},
        }

        for metric in metrics:
            done_factor += 1
            _log(f"  [factor plots] {done_factor}/{total_factor_plots}: {factor_name} × {metric} ...", verbose)

            method_cols = [f"{prefix}_{metric}" for prefix, _, _, _ in method_configs]
            if any(c not in df.columns for c in method_cols):
                continue

            factor_data = []
            for factor_val in factor_values:
                if is_discrete:
                    mask = df_plot["_factor_bin"] == factor_val
                else:
                    bin_numeric = pd.to_numeric(df_plot["_factor_bin"], errors="coerce")
                    mask = np.abs(bin_numeric - float(factor_val)) < 1e-10
                subset = df_plot[mask]
                if len(subset) == 0:
                    continue

                row_data = {"factor_value": factor_val}
                all_valid = True
                for prefix, label, _, _ in method_configs:
                    col = f"{prefix}_{metric}"
                    values = subset[col].dropna().values
                    if len(values) == 0:
                        all_valid = False
                        break
                    row_data[f"{label}_mean"] = float(np.mean(values))
                    row_data[f"{label}_std"] = float(np.std(values))
                    row_data[f"{label}_n"] = int(len(values))
                if not all_valid:
                    continue
                factor_data.append(row_data)

            if len(factor_data) < 2:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))
            factor_vals = []
            for d in factor_data:
                val = d["factor_value"]
                factor_vals.append(float(val.mid) if hasattr(val, "mid") else float(val))

            all_means = []
            for prefix, label, color, marker in method_configs:
                mean_key = f"{label}_mean"
                means = [d[mean_key] for d in factor_data]
                all_means.extend(means)
                ax.plot(
                    factor_vals,
                    means,
                    label=label,
                    marker=marker,
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.85,
                    color=color,
                )

            ax.set_xlabel(factor_info["xlabel"], fontsize=12, fontweight="bold")
            ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight="bold")
            ax.set_title(f'{metric_labels[metric]} across {factor_info["label"]}', fontsize=14, fontweight="bold", pad=10)
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3, linestyle="--")
            if factor_info["log_x"]:
                ax.set_xscale("log")

            y_min, y_max = min(all_means), max(all_means)
            pad = max((y_max - y_min) * 0.1, 0.02)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"factor_comparison_{factor_name}_{metric}.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            _log(f"  [factor plots]     -> wrote {out_path}", verbose)

            # Attach compact metric payload to factor JSON (aligned to factor_data x bins).
            x_metric = [
                int(d["factor_value"]) if is_discrete else _round5(float(d["factor_value"]))
                for d in factor_data
            ]
            metric_payload = {"x": x_metric, "series": {}}
            for _, method_label, _, _ in method_configs:
                metric_payload["series"][method_label] = {
                    "n": [int(d[f"{method_label}_n"]) for d in factor_data],
                    "mean": [_round5(d[f"{method_label}_mean"]) for d in factor_data],
                    "std": [_round5(d[f"{method_label}_std"]) for d in factor_data],
                }
            factor_json["metrics"][metric] = metric_payload

        # Write one JSON per factor for HTML plotting.
        json_path = os.path.join(json_dir, f"factor_comparison_{factor_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(factor_json, f, indent=2)
        _log(f"  [factor json]      -> wrote {json_path}", verbose)


def create_summary_table(df: pd.DataFrame, output_dir: str, metrics: List[str], *, verbose: bool = True) -> None:
    utils.safe_makedirs(output_dir)
    rows = []
    summary_json = {"n": int(len(df)), "metrics": {}}
    for metric in metrics:
        row = {"Metric": metric}
        metric_payload = {"series": {}}
        for _, prefix, label in _method_configs():
            col = f"{prefix}_total_cpu_time" if metric == "cpu_time" and prefix.startswith("twostep_") else f"{prefix}_{metric}"
            if col in df.columns and df[col].notna().any():
                vals = df[col].dropna().values
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                row[f"{label} (mean)"] = mean
                row[f"{label} (std)"] = std
                metric_payload["series"][label] = {"n": int(len(vals)), "mean": _round5(mean), "std": _round5(std)}
            else:
                row[f"{label} (mean)"] = np.nan
                row[f"{label} (std)"] = np.nan
                metric_payload["series"][label] = {"n": 0, "mean": None, "std": None}
        rows.append(row)
        summary_json["metrics"][metric] = metric_payload
    out_csv = os.path.join(output_dir, "method_comparison_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _log(f"  [summary table] wrote {out_csv}", verbose)

    json_dir = os.path.join(os.path.dirname(output_dir), "json")
    utils.safe_makedirs(json_dir)
    out_json = os.path.join(json_dir, "method_comparison_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    _log(f"  [summary json]  wrote {out_json}", verbose)


def analyze_region_stratified(
    results_file: str,
    output_dir: str,
    metrics: Optional[List[str]] = None,
    *,
    verbose: bool = True,
) -> None:
    if metrics is None:
        metrics = [
            "accuracy",
            "f1_region_2",
            "fnr_region_2",
            "precision_region_2",
            "recall_region_2",
            "n_leaves",
            "runtime",
            "cpu_time",
        ]
    t0 = time.perf_counter()
    _log(f"Loading results: {results_file}", verbose)
    if os.path.isfile(results_file):
        size_mb = os.path.getsize(results_file) / (1024 * 1024)
        _log(f"  Pickle file size: {size_mb:.2f} MB", verbose)
    df = pd.read_pickle(results_file)
    _log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns in {time.perf_counter() - t0:.1f}s", verbose)

    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    _log(f"Method comparison plots -> {plots_dir}", verbose)
    t1 = time.perf_counter()
    create_method_comparison_plots(df, plots_dir, metrics, verbose=verbose)
    _log(f"Method comparison plots done in {time.perf_counter() - t1:.1f}s", verbose)
    _log(f"Factor comparison plots -> {plots_dir}", verbose)
    t2 = time.perf_counter()
    create_factor_comparison_plots(df, plots_dir, verbose=verbose)
    _log(f"Factor comparison plots done in {time.perf_counter() - t2:.1f}s", verbose)
    _log(f"Summary table -> {tables_dir}", verbose)
    t3 = time.perf_counter()
    create_summary_table(df, tables_dir, metrics, verbose=verbose)
    _log(f"Summary table done in {time.perf_counter() - t3:.1f}s", verbose)
    _log(f"Diagnostics tables -> {tables_dir}", verbose)
    t4 = time.perf_counter()
    create_diagnostics_tables(df, tables_dir, metrics, verbose=verbose)
    _log(f"Diagnostics tables done in {time.perf_counter() - t4:.1f}s", verbose)
    _log(f"Analysis complete in {time.perf_counter() - t0:.1f}s total.", verbose)


if __name__ == "__main__":
    base_dir = str(REPO_ROOT / "outputs" / "simulations" / "Comprehensive_simulation_v3")
    results_file = os.path.join(base_dir, "aggregated", "v3_lambda_twostep_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "v3_lambda_twostep_comparison", "analysis")
    verbose = "--quiet" not in sys.argv
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found.")
        sys.exit(1)
    print("Region-stratified v3 analysis (verbose). Pass --quiet to silence progress.", flush=True)
    analyze_region_stratified(results_file, output_dir, verbose=verbose)

