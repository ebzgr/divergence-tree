"""
Analysis script for lambda + TwoStep comparison simulation results.

Loads results from aggregated/lambda_twostep_comparison/ and creates:
- Method comparison plots (DivTree λ=0,1,2,4,6,8 + TwoStep tuned/fixed)
- Factor-wise comparison plots across data generation factors
- Summary tables (CSV)

Output saved to aggregated/lambda_twostep_comparison/analysis/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
import utils


# ============================================================================
# Statistical Testing Functions
# ============================================================================

def perform_paired_statistical_tests(
    df: pd.DataFrame,
    metric: str,
    lambda_values: List[int],
    method_prefixes: Dict[int, str],
    reference_lambdas: List[int] = [0, 2],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Perform paired statistical tests comparing reference lambdas to all others.
    
    Uses paired t-tests since the same simulations are used for all lambda values.
    This accounts for the different data generating settings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    metric : str
        Metric name to test.
    lambda_values : List[int]
        List of all lambda values tested.
    method_prefixes : Dict[int, str]
        Dictionary mapping lambda values to method prefixes.
    reference_lambdas : List[int], default=[0, 2]
        List of reference lambda values to compare against all others.
    
    Returns
    -------
    Dict[Tuple[int, int], Dict[str, float]]
        Dictionary mapping (reference_lambda, other_lambda) to test results.
        Results include: p_value, t_statistic, mean_diff, significant
    """
    results = {}
    
    for ref_lambda in reference_lambdas:
        if ref_lambda not in lambda_values:
            continue
        
        ref_prefix = method_prefixes[ref_lambda]
        ref_col = f"{ref_prefix}_{metric}"
        
        if ref_col not in df.columns:
            continue
        
        ref_values = df[ref_col].dropna()
        ref_indices = ref_values.index
        
        for other_lambda in lambda_values:
            if other_lambda == ref_lambda:
                continue
            
            other_prefix = method_prefixes[other_lambda]
            other_col = f"{other_prefix}_{metric}"
            
            if other_col not in df.columns:
                continue
            
            # Get paired values (same simulation indices)
            paired_mask = df.index.isin(ref_indices) & df[other_col].notna()
            paired_ref = df.loc[paired_mask, ref_col].values
            paired_other = df.loc[paired_mask, other_col].values
            
            if len(paired_ref) < 2:
                continue
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(paired_ref, paired_other)
            mean_diff = paired_other.mean() - paired_ref.mean()
            
            # Determine significance
            if p_value < 0.001:
                sig_level = "***"
            elif p_value < 0.01:
                sig_level = "**"
            elif p_value < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"
            
            results[(ref_lambda, other_lambda)] = {
                "p_value": p_value,
                "t_statistic": t_stat,
                "mean_diff": mean_diff,
                "significant": p_value < 0.05,
                "sig_level": sig_level,
                "n": len(paired_ref),
            }
    
    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def create_lambda_comparison_plots(
    df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Create plots showing performance metrics vs method (DivTree lambdas + TwoStep).
    
    Each plot shows one metric, with method on x-axis and metric value on y-axis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    output_dir : str
        Directory to save plots.
    metrics : Optional[List[str]], default=None
        List of metrics to plot. If None, plots all available metrics.
    """
    utils.safe_makedirs(output_dir)
    
    # Lambda values tested (exclude 3 and 10)
    lambda_values = [0, 1, 2, 4, 6, 8]
    
    # Method prefixes: DivTree lambdas + TwoStep
    method_configs = [
        (0, "divtree_lambda0", "DivTree λ=0"),
        (1, "divtree_lambda1_region2", "DivTree λ=1"),
        (2, "divtree_lambda2_region2", "DivTree λ=2"),
        (4, "divtree_lambda4_region2", "DivTree λ=4"),
        (6, "divtree_lambda6_region2", "DivTree λ=6"),
        (8, "divtree_lambda8_region2", "DivTree λ=8"),
        (None, "twostep_tuned", "TwoStep (tuned)"),
        (None, "twostep_fixed", "TwoStep (fixed)"),
    ]
    
    # Default metrics to plot if not specified
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
    
    # Metric labels for display
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_region_2": "F1 Score (Region 2)",
        "fnr_region_2": "False Negative Rate (Region 2)",
        "precision_region_2": "Precision (Region 2)",
        "recall_region_2": "Recall (Region 2)",
        "n_leaves": "Number of Leaves",
        "runtime": "Runtime (seconds)",
        "cpu_time": "CPU Time (seconds)",
    }
    
    print("=" * 80)
    print("CREATING METHOD COMPARISON PLOTS (DivTree + TwoStep)")
    print("=" * 80)
    
    for metric in metrics:
        print(f"\nProcessing metric: {metric_labels.get(metric, metric)}")
        
        # Collect data for each method
        method_data = []
        method_labels = []
        
        for _, prefix, label in method_configs:
            # TwoStep total CPU time = CausalForest + classification tree (for fair comparison)
            if metric == "cpu_time" and prefix in ("twostep_tuned", "twostep_fixed"):
                total_col = f"{prefix}_total_cpu_time"
                if total_col in df.columns:
                    values = df[total_col].dropna().values
                elif "twostep_causal_forest_cpu_time" in df.columns and f"{prefix}_cpu_time" in df.columns:
                    mask = df["twostep_causal_forest_cpu_time"].notna() & df[f"{prefix}_cpu_time"].notna()
                    values = (
                        df.loc[mask, "twostep_causal_forest_cpu_time"]
                        + df.loc[mask, f"{prefix}_cpu_time"]
                    ).values
                else:
                    print(f"  Warning: Total CPU time for {label} not found, skipping")
                    continue
            else:
                metric_col = f"{prefix}_{metric}"
                if metric_col not in df.columns:
                    print(f"  Warning: Column {metric_col} not found, skipping {label}")
                    continue
                values = df[metric_col].dropna().values
            
            if len(values) == 0:
                print(f"  Warning: No valid data for {label}")
                continue
            
            method_data.append({
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "n": len(values),
            })
            method_labels.append(label)
        
        if len(method_data) == 0:
            print(f"  Error: No data found for metric {metric}")
            continue
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(method_labels))
        means = [d["mean"] for d in method_data]
        stds = [d["std"] for d in method_data]
        
        colors = ["#2e86ab"] * 6 + ["#e94f37", "#44af69"]  # DivTree blue, TwoStep tuned red, fixed green
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        ax.set_xlabel("Method", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight="bold")
        ax.set_title(
            f"{metric_labels.get(metric, metric)} by Method",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        
        # Adjust y-axis limits for 0-1 metrics
        metrics_0_to_1 = ["accuracy", "f1_region_2", "precision_region_2", "recall_region_2", "fnr_region_2"]
        if metric in metrics_0_to_1:
            y_min = min(means) - 0.05
            y_max = max(means) + 0.05
            ax.set_ylim(max(0, y_min), min(1, y_max))
        elif min(means) >= 0:
            y_min = 0
            y_max = max(means) * 1.1
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        plot_filename = f"method_comparison_{metric}.png"
        plot_file = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved: {plot_file}")
        
        for label, data in zip(method_labels, method_data):
            print(f"    {label}: mean={data['mean']:.4f}±{data['std']:.4f}, n={data['n']}")


def create_summary_table(
    df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Create a summary table with mean and std for each metric and method.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    output_dir : str
        Directory to save table.
    metrics : Optional[List[str]], default=None
        List of metrics to include. If None, includes all available metrics.
    """
    utils.safe_makedirs(output_dir)
    
    method_configs = [
        (0, "divtree_lambda0", "DivTree λ=0"),
        (1, "divtree_lambda1_region2", "DivTree λ=1"),
        (2, "divtree_lambda2_region2", "DivTree λ=2"),
        (4, "divtree_lambda4_region2", "DivTree λ=4"),
        (6, "divtree_lambda6_region2", "DivTree λ=6"),
        (8, "divtree_lambda8_region2", "DivTree λ=8"),
        (None, "twostep_tuned", "TwoStep tuned"),
        (None, "twostep_fixed", "TwoStep fixed"),
    ]
    
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
    
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_region_2": "F1 Score (Region 2)",
        "fnr_region_2": "FNR (Region 2)",
        "precision_region_2": "Precision (Region 2)",
        "recall_region_2": "Recall (Region 2)",
        "n_leaves": "Number of Leaves",
        "runtime": "Runtime (seconds)",
        "cpu_time": "CPU Time (seconds)",
    }
    
    summary_data = []
    
    for metric in metrics:
        row = {"Metric": metric_labels.get(metric, metric)}
        
        for _, prefix, label in method_configs:
            if metric == "cpu_time" and prefix in ("twostep_tuned", "twostep_fixed"):
                metric_col = f"{prefix}_total_cpu_time"
            else:
                metric_col = f"{prefix}_{metric}"
            
            if metric_col in df.columns:
                values = df[metric_col].dropna().values
                if len(values) > 0:
                    row[f"{label} (mean)"] = f"{np.mean(values):.4f}"
                    row[f"{label} (std)"] = f"{np.std(values):.4f}"
                else:
                    row[f"{label} (mean)"] = "N/A"
                    row[f"{label} (std)"] = "N/A"
            else:
                row[f"{label} (mean)"] = "N/A"
                row[f"{label} (std)"] = "N/A"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    csv_file = os.path.join(output_dir, "method_comparison_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"\nSaved summary table: {csv_file}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))


# ============================================================================
# Factor-wise Comparison Functions
# ============================================================================

def create_factor_comparison_plots(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Create plots comparing DivTree (λ=0, λ=2, λ=4) and TwoStep across the 6 data generation factors.
    
    For each factor, generates 4 plots (accuracy, recall_region_2, fnr_region_2, precision_region_2)
    showing performance across different factor values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    output_dir : str
        Directory to save plots.
    """
    utils.safe_makedirs(output_dir)
    
    # Method prefixes: DivTree + TwoStep
    method_configs = [
        ("divtree_lambda0", "λ=0", "blue", "o"),
        ("divtree_lambda2_region2", "λ=2", "red", "s"),
        ("divtree_lambda4_region2", "λ=4", "green", "^"),
        ("twostep_tuned", "TwoStep (tuned)", "purple", "D"),
        ("twostep_fixed", "TwoStep (fixed)", "orange", "v"),
    ]
    
    # Metrics to plot
    metrics = ["accuracy", "recall_region_2", "fnr_region_2", "precision_region_2"]
    metric_labels = {
        "accuracy": "Accuracy",
        "recall_region_2": "Recall (Region 2)",
        "fnr_region_2": "False Negative Rate (Region 2)",
        "precision_region_2": "Precision (Region 2)",
    }
    
    # Factors to analyze
    factors = {
        "complexity": {
            "column": "complexity",
            "label": "Complexity (m_firm = m_user)",
            "xlabel": "Number of Activating Combinations",
            "is_discrete": True,  # Discrete: integer values
        },
        "noise": {
            "column": "noise",
            "label": "Noise (effect_noise_std)",
            "xlabel": "Effect Noise Standard Deviation",
            "is_discrete": False,  # Continuous: log-uniform distribution
        },
        "sparsity": {
            "column": "sparsity",
            "label": "Sparsity (k)",
            "xlabel": "Number of Categorical Variables",
            "is_discrete": True,  # Discrete: [1, 2, 3, 4, 5, 6]
        },
        "rareness": {
            "column": "rareness",
            "label": "Rareness (positive_ratio)",
            "xlabel": "Proportion in Activating Combinations",
            "is_discrete": False,  # Continuous: uniform [0.01, 0.99]
        },
        "covariance": {
            "column": "covariance",
            "label": "Covariance (similarity)",
            "xlabel": "Proportion of Shared Combinations",
            "is_discrete": False,  # Continuous: uniform [0.0, 1.0]
        },
        "data_size": {
            "column": "data_size",
            "label": "Data Size (n_users_train)",
            "xlabel": "Number of Training Observations",
            "is_discrete": False,  # Continuous: log-uniform distribution
        },
    }
    
    print("=" * 80)
    print("CREATING FACTOR-WISE COMPARISON PLOTS (DivTree + TwoStep)")
    print("=" * 80)
    
    for factor_name, factor_info in factors.items():
        print(f"\nProcessing factor: {factor_info['label']}")
        factor_col = factor_info["column"]
        
        if factor_col not in df.columns:
            print(f"  Warning: Column {factor_col} not found, skipping")
            continue
        
        # Handle discrete vs continuous factors
        is_discrete = factor_info.get("is_discrete", False)
        
        if is_discrete:
            # For discrete factors, use unique values
            factor_values = sorted(df[factor_col].dropna().unique())
            # Create a mapping for binning (each value maps to itself)
            df_plot = df.copy()
            df_plot["_factor_bin"] = df_plot[factor_col]
        else:
            # For continuous factors, create bins
            # Use 15-20 bins, but adjust based on data range
            n_bins = min(20, max(10, len(df[factor_col].dropna().unique()) // 10))
            factor_min = df[factor_col].min()
            factor_max = df[factor_col].max()
            
            # Create bins
            if factor_name == "noise" or factor_name == "data_size":
                # For noise and data_size (log-uniform), use log-spaced bins
                log_min = np.log10(max(factor_min, 1e-6))
                log_max = np.log10(factor_max)
                bin_edges = np.logspace(log_min, log_max, n_bins + 1)
            else:
                # For other continuous factors, use linear bins
                bin_edges = np.linspace(factor_min, factor_max, n_bins + 1)
            
            # Create bin labels (use midpoints)
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            df_plot = df.copy()
            df_plot["_factor_bin"] = pd.cut(
                df_plot[factor_col],
                bins=bin_edges,
                labels=bin_midpoints,
                include_lowest=True
            )
            factor_values = sorted(bin_midpoints)
        
        if len(factor_values) == 0:
            print(f"  Warning: No valid data for factor {factor_name}")
            continue
        
        # For each metric, create a plot
        for metric in metrics:
            print(f"  Creating plot for {metric_labels[metric]}")
            
            # Check all method columns exist
            method_cols = [f"{prefix}_{metric}" for prefix, _, _, _ in method_configs]
            if any(col not in df.columns for col in method_cols):
                print(f"    Warning: Metric columns not found, skipping")
                continue
            
            # Aggregate data by factor value
            factor_data = []
            
            for factor_val in factor_values:
                if is_discrete:
                    mask = df_plot["_factor_bin"] == factor_val
                else:
                    bin_numeric = pd.to_numeric(df_plot["_factor_bin"], errors='coerce')
                    mask = np.abs(bin_numeric - factor_val) < 1e-10
                subset = df_plot[mask]
                
                if len(subset) == 0:
                    continue
                
                # Get values for each method
                row_data = {"factor_value": factor_val}
                all_valid = True
                for prefix, label, _, _ in method_configs:
                    col = f"{prefix}_{metric}"
                    values = subset[col].dropna().values
                    if len(values) == 0:
                        all_valid = False
                        break
                    row_data[f"{label}_mean"] = np.mean(values)
                    row_data[f"{label}_std"] = np.std(values)
                    row_data[f"{label}_n"] = len(values)
                
                if not all_valid:
                    continue
                
                factor_data.append(row_data)
            
            if len(factor_data) == 0:
                print(f"    Warning: No data available for plotting")
                continue
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            factor_vals = []
            for d in factor_data:
                val = d["factor_value"]
                if hasattr(val, 'mid'):
                    factor_vals.append(val.mid)
                else:
                    factor_vals.append(float(val))
            
            # Plot each method
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
                    alpha=0.8,
                    color=color,
                )
            
            ax.set_xlabel(factor_info["xlabel"], fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
            ax.set_title(
                f'{metric_labels[metric]} across {factor_info["label"]}',
                fontsize=14,
                fontweight='bold',
                pad=10
            )
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if factor_name == "noise" or factor_name == "data_size":
                ax.set_xscale('log')
            
            if metric in ["accuracy", "recall_region_2", "fnr_region_2", "precision_region_2"]:
                y_min = min(all_means)
                y_max = max(all_means)
                y_range = y_max - y_min
                y_padding = max(y_range * 0.1, 0.02)
                ax.set_ylim(max(0, y_min - y_padding), min(1, y_max + y_padding))
            
            plt.tight_layout()
            
            plot_filename = f"factor_comparison_{factor_name}_{metric}.png"
            plot_file = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {plot_filename}")
            
            for d in factor_data[:3]:
                factor_val = d['factor_value']
                factor_str = f"{factor_val:.3f}" if isinstance(factor_val, float) else str(factor_val)
                parts = [f"{label}: {d[f'{label}_mean']:.4f}±{d[f'{label}_std']:.4f}" for _, label, _, _ in method_configs]
                print(f"      {factor_info['xlabel']}={factor_str}: " + ", ".join(parts))
            if len(factor_data) > 3:
                print(f"      ... and {len(factor_data) - 3} more factor values")


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_lambda_comparison(
    results_file: str,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Analyze lambda comparison simulation results.
    
    Creates plots showing performance metrics vs lambda values.
    
    Parameters
    ----------
    results_file : str
        Path to the results pickle file.
    output_dir : str
        Directory to save analysis outputs.
    metrics : Optional[List[str]], default=None
        List of metrics to plot. If None, plots all available metrics.
    """
    print("=" * 80)
    print("ANALYZING LAMBDA COMPARISON RESULTS")
    print("=" * 80)
    
    # Load results
    print(f"\nLoading results from: {results_file}")
    df = pd.read_pickle(results_file)
    print(f"Loaded {len(df)} simulations")
    
    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    utils.safe_makedirs(plots_dir)
    utils.safe_makedirs(tables_dir)
    
    # Create plots
    print("\n" + "=" * 80)
    create_lambda_comparison_plots(df, plots_dir, metrics=metrics)
    
    # Create factor-wise comparison plots
    print("\n" + "=" * 80)
    create_factor_comparison_plots(df, plots_dir)
    
    # Create summary table
    print("\n" + "=" * 80)
    create_summary_table(df, tables_dir, metrics=metrics)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Plots saved to: {plots_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    # Configuration - uses TwoStep folder (lambda_twostep_comparison)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(SCRIPT_DIR, "output")
    
    # Lambda + TwoStep comparison results (from lambda_comparison.py simulation)
    results_file = os.path.join(base_dir, "aggregated", "lambda_twostep_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "lambda_twostep_comparison", "analysis")
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        print("Run lambda_comparison.py first to generate simulation results.")
        sys.exit(1)
    
    print(f"Using TwoStep results: {results_file}")
    print(f"Output directory: {output_dir}\n")
    
    # Metrics to plot (None = plot all available)
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
    
    analyze_lambda_comparison(
        results_file=results_file,
        output_dir=output_dir,
        metrics=metrics,
    )

