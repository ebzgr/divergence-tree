"""
Analysis script for lambda comparison simulation results.

Creates plots showing performance metrics vs lambda values.
Each plot shows one metric, with lambda on x-axis and metric value on y-axis.
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
    Create plots showing performance metrics vs lambda values.
    
    Each plot shows one metric, with lambda on x-axis and metric value on y-axis.
    
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
    
    # Lambda values tested
    lambda_values = [0, 1, 2, 3, 4, 6, 8, 10]
    
    # Method prefixes for each lambda
    method_prefixes = {
        0: "divtree_lambda0",
        1: "divtree_lambda1_region2",
        2: "divtree_lambda2_region2",
        3: "divtree_lambda3_region2",
        4: "divtree_lambda4_region2",
        6: "divtree_lambda6_region2",
        8: "divtree_lambda8_region2",
        10: "divtree_lambda10_region2",
    }
    
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
    }
    
    print("=" * 80)
    print("CREATING LAMBDA COMPARISON PLOTS")
    print("=" * 80)
    
    for metric in metrics:
        print(f"\nProcessing metric: {metric_labels.get(metric, metric)}")
        
        # Collect data for each lambda
        lambda_data = {}
        for lambda_val in lambda_values:
            prefix = method_prefixes[lambda_val]
            metric_col = f"{prefix}_{metric}"
            
            if metric_col not in df.columns:
                print(f"  Warning: Column {metric_col} not found, skipping lambda={lambda_val}")
                continue
            
            # Get valid values (non-NaN)
            values = df[metric_col].dropna().values
            
            if len(values) == 0:
                print(f"  Warning: No valid data for lambda={lambda_val}")
                continue
            
            lambda_data[lambda_val] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "n": len(values),
            }
        
        if len(lambda_data) == 0:
            print(f"  Error: No data found for metric {metric}")
            continue
        
        # Perform statistical tests
        stats_results = perform_paired_statistical_tests(
            df, metric, lambda_values, method_prefixes, reference_lambdas=[0, 2]
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        lambda_vals = sorted(lambda_data.keys())
        means = [lambda_data[lam]["mean"] for lam in lambda_vals]
        
        # Plot mean only
        ax.plot(
            lambda_vals,
            means,
            'o-',
            linewidth=2.5,
            markersize=10,
            color='blue',
            alpha=0.8,
            markerfacecolor='blue',
            markeredgecolor='darkblue',
            markeredgewidth=1.5,
        )
        
        # Get initial y-axis range for positioning significance markers
        y_min = min(means)
        y_max = max(means)
        y_range = y_max - y_min
        max_sig_y = y_max  # Track maximum y position needed for significance markers
        
        # Significance markers for lambda 0 vs others
        ref_lambda = 0
        if ref_lambda in lambda_vals:
            ref_idx = lambda_vals.index(ref_lambda)
            ref_y = means[ref_idx]
            
            for other_lambda in lambda_vals:
                if other_lambda == ref_lambda:
                    continue
                
                key = (ref_lambda, other_lambda)
                if key in stats_results:
                    result = stats_results[key]
                    if result["significant"]:
                        other_idx = lambda_vals.index(other_lambda)
                        other_y = means[other_idx]
                        
                        # Draw line and add star
                        max_y = max(ref_y, other_y)
                        line_y = max_y + y_range * 0.08  # 8% of range above max
                        max_sig_y = max(max_sig_y, line_y)
                        
                        ax.plot(
                            [ref_lambda, other_lambda],
                            [line_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        ax.plot(
                            [ref_lambda, ref_lambda],
                            [ref_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        ax.plot(
                            [other_lambda, other_lambda],
                            [other_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        
                        # Add significance star
                        mid_x = (ref_lambda + other_lambda) / 2
                        ax.text(
                            mid_x,
                            line_y,
                            result["sig_level"],
                            ha='center',
                            va='bottom',
                            fontsize=12,
                            fontweight='bold',
                            color='black',
                        )
        
        # Significance markers for lambda 2 vs others
        ref_lambda = 2
        if ref_lambda in lambda_vals:
            ref_idx = lambda_vals.index(ref_lambda)
            ref_y = means[ref_idx]
            
            for other_lambda in lambda_vals:
                if other_lambda == ref_lambda or other_lambda == 0:
                    continue  # Skip lambda 0 (already handled) and lambda 2 itself
                
                key = (ref_lambda, other_lambda)
                if key in stats_results:
                    result = stats_results[key]
                    if result["significant"]:
                        other_idx = lambda_vals.index(other_lambda)
                        other_y = means[other_idx]
                        
                        # Draw line and add star (positioned higher than lambda 0 comparisons)
                        max_y = max(ref_y, other_y)
                        line_y = max_y + y_range * 0.15  # 15% of range above max (higher than lambda 0)
                        max_sig_y = max(max_sig_y, line_y)
                        
                        ax.plot(
                            [ref_lambda, other_lambda],
                            [line_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        ax.plot(
                            [ref_lambda, ref_lambda],
                            [ref_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        ax.plot(
                            [other_lambda, other_lambda],
                            [other_y, line_y],
                            'k-',
                            linewidth=1,
                            alpha=0.6,
                        )
                        
                        # Add significance star
                        mid_x = (ref_lambda + other_lambda) / 2
                        ax.text(
                            mid_x,
                            line_y,
                            result["sig_level"],
                            ha='center',
                            va='bottom',
                            fontsize=12,
                            fontweight='bold',
                            color='black',
                        )
        
        # Customize plot
        ax.set_xlabel('Lambda Value', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
        ax.set_title(
            f'{metric_labels.get(metric, metric)} vs Lambda',
            fontsize=14,
            fontweight='bold',
            pad=10
        )
        ax.set_xticks(lambda_vals)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust y-axis limits for 0-1 metrics
        metrics_0_to_1 = ["accuracy", "f1_region_2", "precision_region_2", "recall_region_2"]
        if metric in metrics_0_to_1:
            # Add 5% padding on each side, but ensure we don't go below 0 or above 1
            y_padding = max(y_range * 0.05, 0.01)  # At least 1% padding
            y_lower = max(0, y_min - y_padding)
            # Extend upper limit to accommodate significance markers, but cap at 1
            y_upper = min(1, max(y_max + y_padding, max_sig_y + y_range * 0.02))
            ax.set_ylim(y_lower, y_upper)
        else:
            # For other metrics, add some padding but keep reasonable range
            y_padding = y_range * 0.1
            # Extend upper limit to accommodate significance markers
            y_upper = max(y_max + y_padding, max_sig_y + y_range * 0.05)
            ax.set_ylim(y_min - y_padding, y_upper)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"lambda_comparison_{metric}.png"
        plot_file = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_file}")
        
        # Print summary statistics
        print(f"  Summary statistics:")
        for lam in lambda_vals:
            data = lambda_data[lam]
            print(f"    λ={lam:2d}: mean={data['mean']:.4f}, n={data['n']}")
        
        # Print significance results
        print(f"  Significance tests (paired t-tests):")
        for (ref_lam, other_lam), result in stats_results.items():
            if result["significant"]:
                print(f"    λ={ref_lam} vs λ={other_lam}: {result['sig_level']} "
                      f"(p={result['p_value']:.4f}, n={result['n']})")


def create_summary_table(
    df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Create a summary table with mean and std for each metric and lambda.
    
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
    
    # Lambda values tested
    lambda_values = [0, 1, 2, 3, 4, 6, 8, 10]
    
    # Method prefixes for each lambda
    method_prefixes = {
        0: "divtree_lambda0",
        1: "divtree_lambda1_region2",
        2: "divtree_lambda2_region2",
        3: "divtree_lambda3_region2",
        4: "divtree_lambda4_region2",
        6: "divtree_lambda6_region2",
        8: "divtree_lambda8_region2",
        10: "divtree_lambda10_region2",
    }
    
    # Default metrics if not specified
    if metrics is None:
        metrics = [
            "accuracy",
            "f1_region_2",
            "fnr_region_2",
            "precision_region_2",
            "recall_region_2",
            "n_leaves",
            "runtime",
        ]
    
    # Metric labels
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_region_2": "F1 Score (Region 2)",
        "fnr_region_2": "FNR (Region 2)",
        "precision_region_2": "Precision (Region 2)",
        "recall_region_2": "Recall (Region 2)",
        "n_leaves": "Number of Leaves",
        "runtime": "Runtime (seconds)",
    }
    
    # Create summary table
    summary_data = []
    
    for metric in metrics:
        row = {"Metric": metric_labels.get(metric, metric)}
        
        for lambda_val in lambda_values:
            prefix = method_prefixes[lambda_val]
            metric_col = f"{prefix}_{metric}"
            
            if metric_col in df.columns:
                values = df[metric_col].dropna().values
                if len(values) > 0:
                    row[f"λ={lambda_val} (mean)"] = f"{np.mean(values):.4f}"
                    row[f"λ={lambda_val} (std)"] = f"{np.std(values):.4f}"
                else:
                    row[f"λ={lambda_val} (mean)"] = "N/A"
                    row[f"λ={lambda_val} (std)"] = "N/A"
            else:
                row[f"λ={lambda_val} (mean)"] = "N/A"
                row[f"λ={lambda_val} (std)"] = "N/A"
        
        summary_data.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, "lambda_comparison_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"\nSaved summary table: {csv_file}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))


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
    
    # Create summary table
    print("\n" + "=" * 80)
    create_summary_table(df, tables_dir, metrics=metrics)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Plots saved to: {plots_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(SCRIPT_DIR, "output")
    
    # Lambda comparison results
    results_file = os.path.join(base_dir, "aggregated", "lambda_comparison", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "lambda_comparison", "analysis")
    
    # Metrics to plot (None = plot all available)
    metrics = [
        "accuracy",
        "f1_region_2",
        "fnr_region_2",
        "precision_region_2",
        "recall_region_2",
        "n_leaves",
        "runtime",
    ]
    
    analyze_lambda_comparison(
        results_file=results_file,
        output_dir=output_dir,
        metrics=metrics,
    )

