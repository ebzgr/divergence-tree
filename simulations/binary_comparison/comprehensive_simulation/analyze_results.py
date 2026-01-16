"""
Unified analysis script for simulation results.

Supports both lambda_comparison and method_comparison simulation types.

Creates:
1. Plots comparing performance across aspects for key metrics (accuracy, FNR region 2, F1 region 2)
2. Statistical test matrices showing pairwise significance for each metric
3. Decision trees for each performance metric showing which method is best in different regions
4. LaTeX tables for statistical test results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from scipy import stats
import optuna

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
import utils


# ============================================================================
# Helper Functions
# ============================================================================

def determine_best_method_per_simulation(
    df: pd.DataFrame,
    metric: str,
    available_methods: List[str],
) -> pd.DataFrame:
    """
    Determine the best method for each simulation based on a metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    metric : str
        Metric name (e.g., "accuracy", "fnr_region_2").
    available_methods : List[str]
        List of available method names.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with simulation aspects and best method for each simulation.
    """
    aspect_cols = ["complexity", "noise", "data_size", "sparsity", "rareness", "covariance"]
    
    if not all(col in df.columns for col in aspect_cols):
        missing = [col for col in aspect_cols if col not in df.columns]
        raise ValueError(f"Missing aspect columns: {missing}")
    
    results = []
    
    for idx, row in df.iterrows():
        # Get aspect values
        aspect_dict = {col: row[col] for col in aspect_cols}
        
        # Get metric values for each method
        method_scores = {}
        for method in available_methods:
            col = f"{method}_{metric}"
            if col in df.columns:
                value = row[col]
                if pd.notna(value):
                    method_scores[method] = value
        
        # Find best method
        if len(method_scores) > 0:
            if metric == "fnr_region_2":
                # For FNR, lower is better
                best_method = min(method_scores.items(), key=lambda x: x[1])[0]
            else:
                # For other metrics (accuracy, f1_region_2), higher is better
                best_method = max(method_scores.items(), key=lambda x: x[1])[0]
            
            aspect_dict["best_method"] = best_method
            results.append(aspect_dict)
    
    return pd.DataFrame(results)


def tune_tree_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> Tuple[Dict, float]:
    """
    Tune decision tree hyperparameters using Optuna.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target labels.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target labels.
    n_trials : int, default=50
        Number of Optuna trials.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best validation accuracy.
    """
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 1, 5, step=1)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20, step=1)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )
        
        tree.fit(X_train, y_train)
        val_score = tree.score(X_val, y_val)
        return val_score
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_score = study.best_value
    
    return best_params, best_score


# ============================================================================
# Statistical Testing Functions
# ============================================================================

def perform_all_pairwise_statistical_tests(
    df: pd.DataFrame,
    available_methods: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    """
    Perform paired t-tests for all method pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    available_methods : List[str]
        List of all methods to compare.
    metrics : List[str]
        List of metrics to test.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with statistical test results for all pairs.
    """
    results = []
    
    for metric in metrics:
        # Compare all pairs of methods
        for i, method1 in enumerate(available_methods):
            method1_col = f"{method1}_{metric}"
            if method1_col not in df.columns:
                continue
            
            for method2 in available_methods[i+1:]:
                method2_col = f"{method2}_{metric}"
                if method2_col not in df.columns:
                    continue
                
                # Paired t-test (since we have the same simulations for both methods)
                valid_mask = df[[method1_col, method2_col]].notna().all(axis=1)
                if valid_mask.sum() < 2:
                    continue
                
                method1_paired = df.loc[valid_mask, method1_col].values
                method2_paired = df.loc[valid_mask, method2_col].values
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(method1_paired, method2_paired)
                
                # Calculate mean difference (method2 - method1)
                mean_diff = method2_paired.mean() - method1_paired.mean()
                
                # Determine if higher is better (for FNR, lower is better; for accuracy and F1, higher is better)
                if metric == "fnr_region_2":
                    improvement = mean_diff < 0  # Negative difference means method2 is better
                    effect_size = -mean_diff / method1_paired.std() if method1_paired.std() > 0 else 0
                else:  # accuracy, f1_region_2
                    improvement = mean_diff > 0  # Positive difference means method2 is better
                    effect_size = mean_diff / method1_paired.std() if method1_paired.std() > 0 else 0
                
                results.append({
                    "metric": metric,
                    "method1": method1,
                    "method2": method2,
                    "n": len(method1_paired),
                    "method1_mean": method1_paired.mean(),
                    "method1_std": method1_paired.std(),
                    "method2_mean": method2_paired.mean(),
                    "method2_std": method2_paired.std(),
                    "mean_difference": mean_diff,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "improvement": improvement,
                    "effect_size": effect_size,
                })
    
    return pd.DataFrame(results)


def generate_statistical_test_matrices(
    stats_df: pd.DataFrame,
    output_dir: str,
    available_methods: List[str],
    method_labels: Dict[str, str],
) -> None:
    """
    Generate statistical test matrices (heatmaps) for each metric showing p-values.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with pairwise statistical test results.
    output_dir : str
        Directory to save matrices.
    available_methods : List[str]
        List of all methods.
    method_labels : Dict[str, str]
        Dictionary mapping method names to labels.
    """
    utils.safe_makedirs(output_dir)
    
    metric_labels = {
        "accuracy": "Accuracy",
        "fnr_region_2": "FNR (Region 2)",
        "f1_region_2": "F1 (Region 2)",
    }
    
    for metric in stats_df['metric'].unique():
        metric_df = stats_df[stats_df['metric'] == metric].copy()
        
        # Create matrix of p-values
        n_methods = len(available_methods)
        p_value_matrix = np.full((n_methods, n_methods), np.nan)
        mean_diff_matrix = np.full((n_methods, n_methods), np.nan)
        
        for _, row in metric_df.iterrows():
            method1 = row['method1']
            method2 = row['method2']
            
            i1 = available_methods.index(method1)
            i2 = available_methods.index(method2)
            
            p_value_matrix[i1, i2] = row['p_value']
            p_value_matrix[i2, i1] = row['p_value']  # Symmetric
            mean_diff_matrix[i1, i2] = row['mean_difference']
            mean_diff_matrix[i2, i1] = -row['mean_difference']  # Symmetric (negated)
        
        # Set diagonal to 1.0 (method compared to itself)
        np.fill_diagonal(p_value_matrix, 1.0)
        np.fill_diagonal(mean_diff_matrix, 0.0)
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # P-value matrix
        im1 = ax1.imshow(p_value_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.05)
        ax1.set_xticks(range(n_methods))
        ax1.set_yticks(range(n_methods))
        ax1.set_xticklabels([method_labels.get(m, m) for m in available_methods], rotation=45, ha='right')
        ax1.set_yticklabels([method_labels.get(m, m) for m in available_methods])
        ax1.set_title(f"P-values: {metric_labels.get(metric, metric)}")
        plt.colorbar(im1, ax=ax1, label='p-value')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if not np.isnan(p_value_matrix[i, j]):
                    if p_value_matrix[i, j] < 0.001:
                        text = "***"
                    elif p_value_matrix[i, j] < 0.01:
                        text = "**"
                    elif p_value_matrix[i, j] < 0.05:
                        text = "*"
                    else:
                        text = f"{p_value_matrix[i, j]:.3f}"
                    ax1.text(j, i, text, ha='center', va='center', fontsize=8)
        
        # Mean difference matrix
        im2 = ax2.imshow(mean_diff_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_xticks(range(n_methods))
        ax2.set_yticks(range(n_methods))
        ax2.set_xticklabels([method_labels.get(m, m) for m in available_methods], rotation=45, ha='right')
        ax2.set_yticklabels([method_labels.get(m, m) for m in available_methods])
        ax2.set_title(f"Mean Differences: {metric_labels.get(metric, metric)}")
        plt.colorbar(im2, ax=ax2, label='Mean Difference')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if not np.isnan(mean_diff_matrix[i, j]):
                    text = f"{mean_diff_matrix[i, j]:.3f}"
                    ax2.text(j, i, text, ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        matrix_file = os.path.join(output_dir, f"{metric}_statistical_test_matrix.png")
        plt.savefig(matrix_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved statistical test matrix: {matrix_file}")


def generate_latex_table(
    stats_df: pd.DataFrame,
    output_file: str,
    method_labels: Dict[str, str],
) -> None:
    """
    Generate LaTeX table from pairwise statistical test results.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with pairwise statistical test results.
    output_file : str
        Path to save LaTeX table.
    method_labels : Dict[str, str]
        Dictionary mapping method names to labels.
    """
    metric_labels = {
        "accuracy": "Accuracy",
        "fnr_region_2": "FNR (Region 2)",
        "f1_region_2": "F1 (Region 2)",
    }
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Pairwise statistical comparison of methods}\n")
        f.write("\\label{tab:pairwise_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Method 1 & Method 2 & Mean Diff & $t$-stat & $p$-value \\\\\n")
        f.write("\\midrule\n")
        
        for metric in stats_df['metric'].unique():
            metric_df = stats_df[stats_df['metric'] == metric].copy()
            
            for _, row in metric_df.iterrows():
                method1_label = method_labels.get(row['method1'], row['method1'])
                method2_label = method_labels.get(row['method2'], row['method2'])
                diff = row['mean_difference']
                t_stat = row['t_statistic']
                p_val = row['p_value']
                
                # Format significance
                if p_val < 0.001:
                    p_str = "$<0.001$"
                    sig_mark = "***"
                elif p_val < 0.01:
                    p_str = f"${p_val:.3f}$"
                    sig_mark = "**"
                elif p_val < 0.05:
                    p_str = f"${p_val:.3f}$"
                    sig_mark = "*"
                else:
                    p_str = f"${p_val:.3f}$"
                    sig_mark = ""
                
                f.write(
                    f"{metric_labels.get(metric, metric)} & "
                    f"{method1_label} & "
                    f"{method2_label}{sig_mark} & "
                    f"${diff:+.4f}$ & "
                    f"${t_stat:.3f}$ & "
                    f"{p_str} \\\\\n"
                )
            
            if metric != stats_df['metric'].unique()[-1]:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\footnotesize\n")
        f.write("\\item * $p<0.05$, ** $p<0.01$, *** $p<0.001$\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to: {output_file}")


# ============================================================================
# Plotting Functions
# ============================================================================

def create_aspect_comparison_plots(
    df: pd.DataFrame,
    output_dir: str,
    available_methods: List[str],
    method_labels: Dict[str, str],
    method_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Create plots comparing performance across aspects for key metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    output_dir : str
        Directory to save plots.
    available_methods : List[str]
        List of available method names.
    method_labels : Dict[str, str]
        Dictionary mapping method names to labels.
    method_colors : Optional[Dict[str, str]], default=None
        Dictionary mapping method names to colors. If None, uses default colors.
    """
    utils.safe_makedirs(output_dir)
    
    # Key metrics to plot
    metrics = ["accuracy", "fnr_region_2", "f1_region_2"]
    metric_labels = {
        "accuracy": "Accuracy",
        "fnr_region_2": "False Negative Rate (Region 2)",
        "f1_region_2": "F1 Score (Region 2)",
    }
    
    # Aspects to plot (including data_size)
    aspects = ["complexity", "noise", "data_size", "sparsity", "rareness", "covariance"]
    aspect_labels = {
        "complexity": "Complexity",
        "noise": "Noise",
        "data_size": "Sample Size (Training)",
        "sparsity": "Sparsity",
        "rareness": "Rareness",
        "covariance": "Covariance",
    }
    
    # Default colors for methods
    if method_colors is None:
        method_colors = {
            "divtree_lambda0": "blue",
            "divtree_lambda1_region2": "green",
            "divtree_lambda2_region2": "orange",
            "divtree_lambda3_region2": "purple",
            "divtree_lambda4_region2": "red",
            "twostep_unconstrained": "cyan",
            "twostep_constrained": "magenta",
            "twostep_constrained_fnr": "brown",
        }
    
    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, aspect in enumerate(aspects):
            ax = axes[idx]
            
            # Collect data for all methods
            for method in available_methods:
                metric_col = f"{method}_{metric}"
                if metric_col not in df.columns:
                    continue
                
                # Create bins for the aspect (for better visualization)
                if aspect == "data_size":
                    # Use log bins for data_size
                    bins = np.logspace(
                        np.log10(df[aspect].min()),
                        np.log10(df[aspect].max()),
                        num=20
                    )
                    df_binned = df.copy()
                    df_binned[f"{aspect}_bin"] = pd.cut(df[aspect], bins=bins, include_lowest=True)
                    bin_centers = [interval.mid for interval in df_binned[f"{aspect}_bin"].cat.categories]
                    
                    means = []
                    stds = []
                    min_samples_per_bin = 3  # Require at least 3 samples per bin for reliable std
                    
                    for bin_center, bin_interval in zip(bin_centers, df_binned[f"{aspect}_bin"].cat.categories):
                        mask = df_binned[f"{aspect}_bin"] == bin_interval
                        values = df_binned.loc[mask, metric_col].dropna()
                        if len(values) >= min_samples_per_bin:
                            mean_val = values.mean()
                            std_val = values.std()
                            # Handle edge cases: NaN std (shouldn't happen with n>=2, but be safe)
                            if np.isnan(std_val) or std_val == 0:
                                # If std is NaN or 0, use a small fraction of the mean as std
                                # This ensures error bars are visible
                                std_val = abs(mean_val) * 0.01 if mean_val != 0 else 0.01
                            means.append(mean_val)
                            stds.append(std_val)
                        else:
                            # Skip bins with too few samples
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    valid_mask = ~np.isnan(means)
                    if np.any(valid_mask):
                        ax.plot(
                            np.array(bin_centers)[valid_mask],
                            np.array(means)[valid_mask],
                            'o-',
                            label=method_labels.get(method, method),
                            color=method_colors.get(method, 'gray'),
                            linewidth=2,
                            markersize=6,
                            alpha=0.7,
                        )
                        # Add error bars (handle potential NaN in stds)
                        stds_array = np.array(stds)[valid_mask]
                        means_array = np.array(means)[valid_mask]
                        # Replace any remaining NaN stds with small fraction of mean
                        stds_array = np.where(
                            np.isnan(stds_array) | (stds_array == 0),
                            np.abs(means_array) * 0.01,
                            stds_array
                        )
                        ax.fill_between(
                            np.array(bin_centers)[valid_mask],
                            means_array - stds_array,
                            means_array + stds_array,
                            alpha=0.2,
                            color=method_colors.get(method, 'gray'),
                        )
                elif aspect == "noise":
                    # Use log bins for noise (since it's log-uniformly distributed)
                    bins = np.logspace(
                        np.log10(max(df[aspect].min(), 0.001)),  # Avoid log(0)
                        np.log10(df[aspect].max()),
                        num=20
                    )
                    df_binned = df.copy()
                    df_binned[f"{aspect}_bin"] = pd.cut(df[aspect], bins=bins, include_lowest=True)
                    bin_centers = [interval.mid for interval in df_binned[f"{aspect}_bin"].cat.categories]
                    
                    means = []
                    stds = []
                    min_samples_per_bin = 3  # Require at least 3 samples per bin for reliable std
                    
                    for bin_center, bin_interval in zip(bin_centers, df_binned[f"{aspect}_bin"].cat.categories):
                        mask = df_binned[f"{aspect}_bin"] == bin_interval
                        values = df_binned.loc[mask, metric_col].dropna()
                        if len(values) >= min_samples_per_bin:
                            mean_val = values.mean()
                            std_val = values.std()
                            # Handle edge cases: NaN std (shouldn't happen with n>=2, but be safe)
                            if np.isnan(std_val) or std_val == 0:
                                # If std is NaN or 0, use a small fraction of the mean as std
                                # This ensures error bars are visible
                                std_val = abs(mean_val) * 0.01 if mean_val != 0 else 0.01
                            means.append(mean_val)
                            stds.append(std_val)
                        else:
                            # Skip bins with too few samples
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    valid_mask = ~np.isnan(means)
                    if np.any(valid_mask):
                        ax.plot(
                            np.array(bin_centers)[valid_mask],
                            np.array(means)[valid_mask],
                            'o-',
                            label=method_labels.get(method, method),
                            color=method_colors.get(method, 'gray'),
                            linewidth=2,
                            markersize=6,
                            alpha=0.7,
                        )
                        # Add error bars (handle potential NaN in stds)
                        stds_array = np.array(stds)[valid_mask]
                        means_array = np.array(means)[valid_mask]
                        # Replace any remaining NaN stds with small fraction of mean
                        stds_array = np.where(
                            np.isnan(stds_array) | (stds_array == 0),
                            np.abs(means_array) * 0.01,
                            stds_array
                        )
                        ax.fill_between(
                            np.array(bin_centers)[valid_mask],
                            means_array - stds_array,
                            means_array + stds_array,
                            alpha=0.2,
                            color=method_colors.get(method, 'gray'),
                        )
                    # Set log scale for noise axis
                    ax.set_xscale('log')
                else:
                    # For other aspects, use linear bins
                    bins = np.linspace(df[aspect].min(), df[aspect].max(), num=20)
                    df_binned = df.copy()
                    df_binned[f"{aspect}_bin"] = pd.cut(df[aspect], bins=bins, include_lowest=True)
                    bin_centers = [interval.mid for interval in df_binned[f"{aspect}_bin"].cat.categories]
                    
                    means = []
                    stds = []
                    min_samples_per_bin = 3  # Require at least 3 samples per bin for reliable std
                    
                    for bin_center, bin_interval in zip(bin_centers, df_binned[f"{aspect}_bin"].cat.categories):
                        mask = df_binned[f"{aspect}_bin"] == bin_interval
                        values = df_binned.loc[mask, metric_col].dropna()
                        if len(values) >= min_samples_per_bin:
                            mean_val = values.mean()
                            std_val = values.std()
                            # Handle edge cases: NaN std (shouldn't happen with n>=2, but be safe)
                            if np.isnan(std_val) or std_val == 0:
                                # If std is NaN or 0, use a small fraction of the mean as std
                                # This ensures error bars are visible
                                std_val = abs(mean_val) * 0.01 if mean_val != 0 else 0.01
                            means.append(mean_val)
                            stds.append(std_val)
                        else:
                            # Skip bins with too few samples
                            means.append(np.nan)
                            stds.append(np.nan)
                    
                    valid_mask = ~np.isnan(means)
                    if np.any(valid_mask):
                        ax.plot(
                            np.array(bin_centers)[valid_mask],
                            np.array(means)[valid_mask],
                            'o-',
                            label=method_labels.get(method, method),
                            color=method_colors.get(method, 'gray'),
                            linewidth=2,
                            markersize=6,
                            alpha=0.7,
                        )
                        # Add error bars (handle potential NaN in stds)
                        stds_array = np.array(stds)[valid_mask]
                        means_array = np.array(means)[valid_mask]
                        # Replace any remaining NaN stds with small fraction of mean
                        stds_array = np.where(
                            np.isnan(stds_array) | (stds_array == 0),
                            np.abs(means_array) * 0.01,
                            stds_array
                        )
                        ax.fill_between(
                            np.array(bin_centers)[valid_mask],
                            means_array - stds_array,
                            means_array + stds_array,
                            alpha=0.2,
                            color=method_colors.get(method, 'gray'),
                        )
            
            ax.set_xlabel(aspect_labels.get(aspect, aspect), fontsize=12)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
            ax.set_title(f"{aspect_labels.get(aspect, aspect)}: {metric_labels.get(metric, metric)}", fontsize=14)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            if aspect == "data_size" or aspect == "noise":
                ax.set_xscale('log')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"{metric}_across_aspects.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_file}")


# ============================================================================
# Decision Tree Functions
# ============================================================================

def build_method_chooser_trees(
    df: pd.DataFrame,
    output_dir: str,
    available_methods: List[str],
    method_labels: Dict[str, str],
    use_tuning: bool = True,
    n_trials: int = 50,
) -> None:
    """
    Build decision trees for each metric showing which method is best.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with simulation results.
    output_dir : str
        Directory to save tree visualizations.
    available_methods : List[str]
        List of available method names.
    method_labels : Dict[str, str]
        Dictionary mapping method names to labels.
    use_tuning : bool, default=True
        Whether to use hyperparameter tuning.
    n_trials : int, default=50
        Number of Optuna trials for tuning.
    """
    utils.safe_makedirs(output_dir)
    
    # Metrics to build trees for
    metrics = ["accuracy", "fnr_region_2", "f1_region_2"]
    metric_labels = {
        "accuracy": "Accuracy",
        "fnr_region_2": "FNR (Region 2)",
        "f1_region_2": "F1 (Region 2)",
    }
    
    aspect_cols = ["complexity", "noise", "data_size", "sparsity", "rareness", "covariance"]
    aspect_labels = {
        "complexity": "Complexity",
        "noise": "Noise",
        "data_size": "Sample Size",
        "sparsity": "Sparsity",
        "rareness": "Rareness",
        "covariance": "Covariance",
    }
    
    print("BUILDING METHOD CHOOSER DECISION TREES")
    print("=" * 80)
    
    for metric in metrics:
        print(f"\n--- Building tree for metric: {metric_labels.get(metric, metric)} ---")
        
        try:
            # Determine best method for each simulation
            scenario_best_df = determine_best_method_per_simulation(
                df, metric, available_methods
            )
            
            if len(scenario_best_df) == 0:
                print(f"  No simulations found for {metric}")
                continue
            
            # Prepare features (aspects) and target (best method)
            X_all = scenario_best_df[aspect_cols].values
            y_all = scenario_best_df["best_method"].values
            
            # Check if we have enough data
            if len(X_all) < 10:
                print(f"  Not enough simulations ({len(X_all)}) for {metric}. Need at least 10.")
                continue
            
            # Split into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            
            # Tune hyperparameters or use defaults
            if use_tuning and len(X_train) >= 20:
                print(f"  Tuning hyperparameters...")
                best_params, best_score = tune_tree_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                print(f"  Best validation accuracy: {best_score:.3f}")
            else:
                best_params = {
                    "max_depth": 5,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5,
                    "criterion": "gini",
                }
                best_score = None
            
            # Train final tree on all data
            tree = DecisionTreeClassifier(
                **best_params,
                random_state=42,
            )
            tree.fit(X_all, y_all)
            
            # Get feature names with labels
            feature_names = [aspect_labels.get(col, col) for col in aspect_cols]
            
            # Get class names (method labels)
            class_names = [method_labels.get(method, method) for method in tree.classes_]
            
            # Plot tree
            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(
                tree,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax,
            )
            ax.set_title(
                f"Method Chooser Tree: {metric_labels.get(metric, metric)}\n"
                f"(Validation Accuracy: {best_score:.3f if best_score else 'N/A'})",
                fontsize=14,
                pad=20,
            )
            
            plt.tight_layout()
            tree_file = os.path.join(output_dir, f"method_chooser_{metric}_tree.png")
            plt.savefig(tree_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved tree: {tree_file}")
            
            # Print feature importances
            importances = tree.feature_importances_
            print(f"  Feature importances:")
            for col, importance in zip(aspect_cols, importances):
                print(f"    {aspect_labels.get(col, col)}: {importance:.3f}")
        
        except Exception as e:
            print(f"  Error building tree for {metric}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_results(
    results_file: str,
    output_dir: str,
    simulation_type: str = "auto",
    use_tuning: bool = True,
    n_trials: int = 50,
) -> None:
    """
    Analyze results from simulation comparison.
    
    Parameters
    ----------
    results_file : str
        Path to the results pickle file.
    output_dir : str
        Directory to save analysis outputs.
    simulation_type : str, default="auto"
        Type of simulation: "lambda_comparison", "method_comparison", "unified_comparison", or "auto" (auto-detect).
    use_tuning : bool, default=True
        Whether to use hyperparameter tuning for decision trees.
    n_trials : int, default=50
        Number of Optuna trials for tuning.
    """
    print("=" * 80)
    print("ANALYZING SIMULATION RESULTS")
    print("=" * 80)
    
    # Load results
    print(f"\nLoading results from: {results_file}")
    df = pd.read_pickle(results_file)
    print(f"Loaded {len(df)} simulations")
    
    # Auto-detect methods from column names
    all_possible_methods = [
        "divtree_lambda0",
        "divtree_lambda1_region2",
        "divtree_lambda2_region2",
        "divtree_lambda3_region2",
        "divtree_lambda4_region2",
        "twostep_unconstrained",
        "twostep_constrained",
        "twostep_constrained_fnr",
    ]
    
    available_methods = []
    for method in all_possible_methods:
        if any(f"{method}_{m}" in df.columns for m in ["accuracy", "f1_region_2", "n_leaves"]):
            available_methods.append(method)
    
    if len(available_methods) == 0:
        print("ERROR: No methods found in data")
        return
    
    print(f"Found {len(available_methods)} methods: {available_methods}")
    
    # Auto-detect simulation type if needed
    if simulation_type == "auto":
        if len(available_methods) == 8:
            # All 8 methods present = unified_comparison
            simulation_type = "unified_comparison"
        elif any("twostep" in m for m in available_methods):
            simulation_type = "method_comparison"
        else:
            simulation_type = "lambda_comparison"
        print(f"Auto-detected simulation type: {simulation_type}")
    
    # Method labels based on simulation type
    if simulation_type == "unified_comparison":
        method_labels = {
            "divtree_lambda0": "DivTree (λ=0)",
            "divtree_lambda1_region2": "DivTree (λ=1, R2)",
            "divtree_lambda2_region2": "DivTree (λ=2, R2)",
            "divtree_lambda3_region2": "DivTree (λ=3, R2)",
            "divtree_lambda4_region2": "DivTree (λ=4, R2)",
            "twostep_unconstrained": "TwoStep (unconstrained)",
            "twostep_constrained": "TwoStep (constrained)",
            "twostep_constrained_fnr": "TwoStep (constrained + FNR)",
        }
    elif simulation_type == "lambda_comparison":
        method_labels = {
            "divtree_lambda0": "DivTree (λ=0)",
            "divtree_lambda1_region2": "DivTree (λ=1, R2)",
            "divtree_lambda2_region2": "DivTree (λ=2, R2)",
            "divtree_lambda3_region2": "DivTree (λ=3, R2)",
            "divtree_lambda4_region2": "DivTree (λ=4, R2)",
        }
    else:  # method_comparison
        method_labels = {
            "divtree_lambda0": "DivTree (λ=0)",
            "divtree_lambda2_region2": "DivTree (λ=2, R2)",
            "twostep_unconstrained": "TwoStep (unconstrained)",
            "twostep_constrained": "TwoStep (constrained)",
            "twostep_constrained_fnr": "TwoStep (constrained + FNR)",
        }
    
    # Update method colors for plotting
    method_colors = {
        "divtree_lambda0": "blue",
        "divtree_lambda1_region2": "green",
        "divtree_lambda2_region2": "orange",
        "divtree_lambda3_region2": "purple",
        "divtree_lambda4_region2": "red",
        "twostep_unconstrained": "cyan",
        "twostep_constrained": "magenta",
        "twostep_constrained_fnr": "brown",
    }
    # Filter colors to only include available methods
    method_colors = {k: v for k, v in method_colors.items() if k in available_methods}
    
    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    trees_dir = os.path.join(output_dir, "trees")
    tables_dir = os.path.join(output_dir, "tables")
    utils.safe_makedirs(plots_dir)
    utils.safe_makedirs(trees_dir)
    utils.safe_makedirs(tables_dir)
    
    # Perform pairwise statistical tests
    print("\n" + "=" * 80)
    print("PERFORMING PAIRWISE STATISTICAL TESTS")
    print("=" * 80)
    metrics_for_tests = ["accuracy", "fnr_region_2", "f1_region_2"]
    stats_df = perform_all_pairwise_statistical_tests(
        df, available_methods, metrics_for_tests
    )
    
    if len(stats_df) > 0:
        # Save CSV
        csv_file = os.path.join(tables_dir, "statistical_tests.csv")
        stats_df.to_csv(csv_file, index=False)
        print(f"Saved statistical test results to: {csv_file}")
        
        # Generate LaTeX table
        latex_file = os.path.join(tables_dir, "statistical_tests.tex")
        generate_latex_table(stats_df, latex_file, method_labels)
        
        # Generate statistical test matrices
        print("\n" + "=" * 80)
        print("GENERATING STATISTICAL TEST MATRICES")
        print("=" * 80)
        generate_statistical_test_matrices(
            stats_df, plots_dir, available_methods, method_labels
        )
    
    # Create aspect comparison plots
    print("\n" + "=" * 80)
    print("CREATING ASPECT COMPARISON PLOTS")
    print("=" * 80)
    create_aspect_comparison_plots(df, plots_dir, available_methods, method_labels, method_colors)
    
    # Build decision trees
    print("\n" + "=" * 80)
    print("BUILDING METHOD CHOOSER TREES")
    print("=" * 80)
    build_method_chooser_trees(
        df, trees_dir, available_methods, method_labels, use_tuning=use_tuning, n_trials=n_trials
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Plots saved to: {plots_dir}")
    print(f"Trees saved to: {trees_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(SCRIPT_DIR, "output")
    
    # Example usage for lambda_comparison
    # results_file = os.path.join(base_dir, "aggregated", "lambda_comparison", "all_simulations_results.pkl")
    # output_dir = os.path.join(base_dir, "aggregated", "lambda_comparison", "analysis")
    # simulation_type = "lambda_comparison"
    
    # Example usage for method_comparison
    # results_file = os.path.join(base_dir, "aggregated", "method_comparison", "all_simulations_results.pkl")
    # output_dir = os.path.join(base_dir, "aggregated", "method_comparison", "analysis")
    # simulation_type = "method_comparison"
    
    # Example usage for unified_comparison
    # results_file = os.path.join(base_dir, "aggregated", "unified_comparison", "all_simulations_results.pkl")
    # output_dir = os.path.join(base_dir, "aggregated", "unified_comparison", "analysis")
    # simulation_type = "unified_comparison"
    
    # Default: analyze region2tuning_test (backward compatibility)
    results_file = os.path.join(base_dir, "aggregated", "region2tuning_test", "all_simulations_results.pkl")
    output_dir = os.path.join(base_dir, "aggregated", "region2tuning_test", "analysis")
    simulation_type = "auto"  # Auto-detect from data
    
    analyze_results(
        results_file=results_file,
        output_dir=output_dir,
        simulation_type=simulation_type,
        use_tuning=True,
        n_trials=50,
    )

