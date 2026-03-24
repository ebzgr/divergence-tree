"""
Base simulation framework with shared utilities for method execution.

This module provides common functions for running lambda comparison simulations
using DivergenceTree.
"""

# Set environment variables to disable threading in joblib BEFORE any imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
import gc

try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
from binary_data_generator import generate_binary_comparison_data

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from divtree.tree import DivergenceTree
from divtree.forest import DivTreeForest
from divtree.tune import tune_with_optuna_holdout
from twostepdivtree.tree import TwoStepDivergenceTree

# Import local modules
import config
import utils
from metrics import compute_all_metrics


# ============================================================================
# Data Generation and Sampling
# ============================================================================

def sample_random_aspects(seed: int) -> Dict[str, Any]:
    """
    Randomly sample one value from each aspect range/list.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary with sampled aspect values.
    """
    rng = np.random.default_rng(seed)
    
    complexity = int(rng.uniform(config.COMPLEXITY_MIN, config.COMPLEXITY_MAX))
    noise = float(np.exp(rng.uniform(np.log(config.NOISE_MIN), np.log(config.NOISE_MAX))))
    data_size = int(np.exp(rng.uniform(np.log(config.DATA_SIZE_MIN), np.log(config.DATA_SIZE_MAX))))
    sparsity = int(rng.choice(config.SPARSITY_VALUES))
    rareness = float(rng.uniform(config.RARENESS_MIN, config.RARENESS_MAX))
    covariance = float(rng.uniform(config.COVARIANCE_MIN, config.COVARIANCE_MAX))
    
    return {
        "complexity": complexity,
        "noise": noise,
        "data_size": data_size,
        "sparsity": sparsity,
        "rareness": rareness,
        "covariance": covariance,
    }


def generate_data_with_params(
    complexity: int,
    noise: float,
    sparsity: int,
    rareness: float,
    covariance: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Dict[str, Any],
]:
    """Generate data with train/validation/test split (50/25/25 per config)."""
    n_categories = [60 // sparsity] * sparsity
    n_users_total = n_users_train + n_users_test
    
    (
        X_all, T_all, YF_all, YC_all, tauF_all, tauC_all,
        region_type_all, functional_form,
    ) = generate_binary_comparison_data(
        n_users=n_users_total,
        k=sparsity,
        n_categories=n_categories,
        m_firm=complexity,
        m_user=complexity,
        similarity=covariance,
        intensity=config.DEFAULT_INTENSITY,
        effect_noise_std=noise,
        firm_outcome_noise_std=0.0,
        user_outcome_noise_std=0.0,
        positive_ratio=rareness,
        random_seed=random_seed,
    )
    
    # Split into train (60%), validation (20%), test (20%)
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    n_train = int(config.TRAIN_FRAC * n_users_total)
    n_val = int(config.VAL_FRAC * n_users_total)
    n_test = n_users_total - n_train - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    
    X_train = X_all[train_indices]
    T_train = T_all[train_indices]
    YF_train = YF_all[train_indices]
    YC_train = YC_all[train_indices]
    region_type_train = region_type_all[train_indices]
    
    X_val = X_all[val_indices]
    T_val = T_all[val_indices]
    YF_val = YF_all[val_indices]
    YC_val = YC_all[val_indices]
    region_type_val = region_type_all[val_indices]
    
    X_test = X_all[test_indices]
    T_test = T_all[test_indices]
    YF_test = YF_all[test_indices]
    YC_test = YC_all[test_indices]
    region_type_test = region_type_all[test_indices]
    
    return (
        X_train, T_train, YF_train, YC_train, region_type_train,
        X_val, T_val, YF_val, YC_val, region_type_val,
        X_test, T_test, YF_test, YC_test, region_type_test,
        functional_form,
    )


# ============================================================================
# CPU Time Measurement
# ============================================================================

def get_cpu_time() -> float:
    """
    Return current process CPU time (user + system) in seconds.
    Uses resource.getrusage on Linux/Unix; falls back to time.process_time() on Windows.
    """
    if _HAS_RESOURCE:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    return time.process_time()


# ============================================================================
# Method Execution Functions
# ============================================================================

def run_divtree_method(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    YF_val: np.ndarray,
    YC_val: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    lambda_: float,
    regions_of_interest: Optional[List[int]],
    random_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run DivergenceTree with specified parameters.

    Uses validation set for hyperparameter tuning.
    Fits final model on train+val, evaluates on test test.

    Parameters
    ----------
    X_train, T_train, YF_train, YC_train : np.ndarray
        Training data.
    X_val, T_val, YF_val, YC_val : np.ndarray
        Validation data (for hyperparameter tuning).
    X_test : np.ndarray
        Test feature matrix.
    region_type_test : np.ndarray
        True region types for test set.
    lambda_ : float
        Lambda parameter for DivTree.
    regions_of_interest : Optional[List[int]]
        List of region IDs to focus on (e.g., [2] for region 2).
    random_seed : int
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary containing metrics, predictions, n_leaves, and runtime.
    """
    result = {}
    try:
        start_time = time.time()
        cpu_start = get_cpu_time()

        fixed_params = {
            "lambda_": lambda_,
            "n_quantiles": config.DIVTREE_FIXED_PARAMS.get("n_quantiles", 2),
            "eps_scale": config.DIVTREE_FIXED_PARAMS.get("eps_scale", 1e-8),
            "random_state": random_seed,
        }
        if regions_of_interest is not None:
            fixed_params["regions_of_interest"] = regions_of_interest

        best_params, _ = tune_with_optuna_holdout(
            X_train, T_train, YF_train, YC_train,
            X_val, T_val, YF_val, YC_val,
            fixed=fixed_params,
            search_space=config.DIVTREE_SEARCH_SPACE,
            n_trials=config.DIVTREE_N_TRIALS,
            random_state=random_seed,
            verbose=verbose,
        )

        # Fit final model on train+val, evaluate on test
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        T_trainval = np.concatenate([T_train, T_val], axis=0)
        YF_trainval = np.concatenate([YF_train, YF_val], axis=0)
        YC_trainval = np.concatenate([YC_train, YC_val], axis=0)

        divtree = DivergenceTree(**best_params)
        divtree.fit(X_trainval, T_trainval, YF_trainval, YC_trainval)

        region_type_pred_test = divtree.predict_region_type(X_test)
        region_type_pred_train = divtree.predict_region_type(X_train)
        
        leaf_effects = divtree.leaf_effects()
        n_leaves = len(leaf_effects["leaves"])
        
        metrics = compute_all_metrics(region_type_test, region_type_pred_test, method_name="")
        runtime = time.time() - start_time
        cpu_time = get_cpu_time() - cpu_start
        
        result = {
            "region_type_pred_train": region_type_pred_train,
            "region_type_pred_test": region_type_pred_test,
            "n_leaves": n_leaves,
            "runtime": runtime,
            "cpu_time": cpu_time,
        }
        result.update(metrics)
        
    except Exception:
        result = {
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
            "cpu_time": np.nan,
        }
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc"]:
            result[metric] = np.nan
    
    return result


def run_twostep_method(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    YF_val: np.ndarray,
    YC_val: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    n_leaves_divtree0: int,
    random_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run TwoStepDivergenceTree with four variants.

    Fits CausalForests once on train+val, then fits four classification trees:
    - twostep_tuned: tuned max_leaf_nodes, optimized on accuracy
    - twostep_recall: tuned max_leaf_nodes, optimized on recall of region 2
    - twostep_cap110: max_leaf_nodes = ceil(1.1 * n_leaves_divtree0)
    - twostep_cap150: max_leaf_nodes = ceil(1.5 * n_leaves_divtree0)

    Parameters
    ----------
    X_train, T_train, YF_train, YC_train : np.ndarray
        Training data.
    X_val, T_val, YF_val, YC_val : np.ndarray
        Validation data.
    X_test : np.ndarray
        Test feature matrix.
    region_type_test : np.ndarray
        True region types for test set.
    n_leaves_divtree0 : int
        Number of leaves from DivTree with lambda=0 (for cap110/cap150).
    random_seed : int
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary with twostep_tuned_*, twostep_recall_*, twostep_cap110_*, twostep_cap150_* metrics and CPU times.
    """
    result = {}
    try:
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        T_trainval = np.concatenate([T_train, T_val], axis=0)
        YF_trainval = np.concatenate([YF_train, YF_val], axis=0)
        YC_trainval = np.concatenate([YC_train, YC_val], axis=0)

        causal_params = {
            **config.TWOSTEP_CAUSAL_FOREST_PARAMS,
            "random_state": random_seed,
        }
        classification_params = {"random_state": random_seed}

        twostep = TwoStepDivergenceTree(
            causal_forest_params=causal_params,
            classification_tree_params=classification_params,
            causal_forest_tune_params={},
        )

        # Step 1: Fit CausalForests once. All four variants share this single first-step run
        # (tauF, tauC, region_types); only the second-step classification tree differs below.
        cpu_cf_start = get_cpu_time()
        twostep.fit(
            X_trainval, T_trainval, YF_trainval, YC_trainval,
            auto_tune_classification_tree=False,
            fit_classification_tree=False,
            verbose=verbose,
        )
        cpu_causal_forest = get_cpu_time() - cpu_cf_start

        # Step 2a: twostep_tuned - tune classification tree (accuracy, max_leaf_nodes)
        cpu_tuned_start = get_cpu_time()
        twostep._fit_classification_tree_step(
            X_trainval,
            auto_tune_classification_tree=True,
            classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS,
            classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TUNE_N_SPLITS,
            classification_tree_scoring="accuracy",
            tune_max_leaf_nodes=True,
            max_leaf_nodes_search_space=config.TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES,
            verbose=verbose,
        )
        pred_tuned_test = twostep.predict_region_type(X_test)
        pred_tuned_train = twostep.predict_region_type(X_train)
        n_leaves_tuned = int((twostep.classification_tree_.tree_.children_left == -1).sum())
        cpu_tuned = get_cpu_time() - cpu_tuned_start
        metrics_tuned = compute_all_metrics(region_type_test, pred_tuned_test, method_name="")

        # Step 2b: twostep_recall - tune classification tree (recall_region_2, max_leaf_nodes)
        cpu_recall_start = get_cpu_time()
        twostep._fit_classification_tree_step(
            X_trainval,
            auto_tune_classification_tree=True,
            classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS,
            classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TUNE_N_SPLITS,
            classification_tree_scoring="recall_region_2",
            tune_max_leaf_nodes=True,
            max_leaf_nodes_search_space=config.TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES,
            verbose=verbose,
        )
        pred_recall_test = twostep.predict_region_type(X_test)
        pred_recall_train = twostep.predict_region_type(X_train)
        n_leaves_recall = int((twostep.classification_tree_.tree_.children_left == -1).sum())
        cpu_recall = get_cpu_time() - cpu_recall_start
        metrics_recall = compute_all_metrics(region_type_test, pred_recall_test, method_name="")

        # Step 2c: twostep_cap110 - fixed max_leaf_nodes = ceil(1.1 * n_leaves_divtree0)
        max_leaf_cap110 = max(2, int(np.ceil(1.1 * n_leaves_divtree0)))
        cpu_cap110_start = get_cpu_time()
        twostep._fit_classification_tree_step(
            X_trainval,
            auto_tune_classification_tree=False,
            max_leaf_nodes=max_leaf_cap110,
            verbose=verbose,
        )
        pred_cap110_test = twostep.predict_region_type(X_test)
        pred_cap110_train = twostep.predict_region_type(X_train)
        n_leaves_cap110 = int((twostep.classification_tree_.tree_.children_left == -1).sum())
        cpu_cap110 = get_cpu_time() - cpu_cap110_start
        metrics_cap110 = compute_all_metrics(region_type_test, pred_cap110_test, method_name="")

        # Step 2d: twostep_cap150 - fixed max_leaf_nodes = ceil(1.5 * n_leaves_divtree0)
        max_leaf_cap150 = max(2, int(np.ceil(1.5 * n_leaves_divtree0)))
        cpu_cap150_start = get_cpu_time()
        twostep._fit_classification_tree_step(
            X_trainval,
            auto_tune_classification_tree=False,
            max_leaf_nodes=max_leaf_cap150,
            verbose=verbose,
        )
        pred_cap150_test = twostep.predict_region_type(X_test)
        pred_cap150_train = twostep.predict_region_type(X_train)
        n_leaves_cap150 = int((twostep.classification_tree_.tree_.children_left == -1).sum())
        cpu_cap150 = get_cpu_time() - cpu_cap150_start
        metrics_cap150 = compute_all_metrics(region_type_test, pred_cap150_test, method_name="")

        result = {
            "twostep_tuned_region_type_pred_train": pred_tuned_train,
            "twostep_tuned_region_type_pred_test": pred_tuned_test,
            "twostep_recall_region_type_pred_train": pred_recall_train,
            "twostep_recall_region_type_pred_test": pred_recall_test,
            "twostep_cap110_region_type_pred_train": pred_cap110_train,
            "twostep_cap110_region_type_pred_test": pred_cap110_test,
            "twostep_cap150_region_type_pred_train": pred_cap150_train,
            "twostep_cap150_region_type_pred_test": pred_cap150_test,
            "twostep_tuned_n_leaves": n_leaves_tuned,
            "twostep_recall_n_leaves": n_leaves_recall,
            "twostep_cap110_n_leaves": n_leaves_cap110,
            "twostep_cap150_n_leaves": n_leaves_cap150,
            "twostep_causal_forest_cpu_time": cpu_causal_forest,
            "twostep_tuned_cpu_time": cpu_tuned,
            "twostep_recall_cpu_time": cpu_recall,
            "twostep_cap110_cpu_time": cpu_cap110,
            "twostep_cap150_cpu_time": cpu_cap150,
            "twostep_tuned_total_cpu_time": cpu_causal_forest + cpu_tuned,
            "twostep_recall_total_cpu_time": cpu_causal_forest + cpu_recall,
            "twostep_cap110_total_cpu_time": cpu_causal_forest + cpu_cap110,
            "twostep_cap150_total_cpu_time": cpu_causal_forest + cpu_cap150,
        }
        for k, v in metrics_tuned.items():
            result[f"twostep_tuned_{k}"] = v
        for k, v in metrics_recall.items():
            result[f"twostep_recall_{k}"] = v
        for k, v in metrics_cap110.items():
            result[f"twostep_cap110_{k}"] = v
        for k, v in metrics_cap150.items():
            result[f"twostep_cap150_{k}"] = v

    except Exception:
        result = {
            "twostep_tuned_region_type_pred_train": None,
            "twostep_tuned_region_type_pred_test": None,
            "twostep_recall_region_type_pred_train": None,
            "twostep_recall_region_type_pred_test": None,
            "twostep_cap110_region_type_pred_train": None,
            "twostep_cap110_region_type_pred_test": None,
            "twostep_cap150_region_type_pred_train": None,
            "twostep_cap150_region_type_pred_test": None,
            "twostep_tuned_n_leaves": np.nan,
            "twostep_recall_n_leaves": np.nan,
            "twostep_cap110_n_leaves": np.nan,
            "twostep_cap150_n_leaves": np.nan,
            "twostep_causal_forest_cpu_time": np.nan,
            "twostep_tuned_cpu_time": np.nan,
            "twostep_recall_cpu_time": np.nan,
            "twostep_cap110_cpu_time": np.nan,
            "twostep_cap150_cpu_time": np.nan,
            "twostep_tuned_total_cpu_time": np.nan,
            "twostep_recall_total_cpu_time": np.nan,
            "twostep_cap110_total_cpu_time": np.nan,
            "twostep_cap150_total_cpu_time": np.nan,
        }
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap110", "twostep_cap150"]:
            for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                          "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                          "balanced_accuracy", "mcc"]:
                result[f"{prefix}_{metric}"] = np.nan

    return result


def run_divtree_forest_method(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    YF_val: np.ndarray,
    YC_val: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    lambda_: float,
    n_estimators: int,
    random_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run DivTreeForest (bagging of DivTree + classification tree).

    Tunes DivTree hyperparameters once, then fits a forest with those params.
    Simulation uses lambda=2, n_estimators=50 by default (from config).

    Parameters
    ----------
    X_train, T_train, YF_train, YC_train : np.ndarray
        Training data.
    X_val, T_val, YF_val, YC_val : np.ndarray
        Validation data (for tuning base DivTree params).
    X_test : np.ndarray
        Test feature matrix.
    region_type_test : np.ndarray
        True region types for test set.
    lambda_ : float
        Lambda for each DivTree in the forest.
    n_estimators : int
        Number of trees in the forest.
    random_seed : int
        Random seed.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary with divtree_forest_* metrics and CPU time.
    """
    result = {}
    try:
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        T_trainval = np.concatenate([T_train, T_val], axis=0)
        YF_trainval = np.concatenate([YF_train, YF_val], axis=0)
        YC_trainval = np.concatenate([YC_train, YC_val], axis=0)

        fixed_params = {
            "lambda_": lambda_,
            "n_quantiles": config.DIVTREE_FIXED_PARAMS.get("n_quantiles", 20),
            "eps_scale": config.DIVTREE_FIXED_PARAMS.get("eps_scale", 1e-8),
            "random_state": random_seed,
            "regions_of_interest": [2],
        }
        best_params, _ = tune_with_optuna_holdout(
            X_train, T_train, YF_train, YC_train,
            X_val, T_val, YF_val, YC_val,
            fixed=fixed_params,
            search_space=config.DIVTREE_SEARCH_SPACE,
            n_trials=config.DIVTREE_N_TRIALS,
            random_state=random_seed,
            verbose=verbose,
        )
        divtree_params = {
            k: v for k, v in best_params.items()
            if k in ("max_partitions", "min_improvement_ratio", "n_quantiles", "eps_scale")
        }

        cpu_start = get_cpu_time()
        forest = DivTreeForest(
            n_estimators=n_estimators,
            lambda_=lambda_,
            regions_of_interest=[2],
            max_samples=config.DIVTREE_FOREST_MAX_SAMPLES,
            max_features=config.DIVTREE_FOREST_MAX_FEATURES,
            random_state=random_seed,
            **divtree_params,
        )
        forest.fit(X_trainval, T_trainval, YF_trainval, YC_trainval)
        cpu_time = get_cpu_time() - cpu_start

        pred_train = forest.predict_region_type(X_train)
        pred_test = forest.predict_region_type(X_test)
        n_leaves = int((forest.classification_tree_.tree_.children_left == -1).sum())

        metrics = compute_all_metrics(region_type_test, pred_test, method_name="")
        result = {
            "divtree_forest_region_type_pred_train": pred_train,
            "divtree_forest_region_type_pred_test": pred_test,
            "divtree_forest_n_leaves": n_leaves,
            "divtree_forest_cpu_time": cpu_time,
        }
        for k, v in metrics.items():
            result[f"divtree_forest_{k}"] = v

    except Exception:
        result = {
            "divtree_forest_region_type_pred_train": None,
            "divtree_forest_region_type_pred_test": None,
            "divtree_forest_n_leaves": np.nan,
            "divtree_forest_cpu_time": np.nan,
        }
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc"]:
            result[f"divtree_forest_{metric}"] = np.nan

    return result


# ============================================================================
# Task Execution with Retry
# ============================================================================

def run_single_task_with_retry(
    task: Tuple[int, Dict[str, Any], int],
    base_dir: str,
    simulation_function: callable,
) -> Dict[str, Any]:
    """
    Run a single task with memory error retry logic.
    
    Parameters
    ----------
    task : Tuple[int, Dict[str, Any], int]
        Tuple of (simulation_id, aspect_values, random_seed).
    base_dir : str
        Base directory for saving results.
    simulation_function : callable
        Function to run the simulation. Should have signature:
        (simulation_id, aspect_values, random_seed, base_dir, verbose) -> Dict[str, Any]
    
    Returns
    -------
    dict
        Simulation results dictionary.
    """
    simulation_id, aspect_values, random_seed = task
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            return simulation_function(
                simulation_id=simulation_id,
                aspect_values=aspect_values,
                random_seed=random_seed,
                base_dir=base_dir,
                verbose=False,
            )
        except Exception as e:
            is_memory_error = (
                isinstance(e, MemoryError) or
                "MemoryError" in type(e).__name__ or
                "Unable to allocate" in str(e)
            )
            
            if is_memory_error and attempt < max_retries - 1:
                gc.collect()
                time.sleep(60)
                continue
            else:
                # Create error result - will be filled by simulation_function
                raise

