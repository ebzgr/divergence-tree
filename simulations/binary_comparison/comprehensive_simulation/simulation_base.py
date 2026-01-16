"""
Base simulation framework with shared utilities for method execution.

This module provides common functions and base classes for running simulations
that compare different methods (DivTree and TwoStepDivergenceTree).
"""

# Set environment variables to disable threading in joblib/econml BEFORE any imports
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
from multiprocessing import Manager

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
from binary_data_generator import generate_binary_comparison_data

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna as tune_divtree
from twostepdivtree.tree import TwoStepDivergenceTree
from sklearn.tree import DecisionTreeClassifier

# Import local modules
import config
import utils
from metrics import compute_all_metrics


# ============================================================================
# Global Semaphore for Limiting Concurrent CausalForest Builds
# ============================================================================

# Global semaphore and manager to limit concurrent CausalForest builds across all workers
# These will be initialized by the main process and shared via Manager
_causal_forest_semaphore = None
_semaphore_manager = None


def init_causal_forest_semaphore(max_concurrent: int = 2):
    """
    Initialize a shared semaphore to limit concurrent CausalForest builds.
    
    This should be called once before starting parallel execution to prevent
    memory issues from too many CausalForests being built simultaneously.
    
    The semaphore is created using multiprocessing.Manager() which allows
    it to be shared across worker processes spawned by joblib. The Manager
    server process must remain alive for the semaphore to work.
    
    Parameters
    ----------
    max_concurrent : int, default=2
        Maximum number of CausalForest builds that can run concurrently
        across all workers. Lower values use less memory but may be slower.
    
    Returns
    -------
    multiprocessing.synchronize.Semaphore
        The shared semaphore that can be used by worker processes.
    """
    global _causal_forest_semaphore, _semaphore_manager
    
    if _causal_forest_semaphore is None:
        _semaphore_manager = Manager()
        _causal_forest_semaphore = _semaphore_manager.Semaphore(max_concurrent)
        # Store manager reference to keep it alive
    
    return _causal_forest_semaphore


def get_causal_forest_semaphore():
    """
    Get the global CausalForest semaphore.
    
    Note: With multiprocessing spawn (Windows), worker processes won't have
    access to module-level variables set in the main process. However, the
    Manager server process runs independently, and the semaphore proxy
    should be accessible if properly initialized.
    
    For this to work, init_causal_forest_semaphore() must be called in the
    main process before starting parallel execution, and the Manager must
    remain alive (which it will as long as the main process is running).
    
    Returns
    -------
    multiprocessing.synchronize.Semaphore or None
        The semaphore if initialized, None otherwise.
    """
    return _causal_forest_semaphore


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
    Dict[str, Any],
]:
    """Generate data with specified parameters."""
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
    
    # Split into train and test sets
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    train_indices = indices[:n_users_train]
    test_indices = indices[n_users_train:]
    
    X_train = X_all[train_indices]
    T_train = T_all[train_indices]
    YF_train = YF_all[train_indices]
    YC_train = YC_all[train_indices]
    region_type_train = region_type_all[train_indices]
    
    X_test = X_all[test_indices]
    T_test = T_all[test_indices]
    YF_test = YF_all[test_indices]
    YC_test = YC_all[test_indices]
    region_type_test = region_type_all[test_indices]
    
    return (
        X_train, T_train, YF_train, YC_train, region_type_train,
        X_test, T_test, YF_test, YC_test, region_type_test,
        functional_form,
    )


# ============================================================================
# Method Execution Functions
# ============================================================================

def run_divtree_method(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    lambda_: float,
    regions_of_interest: Optional[List[int]],
    random_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run DivergenceTree with specified parameters.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    T_train : np.ndarray
        Training treatment indicator.
    YF_train : np.ndarray
        Training firm outcome.
    YC_train : np.ndarray
        Training consumer outcome.
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
        
        fixed_params = {
            "lambda_": lambda_,
            "n_quantiles": config.DIVTREE_FIXED_PARAMS.get("n_quantiles", 2),
            "eps_scale": config.DIVTREE_FIXED_PARAMS.get("eps_scale", 1e-8),
            "random_state": random_seed,
        }
        if regions_of_interest is not None:
            fixed_params["regions_of_interest"] = regions_of_interest
        
        best_params, _ = tune_divtree(
            X_train, T_train, YF_train, YC_train,
            fixed=fixed_params,
            search_space=config.DIVTREE_SEARCH_SPACE,
            n_trials=config.DIVTREE_N_TRIALS,
            n_splits=config.DIVTREE_N_SPLITS,
            random_state=random_seed,
            verbose=verbose,
        )
        
        divtree = DivergenceTree(**best_params)
        divtree.fit(X_train, T_train, YF_train, YC_train)
        
        region_type_pred_test = divtree.predict_region_type(X_test)
        region_type_pred_train = divtree.predict_region_type(X_train)
        
        leaf_effects = divtree.leaf_effects()
        n_leaves = len(leaf_effects["leaves"])
        
        metrics = compute_all_metrics(region_type_test, region_type_pred_test, method_name="")
        runtime = time.time() - start_time
        
        result = {
            "region_type_pred_train": region_type_pred_train,
            "region_type_pred_test": region_type_pred_test,
            "n_leaves": n_leaves,
            "runtime": runtime,
        }
        result.update(metrics)
        
    except Exception:
        result = {
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
        }
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc", "rig"]:
            result[metric] = np.nan
    
    return result


def run_twostep_method(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    random_seed: int,
    max_leaf_nodes: Optional[int] = None,
    classification_tree_scoring: str = "accuracy",
    verbose: bool = False,
    causal_forest_F=None,
    causal_forest_C=None,
    return_tree: bool = False,
) -> Dict[str, Any]:
    """
    Run TwoStepDivergenceTree with specified parameters.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    T_train : np.ndarray
        Training treatment indicator.
    YF_train : np.ndarray
        Training firm outcome.
    YC_train : np.ndarray
        Training consumer outcome.
    X_test : np.ndarray
        Test feature matrix.
    region_type_test : np.ndarray
        True region types for test set.
    random_seed : int
        Random seed for reproducibility.
    max_leaf_nodes : Optional[int], default=None
        Maximum number of leaf nodes for classification tree.
        If None, no constraint is applied.
    classification_tree_scoring : str, default="accuracy"
        Scoring function for classification tree tuning.
        Options: "accuracy", "fnr_region_1", "fnr_region_2", "fnr_region_3", "fnr_region_4"
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
        
        # Prepare classification tree parameters
        classification_tree_params = {
            "random_state": random_seed,
        }
        if max_leaf_nodes is not None:
            classification_tree_params["max_leaf_nodes"] = max_leaf_nodes
        
        # Create TwoStepDivergenceTree
        twostep_tree = TwoStepDivergenceTree(
            causal_forest_params={
                **config.TWOSTEP_CAUSAL_FOREST_PARAMS,
                "random_state": random_seed,
            },
            classification_tree_params=classification_tree_params,
            causal_forest_tune_params=config.TWOSTEP_CAUSAL_FOREST_TUNE_PARAMS,
        )
        
        # If pre-built CausalForests are provided, reuse them (skip step 1)
        if causal_forest_F is not None and causal_forest_C is not None:
            # Set pre-built forests and only train classification tree (step 2)
            twostep_tree.causal_forest_F_ = causal_forest_F
            twostep_tree.causal_forest_C_ = causal_forest_C
            twostep_tree._fit_classification_tree_step(
                X_train,
                auto_tune_classification_tree=config.TWOSTEP_AUTO_TUNE_CLASSIFICATION_TREE,
                classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_TRIALS,
                classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_SPLITS,
                classification_tree_scoring=classification_tree_scoring,
                verbose=verbose,
            )
        else:
            # Build CausalForests from scratch (normal flow)
            treatment_probability = np.full(len(X_train), 0.5)
            
            # Acquire semaphore before building CausalForests to limit concurrent builds
            semaphore = get_causal_forest_semaphore()
            if semaphore is not None:
                semaphore.acquire()
                try:
                    # Fit the tree
                    twostep_tree.fit(
                        X_train,
                        T_train,
                        YF_train,
                        YC_train,
                        auto_tune_classification_tree=config.TWOSTEP_AUTO_TUNE_CLASSIFICATION_TREE,
                        classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_TRIALS,
                        classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_SPLITS,
                        classification_tree_scoring=classification_tree_scoring,
                        treatment_probability=treatment_probability,
                        verbose=verbose,
                    )
                finally:
                    semaphore.release()
            else:
                # No semaphore initialized, run without limiting
                twostep_tree.fit(
                    X_train,
                    T_train,
                    YF_train,
                    YC_train,
                    auto_tune_classification_tree=config.TWOSTEP_AUTO_TUNE_CLASSIFICATION_TREE,
                    classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_TRIALS,
                    classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TREE_TUNE_N_SPLITS,
                    classification_tree_scoring=classification_tree_scoring,
                    treatment_probability=treatment_probability,
                    verbose=verbose,
                )
        
        # Predict on test set
        region_type_pred_test = twostep_tree.predict_region_type(X_test)
        region_type_pred_train = twostep_tree.predict_region_type(X_train)
        
        # Get number of leaves
        leaf_effects = twostep_tree.leaf_effects()
        n_leaves = len(leaf_effects["leaves"])
        
        # Compute metrics
        metrics = compute_all_metrics(region_type_test, region_type_pred_test, method_name="")
        runtime = time.time() - start_time
        
        result = {
            "region_type_pred_train": region_type_pred_train,
            "region_type_pred_test": region_type_pred_test,
            "n_leaves": n_leaves,
            "runtime": runtime,
        }
        result.update(metrics)
        
        # Optionally return the tree instance for extracting CausalForests
        if return_tree:
            result["_tree_instance"] = twostep_tree
        
    except Exception:
        result = {
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
        }
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc", "rig"]:
            result[metric] = np.nan
    
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

