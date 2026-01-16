"""
Random space search simulation for comparing DivergenceTree methods with different lambda values.

DEPRECATED: This script is kept for backward compatibility. For new simulations, please use:
- `lambda_comparison.py` for lambda comparison (includes lambda=3, uses modular framework)
- `method_comparison.py` for method comparison (includes TwoStep methods)

This module implements a random sampling approach where aspect values are randomly
selected from the search space for each simulation, and compares 4 DivergenceTree methods:
1. DivergenceTree with lambda=0
2. DivergenceTree with lambda=1, regions_of_interest=[2]
3. DivergenceTree with lambda=2, regions_of_interest=[2]
4. DivergenceTree with lambda=4, regions_of_interest=[2]

Note: This script does NOT include lambda=3. Use `lambda_comparison.py` for a complete comparison.
"""

# Set environment variables to disable threading in joblib/econml BEFORE any imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# Suppress all warnings via environment variable (propagates to worker processes)
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
import gc

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)  # One level up from simulations/

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
from binary_data_generator import generate_binary_comparison_data

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna as tune_divtree

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
    """Run DivergenceTree with specified parameters."""
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


# ============================================================================
# Single Simulation Execution
# ============================================================================

def run_single_random_simulation(
    simulation_id: int,
    aspect_values: Dict[str, Any],
    random_seed: int,
    base_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single random simulation with all 4 methods."""
    result = {
        "simulation_id": simulation_id,
        **aspect_values,
    }
    
    try:
        data_size = aspect_values["data_size"]
        n_users_train = data_size
        n_users_test = data_size // 2
        
        (
            X_train, T_train, YF_train, YC_train, region_type_train,
            X_test, T_test, YF_test, YC_test, region_type_test,
            functional_form,
        ) = generate_data_with_params(
            complexity=aspect_values["complexity"],
            noise=aspect_values["noise"],
            sparsity=aspect_values["sparsity"],
            rareness=aspect_values["rareness"],
            covariance=aspect_values["covariance"],
            n_users_train=n_users_train,
            n_users_test=n_users_test,
            random_seed=random_seed,
        )
        
        # Method 1: DivergenceTree with lambda=0
        method1_result = run_divtree_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            lambda_=0.0,
            regions_of_interest=None,
            random_seed=random_seed,
            verbose=False,
        )
        for k, v in method1_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"divtree_lambda0_{k}"] = v
        
        # Method 2: DivergenceTree with lambda=1, regions_of_interest=[2]
        method2_result = run_divtree_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            lambda_=1.0,
            regions_of_interest=[2],
            random_seed=random_seed,
            verbose=False,
        )
        for k, v in method2_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"divtree_lambda1_region2_{k}"] = v
        
        # Method 3: DivergenceTree with lambda=2, regions_of_interest=[2]
        method3_result = run_divtree_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            lambda_=2.0,
            regions_of_interest=[2],
            random_seed=random_seed,
            verbose=False,
        )
        for k, v in method3_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"divtree_lambda2_region2_{k}"] = v
        
        # Method 4: DivergenceTree with lambda=4, regions_of_interest=[2]
        method4_result = run_divtree_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            lambda_=4.0,
            regions_of_interest=[2],
            random_seed=random_seed,
            verbose=False,
        )
        for k, v in method4_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"divtree_lambda4_region2_{k}"] = v
        
        # Save data with predictions
        data_dir = os.path.join(base_dir, "data", "region2tuning_test", f"simulation_{simulation_id:06d}")
        utils.safe_makedirs(data_dir)
        
        # Create training DataFrame
        train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        train_df["T"] = T_train
        train_df["YF"] = YF_train
        train_df["YC"] = YC_train
        train_df["region_type_true"] = region_type_train
        
        # Add predictions
        for method_result, col_name in [
            (method1_result, "divtree_lambda0_region_pred"),
            (method2_result, "divtree_lambda1_region2_region_pred"),
            (method3_result, "divtree_lambda2_region2_region_pred"),
            (method4_result, "divtree_lambda4_region2_region_pred"),
        ]:
            pred = method_result.get("region_type_pred_train")
            train_df[col_name] = pred if pred is not None else np.nan
        
        # Create test DataFrame
        test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        test_df["T"] = T_test
        test_df["YF"] = YF_test
        test_df["YC"] = YC_test
        test_df["region_type_true"] = region_type_test
        
        # Add predictions
        for method_result, col_name in [
            (method1_result, "divtree_lambda0_region_pred"),
            (method2_result, "divtree_lambda1_region2_region_pred"),
            (method3_result, "divtree_lambda2_region2_region_pred"),
            (method4_result, "divtree_lambda4_region2_region_pred"),
        ]:
            pred = method_result.get("region_type_pred_test")
            test_df[col_name] = pred if pred is not None else np.nan
        
        # Save DataFrames
        train_df.to_pickle(os.path.join(data_dir, "train_data.pkl"))
        test_df.to_pickle(os.path.join(data_dir, "test_data.pkl"))
        utils.save_data({"functional_form": functional_form}, 
                       os.path.join(data_dir, "functional_form.pickle"))
        
    except Exception:
        # Mark all metrics as NaN
        metrics = [
            "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
            "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
            "balanced_accuracy", "mcc", "rig", "n_leaves", "runtime"
        ]
        for method_prefix in ["divtree_lambda0", "divtree_lambda1_region2", 
                              "divtree_lambda2_region2", "divtree_lambda4_region2"]:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
    
    return result


def run_single_task_with_retry(
    task: Tuple[int, Dict[str, Any], int],
    base_dir: str,
) -> Dict[str, Any]:
    """Run a single task with memory error retry logic."""
    simulation_id, aspect_values, random_seed = task
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            return run_single_random_simulation(
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
                # Create error result
                result = {
                    "simulation_id": simulation_id,
                    **aspect_values,
                }
                metrics = [
                    "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                    "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                    "balanced_accuracy", "mcc", "rig", "n_leaves", "runtime"
                ]
                for method_prefix in ["divtree_lambda0", "divtree_lambda1_region2", 
                                      "divtree_lambda2_region2", "divtree_lambda4_region2"]:
                    for metric in metrics:
                        result[f"{method_prefix}_{metric}"] = np.nan
                return result


# ============================================================================
# Main Execution Function
# ============================================================================

def run_random_space_comparison(
    n_simulations: int,
    base_dir: str,
    n_jobs: int = -1,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
    batch_size: int = 100,
) -> None:
    """
    Run random space search comparison of 4 DivergenceTree methods:
    1. lambda=0
    2. lambda=1, regions_of_interest=[2]
    3. lambda=2, regions_of_interest=[2]
    4. lambda=4, regions_of_interest=[2]
     
    Parameters
    ----------
    n_simulations : int
        Number of simulations to run.
    base_dir : str
        Base directory for saving results.
    n_jobs : int
        Number of parallel jobs. -1 means use all available CPUs minus 1 
        (leaves 1 core free for system tasks to reduce context switching).
    base_random_seed : int
        Base random seed. Each simulation gets a unique seed.
    verbose : bool
        Whether to print progress.
    batch_size : int
        Number of simulations to run before saving incrementally.
    """
    # Calculate effective number of jobs (leave 1 core free for system tasks)
    if n_jobs <= 0:
        cpu_count = os.cpu_count() or 1
        effective_n_jobs = max(1, cpu_count - 1)  # Leave 1 core free
    else:
        effective_n_jobs = n_jobs
    
    if verbose:
        print("=" * 60)
        print("REGION 2 TUNING TEST - DIVERGENCE TREE COMPARISON")
        print("=" * 60)
        print(f"Number of simulations: {n_simulations}")
        print(f"Parallel jobs: {effective_n_jobs} (leaving 1 core free for system tasks)")
        print(f"Base random seed: {base_random_seed}")
        print(f"Batch size for incremental saving: {batch_size}")
    
    # Setup directories
    aggregated_dir = os.path.join(base_dir, "aggregated", "region2tuning_test")
    utils.safe_makedirs(aggregated_dir)
    results_file = os.path.join(aggregated_dir, "all_simulations_results.pkl")
    
    # Create list of all simulation tasks
    all_tasks = []
    start_id = 1
    for i in range(n_simulations):
        simulation_id = start_id + i
        random_seed = base_random_seed + simulation_id * 1000
        aspect_values = sample_random_aspects(random_seed)
        all_tasks.append((simulation_id, aspect_values, random_seed))
    
    if verbose:
        print(f"\nStarting parallel execution of {len(all_tasks)} simulations...")
        print(f"Simulation IDs: {start_id} to {start_id + n_simulations - 1}")
    
    # Process in batches for incremental saving
    all_results = []
    n_batches = (n_simulations + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_simulations)
        batch_tasks = all_tasks[start_idx:end_idx]
        
        if verbose:
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches}: simulations {start_idx + 1} to {end_idx} ({len(batch_tasks)} simulations)...")
        
        try:
            # Run batch in parallel
            batch_results = Parallel(n_jobs=effective_n_jobs, verbose=10 if verbose else 0)(
                delayed(run_single_task_with_retry)(task, base_dir) for task in batch_tasks
            )
            
            # Append batch results
            all_results.extend(batch_results)
            
            # Save incrementally (overwrites existing file)
            df = pd.DataFrame(all_results)
            df.to_pickle(results_file)
            if verbose:
                print(f"  Saved {len(all_results)}/{n_simulations} results to {results_file}")
        
        except KeyboardInterrupt:
            if verbose:
                print(f"\n\nInterrupted by user. Saving current progress...")
            
            # Save what we have so far
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_pickle(results_file)
                if verbose:
                    print(f"Progress saved to {results_file}")
                    print(f"Completed {len(all_results)}/{n_simulations} simulations before interruption")
            raise
    
    if verbose:
        print(f"\nCompleted! Results saved to {results_file}")
        print(f"Total simulations: {n_simulations}")


if __name__ == "__main__":
    base_dir = os.path.join(SCRIPT_DIR, "output")
    n_simulations = 10000
    n_jobs = -1
    
    run_random_space_comparison(
        n_simulations=n_simulations,
        base_dir=base_dir,
        n_jobs=n_jobs,
        verbose=True,
        batch_size=1000,
    )
