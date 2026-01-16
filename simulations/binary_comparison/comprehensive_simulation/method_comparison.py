"""
Method comparison simulation: Compare 5 different methods.

This simulation compares:
1. DivTree with lambda=0
2. DivTree with lambda=2, regions_of_interest=[2]
3. TwoStepDivergenceTree with no max_leaf_nodes constraint
4. TwoStepDivergenceTree with max_leaf_nodes from divtree_lambda0
5. TwoStepDivergenceTree with max_leaf_nodes constraint + FNR region 2 scoring
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from joblib import Parallel, delayed

# Import base utilities
from simulation_base import (
    sample_random_aspects,
    generate_data_with_params,
    run_divtree_method,
    run_twostep_method,
    run_single_task_with_retry,
)

# Import local modules
import config
import utils

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Single Simulation Execution
# ============================================================================

def run_single_method_simulation(
    simulation_id: int,
    aspect_values: Dict[str, Any],
    random_seed: int,
    base_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single random simulation with all 5 methods.
    
    Parameters
    ----------
    simulation_id : int
        Unique simulation ID.
    aspect_values : Dict[str, Any]
        Dictionary of aspect values (complexity, noise, data_size, etc.).
    random_seed : int
        Random seed for reproducibility.
    base_dir : str
        Base directory for saving results.
    verbose : bool, default=False
        Whether to print progress.
    
    Returns
    -------
    dict
        Dictionary containing simulation results with all method metrics.
    """
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
        
        # Method 1: DivergenceTree with lambda=0 (run first to get n_leaves)
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
        
        # Extract n_leaves from method 1 for use in methods 4 and 5
        n_leaves_from_method1 = method1_result.get("n_leaves")
        if pd.isna(n_leaves_from_method1) or n_leaves_from_method1 is None:
            n_leaves_from_method1 = None
        else:
            n_leaves_from_method1 = int(n_leaves_from_method1)
        
        # Method 2: DivergenceTree with lambda=2, regions_of_interest=[2]
        method2_result = run_divtree_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            lambda_=2.0,
            regions_of_interest=[2],
            random_seed=random_seed,
            verbose=False,
        )
        for k, v in method2_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"divtree_lambda2_region2_{k}"] = v
        
        # Method 3: TwoStepDivergenceTree with no max_leaf_nodes constraint
        method3_result = run_twostep_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            random_seed=random_seed,
            max_leaf_nodes=None,
            classification_tree_scoring="accuracy",
            verbose=False,
        )
        for k, v in method3_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"twostep_unconstrained_{k}"] = v
        
        # Method 4: TwoStepDivergenceTree with max_leaf_nodes from method 1
        method4_result = run_twostep_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            random_seed=random_seed,
            max_leaf_nodes=n_leaves_from_method1,
            classification_tree_scoring="accuracy",
            verbose=False,
        )
        for k, v in method4_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"twostep_constrained_{k}"] = v
        
        # Method 5: TwoStepDivergenceTree with max_leaf_nodes constraint + FNR region 2 scoring
        method5_result = run_twostep_method(
            X_train, T_train, YF_train, YC_train,
            X_test, region_type_test,
            random_seed=random_seed,
            max_leaf_nodes=n_leaves_from_method1,
            classification_tree_scoring="fnr_region_2",
            verbose=False,
        )
        for k, v in method5_result.items():
            if k not in ["region_type_pred_train", "region_type_pred_test"]:
                result[f"twostep_constrained_fnr_{k}"] = v
        
        # Save data with predictions
        data_dir = os.path.join(base_dir, "data", "method_comparison", f"simulation_{simulation_id:06d}")
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
            (method2_result, "divtree_lambda2_region2_region_pred"),
            (method3_result, "twostep_unconstrained_region_pred"),
            (method4_result, "twostep_constrained_region_pred"),
            (method5_result, "twostep_constrained_fnr_region_pred"),
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
            (method2_result, "divtree_lambda2_region2_region_pred"),
            (method3_result, "twostep_unconstrained_region_pred"),
            (method4_result, "twostep_constrained_region_pred"),
            (method5_result, "twostep_constrained_fnr_region_pred"),
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
        for method_prefix in ["divtree_lambda0", "divtree_lambda2_region2",
                              "twostep_unconstrained", "twostep_constrained",
                              "twostep_constrained_fnr"]:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
    
    return result


# ============================================================================
# Main Execution Function
# ============================================================================

def run_method_comparison(
    n_simulations: int,
    base_dir: str,
    n_jobs: int = -1,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
    batch_size: int = 1000,
) -> None:
    """
    Run method comparison simulation: Compare 5 different methods.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations to run.
    base_dir : str
        Base directory for saving results.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available CPUs minus 1 
        (leaves 1 core free for system tasks to reduce context switching).
    base_random_seed : int, default=config.BASE_RANDOM_SEED
        Base random seed. Each simulation gets a unique seed.
    verbose : bool, default=True
        Whether to print progress.
    batch_size : int, default=1000
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
        print("METHOD COMPARISON - DIVTREE VS TWOSTEP")
        print("=" * 60)
        print(f"Number of simulations: {n_simulations}")
        print(f"Parallel jobs: {effective_n_jobs} (leaving 1 core free for system tasks)")
        print(f"Base random seed: {base_random_seed}")
        print(f"Batch size for incremental saving: {batch_size}")
        print("\nMethods:")
        print("  1. DivTree lambda=0")
        print("  2. DivTree lambda=2, regions_of_interest=[2]")
        print("  3. TwoStepDivergenceTree (unconstrained)")
        print("  4. TwoStepDivergenceTree (constrained to n_leaves from method 1)")
        print("  5. TwoStepDivergenceTree (constrained + FNR region 2 scoring)")
    
    # Setup directories
    aggregated_dir = os.path.join(base_dir, "aggregated", "method_comparison")
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
                delayed(run_single_task_with_retry)(
                    task, base_dir, run_single_method_simulation
                ) for task in batch_tasks
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
    
    run_method_comparison(
        n_simulations=n_simulations,
        base_dir=base_dir,
        n_jobs=n_jobs,
        verbose=True,
        batch_size=1000,
    )

