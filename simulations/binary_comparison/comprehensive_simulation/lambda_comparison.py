"""
Lambda comparison simulation: Compare DivTree with lambda values 0, 1, 2, 3, 4, 6, 8, 10.

This simulation compares 8 DivergenceTree methods:
1. DivergenceTree with lambda=0
2. DivergenceTree with lambda=1, regions_of_interest=[2]
3. DivergenceTree with lambda=2, regions_of_interest=[2]
4. DivergenceTree with lambda=3, regions_of_interest=[2]
5. DivergenceTree with lambda=4, regions_of_interest=[2]
6. DivergenceTree with lambda=6, regions_of_interest=[2]
7. DivergenceTree with lambda=8, regions_of_interest=[2]
8. DivergenceTree with lambda=10, regions_of_interest=[2]

The generated data is saved for later use in method comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from joblib import Parallel, delayed

# Import base utilities
from simulation_base import (
    sample_random_aspects,
    generate_data_with_params,
    run_divtree_method,
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

def run_single_lambda_simulation(
    simulation_id: int,
    aspect_values: Dict[str, Any],
    random_seed: int,
    base_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single random simulation with all 8 lambda methods.
    
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
    
    # Lambda values to test
    lambda_values = [0, 1, 2, 3, 4, 6, 8, 10]
    
    try:
        data_size = aspect_values["data_size"]
        n_users_train = data_size
        n_users_test = data_size // 2
        
        # Check if data already exists
        data_dir = os.path.join(base_dir, "data", "lambda_comparison", f"simulation_{simulation_id:06d}")
        train_data_file = os.path.join(data_dir, "train_data.pkl")
        test_data_file = os.path.join(data_dir, "test_data.pkl")
        functional_form_file = os.path.join(data_dir, "functional_form.pickle")
        
        if os.path.exists(train_data_file) and os.path.exists(test_data_file) and os.path.exists(functional_form_file):
            # Load existing data
            if verbose:
                print(f"Loading existing data for simulation {simulation_id}")
            train_df = pd.read_pickle(train_data_file)
            test_df = pd.read_pickle(test_data_file)
            functional_form = utils.load_data(functional_form_file)["functional_form"]
            
            # Extract data from DataFrames
            X_train = train_df[[col for col in train_df.columns if col.startswith("feature_")]].values
            T_train = train_df["T"].values
            YF_train = train_df["YF"].values
            YC_train = train_df["YC"].values
            region_type_train = train_df["region_type_true"].values
            
            X_test = test_df[[col for col in test_df.columns if col.startswith("feature_")]].values
            T_test = test_df["T"].values
            YF_test = test_df["YF"].values
            YC_test = test_df["YC"].values
            region_type_test = test_df["region_type_true"].values
        else:
            # Generate new data
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
            
            # Save data for later reuse
            utils.safe_makedirs(data_dir)
            train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
            train_df["T"] = T_train
            train_df["YF"] = YF_train
            train_df["YC"] = YC_train
            train_df["region_type_true"] = region_type_train
            
            test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
            test_df["T"] = T_test
            test_df["YF"] = YF_test
            test_df["YC"] = YC_test
            test_df["region_type_true"] = region_type_test
            
            train_df.to_pickle(train_data_file)
            test_df.to_pickle(test_data_file)
            utils.save_data({"functional_form": functional_form}, functional_form_file)
        
        # Run all lambda methods
        method_results = []
        for lambda_val in lambda_values:
            method_result = run_divtree_method(
                X_train, T_train, YF_train, YC_train,
                X_test, region_type_test,
                lambda_=float(lambda_val),
                regions_of_interest=None if lambda_val == 0 else [2],
                random_seed=random_seed,
                verbose=False,
            )
            method_results.append((lambda_val, method_result))
            
            # Store results with appropriate prefix
            if lambda_val == 0:
                prefix = "divtree_lambda0"
            else:
                prefix = f"divtree_lambda{lambda_val}_region2"
            
            for k, v in method_result.items():
                if k not in ["region_type_pred_train", "region_type_pred_test"]:
                    result[f"{prefix}_{k}"] = v
        
        # Update DataFrames with predictions and save
        for lambda_val, method_result in method_results:
            if lambda_val == 0:
                col_name = "divtree_lambda0_region_pred"
            else:
                col_name = f"divtree_lambda{lambda_val}_region2_region_pred"
            
            pred_train = method_result.get("region_type_pred_train")
            pred_test = method_result.get("region_type_pred_test")
            
            if pred_train is not None:
                train_df[col_name] = pred_train
            else:
                train_df[col_name] = np.nan
            
            if pred_test is not None:
                test_df[col_name] = pred_test
            else:
                test_df[col_name] = np.nan
        
        # Save updated DataFrames
        train_df.to_pickle(train_data_file)
        test_df.to_pickle(test_data_file)
        
    except Exception:
        # Mark all metrics as NaN
        metrics = [
            "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
            "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
            "precision_region_2", "recall_region_2",
            "balanced_accuracy", "mcc", "rig", "n_leaves", "runtime"
        ]
        method_prefixes = ["divtree_lambda0"]
        for lambda_val in [1, 2, 3, 4, 6, 8, 10]:
            method_prefixes.append(f"divtree_lambda{lambda_val}_region2")
        
        for method_prefix in method_prefixes:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
    
    return result


# ============================================================================
# Main Execution Function
# ============================================================================

def run_lambda_comparison(
    n_simulations: int,
    base_dir: str,
    n_jobs: int = -1,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
    batch_size: int = 1000,
) -> None:
    """
    Run lambda comparison simulation: Compare DivTree with lambda values 0, 1, 2, 3, 4, 6, 8, 10.
    
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
        print("LAMBDA COMPARISON - DIVERGENCE TREE COMPARISON")
        print("=" * 60)
        print(f"Number of simulations: {n_simulations}")
        print(f"Parallel jobs: {effective_n_jobs} (leaving 1 core free for system tasks)")
        print(f"Base random seed: {base_random_seed}")
        print(f"Batch size for incremental saving: {batch_size}")
        print("\nMethods (8 total):")
        print("  1. DivTree lambda=0")
        print("  2. DivTree lambda=1, regions_of_interest=[2]")
        print("  3. DivTree lambda=2, regions_of_interest=[2]")
        print("  4. DivTree lambda=3, regions_of_interest=[2]")
        print("  5. DivTree lambda=4, regions_of_interest=[2]")
        print("  6. DivTree lambda=6, regions_of_interest=[2]")
        print("  7. DivTree lambda=8, regions_of_interest=[2]")
        print("  8. DivTree lambda=10, regions_of_interest=[2]")
        print("\nNote: Generated data is saved for later use in method comparison.")
    
    # Setup directories
    aggregated_dir = os.path.join(base_dir, "aggregated", "lambda_comparison")
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
                    task, base_dir, run_single_lambda_simulation
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
    
    run_lambda_comparison(
        n_simulations=n_simulations,
        base_dir=base_dir,
        n_jobs=n_jobs,
        verbose=True,
        batch_size=1000,
    )

