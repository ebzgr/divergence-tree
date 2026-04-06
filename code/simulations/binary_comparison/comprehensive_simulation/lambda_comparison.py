"""
Lambda and TwoStep comparison simulation.

Compares DivTree with lambda values 0, 1, 2, 4, 6, 8 and TwoStep with four variants:
- TwoStep tuned: classification tree tuned on accuracy (max_leaf_nodes tuned)
- TwoStep recall: classification tree tuned on recall of region 2
- TwoStep cap110: max_leaf_nodes = ceil(1.1 * n_leaves from DivTree lambda=0)
- TwoStep cap150: max_leaf_nodes = ceil(1.5 * n_leaves from DivTree lambda=0)

CausalForests are fit once per simulation and reused for all TwoStep variants.
Results are saved to aggregated/lambda_twostep_comparison.
"""

import os
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from joblib import Parallel, delayed

# Import base utilities
from simulation_base import (
    sample_random_aspects,
    generate_data_with_params,
    run_divtree_method,
    run_twostep_method,
    # run_divtree_forest_method,  # Commented out for now; add back later
    run_single_task_with_retry,
)

# Import local modules
import config
import utils

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> Path:
    return next(p for p in Path(SCRIPT_DIR).resolve().parents if (p / "pyproject.toml").exists())


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
    Run a single random simulation with DivTree (lambda 0,1,2,4,6,8) and TwoStep (tuned, recall, cap110, cap150).
    
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
    
    # Lambda values to test (exclude 3 and 10)
    lambda_values = [0, 1, 2, 4, 6, 8]
    
    try:
        data_size = aspect_values["data_size"]
        n_users_train = data_size
        n_users_test = data_size // 2
        
        # Check if data already exists (train/val/test format)
        data_dir = os.path.join(base_dir, "data", "lambda_twostep_comparison", f"simulation_{simulation_id:06d}")
        train_data_file = os.path.join(data_dir, "train_data.pkl")
        val_data_file = os.path.join(data_dir, "val_data.pkl")
        test_data_file = os.path.join(data_dir, "test_data.pkl")
        functional_form_file = os.path.join(data_dir, "functional_form.pickle")

        if all(os.path.exists(f) for f in [train_data_file, val_data_file, test_data_file, functional_form_file]):
            # Load existing data
            if verbose:
                print(f"Loading existing data for simulation {simulation_id}")
            train_df = pd.read_pickle(train_data_file)
            val_df = pd.read_pickle(val_data_file)
            test_df = pd.read_pickle(test_data_file)
            functional_form = utils.load_data(functional_form_file)["functional_form"]

            X_train = train_df[[col for col in train_df.columns if col.startswith("feature_")]].values
            T_train = train_df["T"].values
            YF_train = train_df["YF"].values
            YC_train = train_df["YC"].values
            region_type_train = train_df["region_type_true"].values

            X_val = val_df[[col for col in val_df.columns if col.startswith("feature_")]].values
            T_val = val_df["T"].values
            YF_val = val_df["YF"].values
            YC_val = val_df["YC"].values

            X_test = test_df[[col for col in test_df.columns if col.startswith("feature_")]].values
            T_test = test_df["T"].values
            YF_test = test_df["YF"].values
            YC_test = test_df["YC"].values
            region_type_test = test_df["region_type_true"].values
        else:
            # Generate new data (train/val/test 60/20/20)
            (
                X_train, T_train, YF_train, YC_train, region_type_train,
                X_val, T_val, YF_val, YC_val, region_type_val,
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

            val_df = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
            val_df["T"] = T_val
            val_df["YF"] = YF_val
            val_df["YC"] = YC_val
            val_df["region_type_true"] = region_type_val

            test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
            test_df["T"] = T_test
            test_df["YF"] = YF_test
            test_df["YC"] = YC_test
            test_df["region_type_true"] = region_type_test

            train_df.to_pickle(train_data_file)
            val_df.to_pickle(val_data_file)
            test_df.to_pickle(test_data_file)
            utils.save_data({"functional_form": functional_form}, functional_form_file)
        
        # Run all lambda methods
        method_results = []
        n_leaves_divtree0 = 4  # default fallback for TwoStep cap110/cap150
        for lambda_val in lambda_values:
            method_result = run_divtree_method(
                X_train, T_train, YF_train, YC_train,
                X_val, T_val, YF_val, YC_val,
                X_test, region_type_test,
                lambda_=float(lambda_val),
                regions_of_interest=None if lambda_val == 0 else [2],
                random_seed=random_seed,
                verbose=False,
            )
            method_results.append((lambda_val, method_result))
            if lambda_val == 0:
                n_leaves = method_result.get("n_leaves")
                if n_leaves is not None and np.isfinite(n_leaves):
                    n_leaves_divtree0 = max(2, int(n_leaves))
            
            # Store results with appropriate prefix
            if lambda_val == 0:
                prefix = "divtree_lambda0"
            else:
                prefix = f"divtree_lambda{lambda_val}_region2"
            
            for k, v in method_result.items():
                if k not in ["region_type_pred_train", "region_type_pred_test"]:
                    result[f"{prefix}_{k}"] = v
        
        # Run TwoStep (CausalForests once, four classification tree variants)
        twostep_result = run_twostep_method(
            X_train, T_train, YF_train, YC_train,
            X_val, T_val, YF_val, YC_val,
            X_test, region_type_test,
            n_leaves_divtree0=n_leaves_divtree0,
            random_seed=random_seed,
            verbose=False,
        )
        pred_keys_to_skip = [
            "twostep_tuned_region_type_pred_train", "twostep_tuned_region_type_pred_test",
            "twostep_recall_region_type_pred_train", "twostep_recall_region_type_pred_test",
            "twostep_cap110_region_type_pred_train", "twostep_cap110_region_type_pred_test",
            "twostep_cap150_region_type_pred_train", "twostep_cap150_region_type_pred_test",
        ]
        for k, v in twostep_result.items():
            if k not in pred_keys_to_skip:
                result[k] = v

        # DivTreeForest commented out for now; add back later.
        # forest_result = run_divtree_forest_method(
        #     X_train, T_train, YF_train, YC_train,
        #     X_val, T_val, YF_val, YC_val,
        #     X_test, region_type_test,
        #     lambda_=config.DIVTREE_FOREST_LAMBDA,
        #     n_estimators=config.DIVTREE_FOREST_N_ESTIMATORS,
        #     random_seed=random_seed,
        #     verbose=False,
        # )
        # for k, v in forest_result.items():
        #     if k not in ["divtree_forest_region_type_pred_train", "divtree_forest_region_type_pred_test"]:
        #         result[k] = v

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
        
        # Add TwoStep predictions to DataFrames
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap110", "twostep_cap150"]:
            pred_train = twostep_result.get(f"{prefix}_region_type_pred_train")
            pred_test = twostep_result.get(f"{prefix}_region_type_pred_test")
            col_name = f"{prefix}_region_pred"
            train_df[col_name] = pred_train if pred_train is not None else np.nan
            test_df[col_name] = pred_test if pred_test is not None else np.nan

        # DivTreeForest commented out for now.
        # pred_train = forest_result.get("divtree_forest_region_type_pred_train")
        # pred_test = forest_result.get("divtree_forest_region_type_pred_test")
        # train_df["divtree_forest_region_pred"] = pred_train if pred_train is not None else np.nan
        # test_df["divtree_forest_region_pred"] = pred_test if pred_test is not None else np.nan

        # Save updated DataFrames
        train_df.to_pickle(train_data_file)
        test_df.to_pickle(test_data_file)
        
    except Exception:
        # Mark all metrics as NaN
        metrics = [
            "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
            "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
            "precision_region_2", "recall_region_2",
            "balanced_accuracy", "mcc", "n_leaves", "runtime", "cpu_time"
        ]
        method_prefixes = ["divtree_lambda0"]
        for lambda_val in [1, 2, 4, 6, 8]:
            method_prefixes.append(f"divtree_lambda{lambda_val}_region2")
        
        for method_prefix in method_prefixes:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
        
        # TwoStep error fallback
        twostep_metrics = [
            "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
            "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
            "balanced_accuracy", "mcc", "n_leaves"
        ]
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap110", "twostep_cap150"]:
            for metric in twostep_metrics:
                result[f"{prefix}_{metric}"] = np.nan
            result[f"{prefix}_cpu_time"] = np.nan
            result[f"{prefix}_total_cpu_time"] = np.nan
        result["twostep_causal_forest_cpu_time"] = np.nan

        # DivTreeForest commented out for now.
        # for metric in twostep_metrics:
        #     result[f"divtree_forest_{metric}"] = np.nan
        # result["divtree_forest_cpu_time"] = np.nan

    finally:
        # Release memory before worker picks up next task (CausalForests are memory-heavy)
        gc.collect()
    
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
    Run lambda comparison simulation: Compare DivTree with lambda values 0, 1, 2, 4, 6, 8
    and TwoStep (tuned, recall, cap110, cap150). DivTreeForest is currently disabled.
    
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
        print("LAMBDA + TWOSTEP COMPARISON")
        print("=" * 60)
        print(f"Number of simulations: {n_simulations}")
        print(f"Parallel jobs: {effective_n_jobs} (leaving 1 core free for system tasks)")
        print(f"Base random seed: {base_random_seed}")
        print(f"Batch size for incremental saving: {batch_size}")
        print("\nMethods:")
        print("  DivTree: lambda=0, 1, 2, 4, 6, 8 (regions_of_interest=[2] for lambda>0)")
        print("  TwoStep tuned: tuned on accuracy")
        print("  TwoStep recall: tuned on recall of region 2")
        print("  TwoStep cap110/cap150: leaf cap 1.1x/1.5x DivTree lambda=0")
        print("  (DivTreeForest is commented out for now)")
        print("\nResults saved to aggregated/lambda_twostep_comparison")
    
    # Setup directories
    aggregated_dir = os.path.join(base_dir, "aggregated", "lambda_twostep_comparison")
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
            
            # Force garbage collection between batches to reduce memory (CausalForests are heavy)
            gc.collect()
        
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
    base_dir = str(_repo_root() / "outputs" / "simulations" / "comprehensive_simulation")
    n_simulations = 10000
    # Reduce n_jobs if hitting memory limits (CausalForests use ~2 * n_estimators trees per simulation)
    n_jobs = 20
    
    run_lambda_comparison(
        n_simulations=n_simulations,
        base_dir=base_dir,
        n_jobs=n_jobs,
        verbose=True,
        batch_size=100,
    )

