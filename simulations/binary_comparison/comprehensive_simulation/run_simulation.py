"""
Main simulation orchestration script for comprehensive divergence tree comparison.

Runs multiple replications of both DivergenceTree and TwoStepDivergenceTree
across different data generation settings.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
from joblib import Parallel, delayed
import gc

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SIMULATIONS_DIR))

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
from binary_data_generator import generate_binary_comparison_data

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna as tune_divtree
from twostepdivtree.tree import TwoStepDivergenceTree

# Import local modules
import config
import utils
from metrics import compute_all_metrics


def run_divtree_with_params(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    X_test: np.ndarray,
    region_type_test: np.ndarray,
    co_movement: str,
    lambda_: float,
    random_seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run DivergenceTree with specific parameters.
    
    Parameters
    ----------
    X_train, T_train, YF_train, YC_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test features.
    region_type_test : np.ndarray
        True region types for test set.
    co_movement : str
        "both" or "converge".
    lambda_ : float
        Lambda parameter.
    random_seed : int
        Random seed.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    dict
        Dictionary with metrics, predictions, treatment effects, and tree info.
    """
    result = {}
    try:
        if verbose:
            print(f"  Running DivergenceTree (co_movement={co_movement}, lambda={lambda_})...")
        start_time = time.time()
        
        # Tune hyperparameters
        fixed_params = {
            **config.DIVTREE_FIXED_PARAMS,
            "co_movement": co_movement,
            "lambda_": lambda_,
            "random_state": random_seed,
        }
        best_params, best_loss = tune_divtree(
            X_train,
            T_train,
            YF_train,
            YC_train,
            fixed=fixed_params,
            search_space=config.DIVTREE_SEARCH_SPACE,
            n_trials=config.DIVTREE_N_TRIALS,
            n_splits=config.DIVTREE_N_SPLITS,
            random_state=random_seed,
            verbose=verbose,
        )
        
        # Train final tree
        divtree = DivergenceTree(**best_params)
        divtree.fit(X_train, T_train, YF_train, YC_train)
        
        # Get treatment effects for train and test
        leaves_train = divtree.predict_leaf(X_train)
        leaves_test = divtree.predict_leaf(X_test)
        tauF_train = np.array([leaf.tauF if leaf.tauF is not None else 0.0 for leaf in leaves_train])
        tauC_train = np.array([leaf.tauC if leaf.tauC is not None else 0.0 for leaf in leaves_train])
        tauF_test = np.array([leaf.tauF if leaf.tauF is not None else 0.0 for leaf in leaves_test])
        tauC_test = np.array([leaf.tauC if leaf.tauC is not None else 0.0 for leaf in leaves_test])
        
        # Predict region types
        region_type_pred_train = divtree.predict_region_type(X_train)
        region_type_pred_test = divtree.predict_region_type(X_test)
        
        # Get number of leaves
        leaf_effects = divtree.leaf_effects()
        n_leaves = len(leaf_effects["leaves"])
        
        # Compute metrics with appropriate prefix
        method_prefix = f"divtree_{co_movement}_lambda{int(lambda_)}"
        metrics = compute_all_metrics(region_type_test, region_type_pred_test, method_name=method_prefix)
        
        runtime = time.time() - start_time
        
        # Add n_leaves and runtime to metrics
        metrics[f"{method_prefix}_n_leaves"] = n_leaves
        metrics[f"{method_prefix}_runtime"] = runtime
        
        result = {
            "tree": divtree,
            "tauF_train": tauF_train,
            "tauC_train": tauC_train,
            "tauF_test": tauF_test,
            "tauC_test": tauC_test,
            "region_type_pred_train": region_type_pred_train,
            "region_type_pred_test": region_type_pred_test,
            "n_leaves": n_leaves,
            "runtime": runtime,
            **metrics,
        }
        
    except Exception as e:
        if verbose:
            print(f"  DivergenceTree ({co_movement}, lambda={lambda_}) failed: {e}")
        method_prefix = f"divtree_{co_movement}_lambda{int(lambda_)}"
        result = {
            "tree": None,
            "tauF_train": None,
            "tauC_train": None,
            "tauF_test": None,
            "tauC_test": None,
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
        }
        # Add NaN metrics with proper prefix
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc", "rig", "n_leaves", "runtime"]:
            result[f"{method_prefix}_{metric}"] = np.nan
    
    return result


def generate_data_with_params(
    complexity: int,
    noise: float,
    outcome_noise: float,
    sparsity: int,
    rareness: float,
    covariance: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Generate data with specified parameters.

    Parameters
    ----------
    complexity : int
        m_firm = m_user value.
    noise : float
        effect_noise_std value.
    outcome_noise : float
        firm_outcome_noise_std = user_outcome_noise_std value.
    sparsity : int
        k value.
    rareness : float
        positive_ratio value.
    covariance : float
        similarity value.
    n_users_train : int
        Number of training observations.
    n_users_test : int
        Number of test observations.
    random_seed : int
        Random seed.

    Returns
    -------
    Tuple of data arrays and functional_form dict.
    """
    # Calculate n_categories to keep total features = 60
    n_categories = [60 // sparsity] * sparsity

    # Generate all data at once
    n_users_total = n_users_train + n_users_test
    (
        X_all,
        T_all,
        YF_all,
        YC_all,
        tauF_all,
        tauC_all,
        region_type_all,
        functional_form,
    ) = generate_binary_comparison_data(
        n_users=n_users_total,
        k=sparsity,
        n_categories=n_categories,
        m_firm=complexity,
        m_user=complexity,
        similarity=covariance,
        intensity=config.DEFAULT_INTENSITY,
        effect_noise_std=noise,
        firm_outcome_noise_std=outcome_noise,
        user_outcome_noise_std=outcome_noise,
        positive_ratio=rareness,
        random_seed=random_seed,
    )

    # Split into train and test sets
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    train_indices = indices[:n_users_train]
    test_indices = indices[n_users_train:]

    # Split the data
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
        X_train,
        T_train,
        YF_train,
        YC_train,
        region_type_train,
        X_test,
        T_test,
        YF_test,
        YC_test,
        region_type_test,
        functional_form,
    )


def run_single_scenario_with_retry(
    complexity: int,
    noise: float,
    outcome_noise: float,
    sparsity: int,
    rareness: float,
    covariance: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
    base_dir: str,
    mode: str,
    scenario_name: str,
    run_number: int,
    verbose: bool = True,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Wrapper for run_single_scenario with memory error retry logic.
    
    Retries up to max_retries times with 60 second wait on memory errors.
    """
    for attempt in range(max_retries):
        try:
            return run_single_scenario(
                complexity=complexity,
                noise=noise,
                outcome_noise=outcome_noise,
                sparsity=sparsity,
                rareness=rareness,
                covariance=covariance,
                n_users_train=n_users_train,
                n_users_test=n_users_test,
                random_seed=random_seed,
                base_dir=base_dir,
                mode=mode,
                scenario_name=scenario_name,
                run_number=run_number,
                verbose=verbose,
            )
        except Exception as e:
            # Check if it's a memory error (various forms)
            is_memory_error = (
                isinstance(e, MemoryError) or
                "MemoryError" in str(type(e).__name__) or
                "_ArrayMemoryError" in str(type(e).__name__) or
                "Unable to allocate" in str(e)
            )
            
            if is_memory_error:
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"Memory error on attempt {attempt + 1}/{max_retries}. Waiting 60 seconds before retry...")
                    gc.collect()  # Force garbage collection
                    time.sleep(60)  # Wait 60 seconds
                else:
                    # Last attempt failed, raise the error
                    if verbose:
                        print(f"Memory error after {max_retries} attempts. Marking as failed.")
                    raise
            else:
                # For non-memory errors, don't retry
                raise


def run_single_scenario(
    complexity: int,
    noise: float,
    outcome_noise: float,
    sparsity: int,
    rareness: float,
    covariance: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
    base_dir: str,
    mode: str,
    scenario_name: str,
    run_number: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single scenario: generate data, run both algorithms, compute metrics.

    Parameters
    ----------
    complexity : int
        m_firm = m_user value.
    noise : float
        effect_noise_std value.
    outcome_noise : float
        firm_outcome_noise_std = user_outcome_noise_std value.
    sparsity : int
        k value.
    rareness : float
        positive_ratio value.
    covariance : float
        similarity value.
    n_users_train : int
        Number of training observations.
    n_users_test : int
        Number of test observations.
    random_seed : int
        Random seed.
    base_dir : str
        Base directory for saving results.
    mode : str
        Either "one_at_a_time" or "grid_search".
    scenario_name : str
        Scenario name.
    run_number : int
        Run number (1-indexed).
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary containing all results and metrics.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Run {run_number}: {scenario_name}")
        print(f"{'='*60}")

    results = {
        "run_number": run_number,
        "complexity": complexity,
        "noise": noise,
        "outcome_noise": outcome_noise,
        "sparsity": sparsity,
        "rareness": rareness,
        "covariance": covariance,
    }

    try:
        # Generate data
        if verbose:
            print("Generating data...")
        data_start_time = time.time()
        (
            X_train,
            T_train,
            YF_train,
            YC_train,
            region_type_train,
            X_test,
            T_test,
            YF_test,
            YC_test,
            region_type_test,
            functional_form,
        ) = generate_data_with_params(
            complexity=complexity,
            noise=noise,
            outcome_noise=outcome_noise,
            sparsity=sparsity,
            rareness=rareness,
            covariance=covariance,
            n_users_train=n_users_train,
            n_users_test=n_users_test,
            random_seed=random_seed,
        )
        data_time = time.time() - data_start_time

        # Initialize variables for treatment effects and predictions
        # Store predictions from all 3 DivergenceTree versions
        divtree_predictions = {}  # Will store predictions from all 3 versions

        # Run DivergenceTree 3 times with different parameters
        if verbose:
            print("Running DivergenceTree (3 versions)...")
        
        divtree_results = {}
        divtree_configs = [
            ("both", 0),      # lambda = 0, type = both
            ("both", 2),      # lambda = 2, type = both
            ("diverge", 2),   # lambda = 2, type = diverge
        ]
        
        for co_movement, lambda_val in divtree_configs:
            divtree_result = run_divtree_with_params(
                X_train, T_train, YF_train, YC_train,
                X_test, region_type_test,
                co_movement, lambda_val, random_seed, verbose
            )
            config_name = f"{co_movement}_lambda{int(lambda_val)}"
            divtree_results[config_name] = divtree_result
            
            # Store predictions from all versions for data saving
            if divtree_result["region_type_pred_train"] is not None:
                divtree_predictions[f"{config_name}_train"] = divtree_result["region_type_pred_train"]
                divtree_predictions[f"{config_name}_test"] = divtree_result["region_type_pred_test"]
            
            # Use first successful result for treatment effects (default: both_lambda2)
            if co_movement == "both" and lambda_val == 2:
                if divtree_result["tauF_train"] is not None:
                    divtree_tauF_train = divtree_result["tauF_train"]
                    divtree_tauC_train = divtree_result["tauC_train"]
                    divtree_tauF_test = divtree_result["tauF_test"]
                    divtree_tauC_test = divtree_result["tauC_test"]
            
            # Update results with only scalar metrics (exclude large objects and arrays)
            # Remove large objects/arrays that are already saved separately
            metrics_only = {k: v for k, v in divtree_result.items() 
                           if k not in ["tree", "tauF_train", "tauC_train", "tauF_test", "tauC_test",
                                       "region_type_pred_train", "region_type_pred_test"]}
            results.update(metrics_only)

        # TwoStepDivergenceTree is not run for grid search (only 3 DivergenceTree versions)
        # If TwoStepDivergenceTree is enabled in the future, treatment probability should be passed:
        # treatment_probability = np.full(n_users_train, 0.5)  # 0.5 for all observations (random assignment)
        # twostep_tree.fit(X_train, T_train, YF_train, YC_train, treatment_probability=treatment_probability)

        results["data_generation_time"] = data_time

        # Save data as pandas DataFrames with treatment effects and predictions
        data_dir = utils.get_data_dir(base_dir, mode, scenario_name)
        utils.safe_makedirs(data_dir)  # Ensure directory exists before saving
        
        # Create training DataFrame
        train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        train_df["T"] = T_train
        train_df["YF"] = YF_train
        train_df["YC"] = YC_train
        train_df["region_type_true"] = region_type_train
        
        # Add treatment effects and predictions from all DivergenceTree versions
        if divtree_tauF_train is not None:
            train_df["divtree_tauF"] = divtree_tauF_train
            train_df["divtree_tauC"] = divtree_tauC_train
        else:
            train_df["divtree_tauF"] = np.nan
            train_df["divtree_tauC"] = np.nan
        
        # Add region type predictions from all 3 DivergenceTree versions
        for config_name in ["both_lambda0", "both_lambda2", "diverge_lambda2"]:
            train_key = f"{config_name}_train"
            if train_key in divtree_predictions:
                train_df[f"divtree_{config_name}_region_type_pred"] = divtree_predictions[train_key]
            else:
                train_df[f"divtree_{config_name}_region_type_pred"] = np.nan
        
        # Create test DataFrame
        test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        test_df["T"] = T_test
        test_df["YF"] = YF_test
        test_df["YC"] = YC_test
        test_df["region_type_true"] = region_type_test
        
        # Add treatment effects and predictions from all DivergenceTree versions
        if divtree_tauF_test is not None:
            test_df["divtree_tauF"] = divtree_tauF_test
            test_df["divtree_tauC"] = divtree_tauC_test
        else:
            test_df["divtree_tauF"] = np.nan
            test_df["divtree_tauC"] = np.nan
        
        # Add region type predictions from all 3 DivergenceTree versions
        for config_name in ["both_lambda0", "both_lambda2", "diverge_lambda2"]:
            test_key = f"{config_name}_test"
            if test_key in divtree_predictions:
                test_df[f"divtree_{config_name}_region_type_pred"] = divtree_predictions[test_key]
            else:
                test_df[f"divtree_{config_name}_region_type_pred"] = np.nan
        
        # Save DataFrames
        train_data_file = os.path.join(data_dir, "train_data.pkl")
        test_data_file = os.path.join(data_dir, "test_data.pkl")
        train_df.to_pickle(train_data_file)
        test_df.to_pickle(test_data_file)
        
        # Save functional form separately
        functional_form_file = os.path.join(data_dir, "functional_form.pickle")
        utils.save_data({"functional_form": functional_form}, functional_form_file)

        if verbose:
            print(f"Completed in {time.time() - data_start_time:.2f} seconds")

    except Exception as e:
        if verbose:
            print(f"Scenario failed: {e}")
        import traceback
        traceback.print_exc()
        # Mark all metrics as NaN for all 3 DivergenceTree versions
        divtree_configs = ["both_lambda0", "both_lambda2", "diverge_lambda2"]
        
        for config_name in divtree_configs:
            for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                          "fnr_region_2", "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                          "balanced_accuracy", "mcc", "rig", "n_leaves", "runtime"]:
                key = f"divtree_{config_name}_{metric}"
                if key not in results:
                    results[key] = np.nan
        
        if "data_generation_time" not in results:
            results["data_generation_time"] = np.nan

    return results


def run_grid_search(
    base_dir: str,
    n_replications: int = config.DEFAULT_N_REPLICATIONS,
    verbose: bool = True,
    complexity_values: Optional[List[int]] = None,
    noise_values: Optional[List[float]] = None,
    outcome_noise_values: Optional[List[float]] = None,
    sparsity_values: Optional[List[int]] = None,
    rareness_values: Optional[List[float]] = None,
    covariance_values: Optional[List[float]] = None,
    mode: str = "grid_search",
    aspect_name: Optional[str] = None,
    n_jobs: int = -1,
) -> None:
    """
    Run grid search: all combinations of aspect values.

    Parameters
    ----------
    base_dir : str
        Base directory for saving results.
    n_replications : int, default=config.DEFAULT_N_REPLICATIONS
        Number of replications per scenario.
    verbose : bool, default=True
        Whether to print progress.
    complexity_values : List[int], optional
        List of complexity values to test. If None, uses config.COMPLEXITY_VALUES.
    noise_values : List[float], optional
        List of noise values (effect_noise_std) to test. If None, uses config.NOISE_VALUES.
    outcome_noise_values : List[float], optional
        List of outcome noise values (outcome_noise_std) to test. If None, uses config.OUTCOME_NOISE_VALUES.
    sparsity_values : List[int], optional
        List of sparsity values to test. If None, uses config.SPARSITY_VALUES.
    rareness_values : List[float], optional
        List of rareness values to test. If None, uses config.RARENESS_VALUES.
    covariance_values : List[float], optional
        List of covariance values to test. If None, uses config.COVARIANCE_VALUES.
    mode : str, default="grid_search"
        Mode for saving results ("grid_search" or "one_at_a_time").
    aspect_name : str, optional
        Aspect name for one_at_a_time mode (used for output filename).
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available CPUs.
        Set to 1 to disable parallelization.
    """
    # Use defaults if not provided
    if complexity_values is None:
        complexity_values = config.COMPLEXITY_VALUES
    if noise_values is None:
        noise_values = config.NOISE_VALUES
    if outcome_noise_values is None:
        outcome_noise_values = config.OUTCOME_NOISE_VALUES
    if sparsity_values is None:
        sparsity_values = config.SPARSITY_VALUES
    if rareness_values is None:
        rareness_values = config.RARENESS_VALUES
    if covariance_values is None:
        covariance_values = config.COVARIANCE_VALUES

    if verbose:
        if mode == "grid_search":
            print("=" * 60)
            print("GRID SEARCH")
            print("=" * 60)
        else:
            print("=" * 60)
            print(f"ONE-AT-A-TIME ANALYSIS: {aspect_name}")
            print("=" * 60)

    # Generate all combinations
    # Include both noise and outcome_noise as separate dimensions
    all_combinations = list(
        product(
            complexity_values,
            noise_values,
            outcome_noise_values,
            sparsity_values,
            rareness_values,
            covariance_values,
        )
    )

    if verbose:
        print(f"Total scenarios: {len(all_combinations)}")
        print(f"Replications per scenario: {n_replications}")
        print(f"Total runs: {len(all_combinations) * n_replications}")
        print(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all CPUs'}")

    # Create list of all tasks: (scenario_id, scenario_params, run_number)
    all_tasks = []
    scenario_id = 0
    for combination in all_combinations:
        scenario_id += 1
        
        # Unpack combination: (complexity, noise, outcome_noise, sparsity, rareness, covariance)
        complexity, noise, outcome_noise, sparsity, rareness, covariance = combination
        
        scenario_params = {
            "complexity": complexity,
            "noise": noise,
            "outcome_noise": outcome_noise,
            "sparsity": sparsity,
            "rareness": rareness,
            "covariance": covariance,
            "scenario_id": scenario_id,
        }
        for run_number in range(1, n_replications + 1):
            all_tasks.append((scenario_params, run_number))

    # Helper function to run a single task
    def run_single_task(task):
        scenario_params, run_number = task
        scenario_id = scenario_params["scenario_id"]
        random_seed = config.BASE_RANDOM_SEED + scenario_id * 1000 + run_number
        # Create scenario name
        aspect_value = (
            scenario_params["complexity"] if aspect_name == "complexity" else
            scenario_params["noise"] if aspect_name == "noise" else
            scenario_params["outcome_noise"] if aspect_name == "outcome_noise" else
            scenario_params["sparsity"] if aspect_name == "sparsity" else
            scenario_params["rareness"] if aspect_name == "rareness" else
            scenario_params["covariance"] if aspect_name == "covariance" else None
        )
        
        scenario_name = utils.create_scenario_name(
            mode=mode,
            scenario_id=scenario_id if mode == "grid_search" else None,
            aspect=aspect_name if (mode == "one_at_a_time" or mode == "one_at_a_time_constrained") else None,
            aspect_value=aspect_value,
            run_number=run_number,
        )

        try:
            result = run_single_scenario_with_retry(
                complexity=scenario_params["complexity"],
                noise=scenario_params["noise"],
                outcome_noise=scenario_params["outcome_noise"],
                sparsity=scenario_params["sparsity"],
                rareness=scenario_params["rareness"],
                covariance=scenario_params["covariance"],
                n_users_train=config.DEFAULT_N_USERS_TRAIN,
                n_users_test=config.DEFAULT_N_USERS_TEST,
                random_seed=random_seed,
                base_dir=base_dir,
                mode=mode,
                scenario_name=scenario_name,
                run_number=run_number,
                verbose=False,  # Less verbose for parallel execution
            )

            if mode == "grid_search":
                result["scenario_id"] = scenario_id

            return result
        except Exception as e:
            # Return error result
            if verbose:
                print(f"Error in scenario {scenario_id}, run {run_number}: {e}")
            result = {
                "run_number": run_number,
                "complexity": scenario_params["complexity"],
                "noise": scenario_params["noise"],
                "outcome_noise": scenario_params["outcome_noise"],
                "sparsity": scenario_params["sparsity"],
                "rareness": scenario_params["rareness"],
                "covariance": scenario_params["covariance"],
            }
            if mode == "grid_search":
                result["scenario_id"] = scenario_id
            # Mark all metrics as NaN
            for key in [
                "divtree_accuracy",
                "divtree_acc_region_1",
                "divtree_acc_region_2",
                "divtree_acc_region_3",
                "divtree_acc_region_4",
                "divtree_fnr_region_2",
                "divtree_f1_region_1",
                "divtree_f1_region_2",
                "divtree_f1_region_3",
                "divtree_f1_region_4",
                "divtree_balanced_accuracy",
                "divtree_mcc",
                "divtree_rig",
                "divtree_n_leaves",
                "divtree_runtime",
                "twostep_accuracy",
                "twostep_acc_region_1",
                "twostep_acc_region_2",
                "twostep_acc_region_3",
                "twostep_acc_region_4",
                "twostep_fnr_region_2",
                "twostep_f1_region_1",
                "twostep_f1_region_2",
                "twostep_f1_region_3",
                "twostep_f1_region_4",
                "twostep_balanced_accuracy",
                "twostep_mcc",
                "twostep_rig",
                "twostep_n_leaves",
                "twostep_runtime",
                "data_generation_time",
            ]:
                result[key] = np.nan
            return result

    # Run all tasks in parallel
    if verbose:
        print(f"\nStarting parallel execution of {len(all_tasks)} tasks...")
    
    all_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_single_task)(task) for task in all_tasks
    )

    # Save intermediate results for grid_search (every 10 scenarios)
    if mode == "grid_search":
        # Group results by scenario_id
        scenario_results = {}
        for result in all_results:
            sid = result.get("scenario_id", 0)
            if sid not in scenario_results:
                scenario_results[sid] = []
            scenario_results[sid].append(result)
        
        # Save intermediate results
        for scenario_id in sorted(scenario_results.keys()):
            if scenario_id % 10 == 0:
                # Collect all results up to this scenario
                results_up_to_scenario = [
                    r for sid, results in scenario_results.items()
                    for r in results if sid <= scenario_id
                ]
                df = pd.DataFrame(results_up_to_scenario)
                aggregated_dir = utils.get_aggregated_dir(base_dir, mode)
                utils.safe_makedirs(aggregated_dir)
                intermediate_file = os.path.join(
                    aggregated_dir, f"all_scenarios_results_intermediate_{scenario_id}.pkl"
                )
                df.to_pickle(intermediate_file)
                if verbose:
                    print(f"Saved intermediate results to {intermediate_file}")

    # Save final aggregated results
    df = pd.DataFrame(all_results)
    aggregated_dir = utils.get_aggregated_dir(base_dir, mode)
    utils.safe_makedirs(aggregated_dir)
    
    if (mode == "one_at_a_time" or mode == "one_at_a_time_constrained") and aspect_name:
        output_file = os.path.join(aggregated_dir, f"{aspect_name}_results.pkl")
    else:
        output_file = os.path.join(aggregated_dir, "all_scenarios_results.pkl")
    
    df.to_pickle(output_file)
    if verbose:
        print(f"\nSaved aggregated results to {output_file}")


def run_simulation_for_aspect(
    aspect_name: str,
    base_dir: str,
    n_replications: int = config.DEFAULT_N_REPLICATIONS,
    verbose: bool = True,
    n_jobs: int = -1,
    mode: str = "one_at_a_time",
) -> None:
    """
    Run simulation for a single aspect: vary one aspect at a time.

    This function uses run_grid_search internally by setting non-varying aspects
    to single-element lists with default values.

    Parameters
    ----------
    aspect_name : str
        Name of the aspect to vary: "complexity", "noise", "sparsity", "rareness", or "covariance".
    base_dir : str
        Base directory for saving results.
    n_replications : int, default=config.DEFAULT_N_REPLICATIONS
        Number of replications per setting.
    verbose : bool, default=True
        Whether to print progress.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available CPUs.
        Set to 1 to disable parallelization.
    """
    # Map aspect names to their value lists and default values
    aspect_config = {
        "complexity": {
            "values": config.COMPLEXITY_VALUES,
            "default": config.DEFAULT_M_FIRM,
        },
        "noise": {
            "values": config.NOISE_VALUES,
            "default": config.DEFAULT_EFFECT_NOISE_STD,
        },
        "outcome_noise": {
            "values": config.OUTCOME_NOISE_VALUES,
            "default": config.DEFAULT_FIRM_OUTCOME_NOISE_STD,
        },
        "sparsity": {
            "values": config.SPARSITY_VALUES,
            "default": config.DEFAULT_K,
        },
        "rareness": {
            "values": config.RARENESS_VALUES,
            "default": config.DEFAULT_POSITIVE_RATIO,
        },
        "covariance": {
            "values": config.COVARIANCE_VALUES,
            "default": config.DEFAULT_SIMILARITY,
        },
    }

    if aspect_name not in aspect_config:
        raise ValueError(
            f"Unknown aspect: {aspect_name}. "
            f"Must be one of: {list(aspect_config.keys())}"
        )

    # Create search space: vary the specified aspect, others are single-element lists
    complexity_values = (
        aspect_config["complexity"]["values"]
        if aspect_name == "complexity"
        else [aspect_config["complexity"]["default"]]
    )
    noise_values = (
        aspect_config["noise"]["values"]
        if aspect_name == "noise"
        else [aspect_config["noise"]["default"]]
    )
    outcome_noise_values = (
        aspect_config["outcome_noise"]["values"]
        if aspect_name == "outcome_noise"
        else [aspect_config["outcome_noise"]["default"]]
    )
    sparsity_values = (
        aspect_config["sparsity"]["values"]
        if aspect_name == "sparsity"
        else [aspect_config["sparsity"]["default"]]
    )
    rareness_values = (
        aspect_config["rareness"]["values"]
        if aspect_name == "rareness"
        else [aspect_config["rareness"]["default"]]
    )
    covariance_values = (
        aspect_config["covariance"]["values"]
        if aspect_name == "covariance"
        else [aspect_config["covariance"]["default"]]
    )

    # Use run_grid_search with the modified search space
    run_grid_search(
        base_dir=base_dir,
        n_replications=n_replications,
        verbose=verbose,
        complexity_values=complexity_values,
        noise_values=noise_values,
        outcome_noise_values=outcome_noise_values,
        sparsity_values=sparsity_values,
        rareness_values=rareness_values,
        covariance_values=covariance_values,
        mode=mode,
        aspect_name=aspect_name,
        n_jobs=n_jobs,
    )




if __name__ == "__main__":
    # Configuration
    base_dir = os.path.join(SCRIPT_DIR, "output")
    n_replications = config.DEFAULT_N_REPLICATIONS
    verbose = True
    n_jobs = -1  # Use all available CPUs. Set to 1 to disable parallelization.

    # Grid search configuration
    # 3 values each for: complexity, noise, sparsity, rareness, covariance
    # Fixed at default: outcome_noise
    complexity_grid_values = [2, 9, 25]  # first, middle, last
    noise_grid_values = [0, 1, 9]  # first, middle, last
    sparsity_grid_values = [1, 3, 6]  # first, middle, last
    rareness_grid_values = [0.01, 0.2, 0.95]  # first, middle, last
    outcome_noise_grid_values = [config.DEFAULT_FIRM_OUTCOME_NOISE_STD]  # fixed at default
    covariance_grid_values = [0.2, 0.5, 0.8]  # 3 values for grid search
    
    # Run grid search
    run_grid_search(
        base_dir=base_dir,
        n_replications=n_replications,
        verbose=verbose,
        complexity_values=complexity_grid_values,
        noise_values=noise_grid_values,
        outcome_noise_values=outcome_noise_grid_values,
        sparsity_values=sparsity_grid_values,
        rareness_values=rareness_grid_values,
        covariance_values=covariance_grid_values,
        mode="grid_search",
        aspect_name=None,
        n_jobs=n_jobs,
    )

