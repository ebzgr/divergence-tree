"""
Base utilities for region-stratified comprehensive simulation v3.
"""

import copy
import gc
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)

sys.path.append(os.path.join(BINARY_COMPARISON_DIR))
from region_stratified_dgp_v3 import generate_region_stratified_data

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna_holdout
from twostepdivtree.tree import TwoStepDivergenceTree

import config

sys.path.append(os.path.join(BINARY_COMPARISON_DIR, "comprehensive_simulation"))
import utils
from metrics import compute_all_metrics


def get_cpu_time() -> float:
    if _HAS_RESOURCE:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    return time.process_time()


def _alloc_categories(total_categories: int, k: int) -> list[int]:
    """
    Split total_categories across k variables with near-equal counts.
    Difference between min and max is at most 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    base = total_categories // k
    rem = total_categories % k
    return [base + 1 if i < rem else base for i in range(k)]


def sample_random_aspects(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        # Log-uniform sampling
        "noise": float(np.exp(rng.uniform(np.log(config.NOISE_MIN), np.log(config.NOISE_MAX)))),
        "data_size": int(np.exp(rng.uniform(np.log(config.DATA_SIZE_MIN), np.log(config.DATA_SIZE_MAX)))),
        "sparsity": int(rng.choice(config.SPARSITY_VALUES)),
        # Uniform share for region 2 on [1%, 50%]
        "rareness": float(rng.uniform(config.RARENESS_MIN, config.RARENESS_MAX)),
        # Log-uniform treatment-effect intensity
        "intensity": float(np.exp(rng.uniform(np.log(config.INTENSITY_MIN), np.log(config.INTENSITY_MAX)))),
    }


def generate_data_with_params(
    noise: float,
    sparsity: int,
    rareness: float,
    intensity: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
):
    n_categories = _alloc_categories(total_categories=50, k=sparsity)
    n_users_total = n_users_train + n_users_test
    X_all, T_all, YF_all, YC_all, _, _, region_type_all, functional_form = generate_region_stratified_data(
        n_users=n_users_total,
        k=sparsity,
        n_categories=n_categories,
        rareness_region2=rareness,
        intensity=intensity,
        effect_noise_std=config.DEFAULT_EFFECT_NOISE_STD,
        # v3: noise is additive outcome noise.
        firm_outcome_noise_std=noise,
        user_outcome_noise_std=noise,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    n_train = int(config.TRAIN_FRAC * n_users_total)
    n_val = int(config.VAL_FRAC * n_users_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return (
        X_all[train_indices],
        T_all[train_indices],
        YF_all[train_indices],
        YC_all[train_indices],
        region_type_all[train_indices],
        X_all[val_indices],
        T_all[val_indices],
        YF_all[val_indices],
        YC_all[val_indices],
        region_type_all[val_indices],
        X_all[test_indices],
        T_all[test_indices],
        YF_all[test_indices],
        YC_all[test_indices],
        region_type_all[test_indices],
        functional_form,
    )


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
    regions_of_interest,
    random_seed: int,
    verbose: bool = False,
    return_model: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        start = time.time()
        cpu_start = get_cpu_time()
        fixed_params = {
            "lambda_": lambda_,
            "n_quantiles": config.DIVTREE_FIXED_PARAMS["n_quantiles"],
            "eps_scale": config.DIVTREE_FIXED_PARAMS["eps_scale"],
            "random_state": random_seed,
        }
        if regions_of_interest is not None:
            fixed_params["regions_of_interest"] = regions_of_interest
        best_params, _ = tune_with_optuna_holdout(
            X_train,
            T_train,
            YF_train,
            YC_train,
            X_val,
            T_val,
            YF_val,
            YC_val,
            fixed=fixed_params,
            search_space=config.DIVTREE_SEARCH_SPACE,
            n_trials=config.DIVTREE_N_TRIALS,
            random_state=random_seed,
            verbose=verbose,
        )
        X_tv = np.concatenate([X_train, X_val], axis=0)
        T_tv = np.concatenate([T_train, T_val], axis=0)
        YF_tv = np.concatenate([YF_train, YF_val], axis=0)
        YC_tv = np.concatenate([YC_train, YC_val], axis=0)
        tree = DivergenceTree(**best_params)
        tree.fit(X_tv, T_tv, YF_tv, YC_tv)
        pred_test = tree.predict_region_type(X_test)
        pred_train = tree.predict_region_type(X_train)
        n_leaves = len(tree.leaf_effects()["leaves"])
        metrics = compute_all_metrics(region_type_test, pred_test, method_name="")
        result.update(
            {
                "region_type_pred_train": pred_train,
                "region_type_pred_test": pred_test,
                "n_leaves": n_leaves,
                "runtime": time.time() - start,
                "cpu_time": get_cpu_time() - cpu_start,
            }
        )
        result.update(metrics)
        if return_model:
            result["_model"] = tree
    except Exception:
        result = {
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
            "cpu_time": np.nan,
        }
        for metric in [
            "accuracy",
            "acc_region_1",
            "acc_region_2",
            "acc_region_3",
            "acc_region_4",
            "fnr_region_2",
            "precision_region_2",
            "recall_region_2",
            "f1_region_1",
            "f1_region_2",
            "f1_region_3",
            "f1_region_4",
            "balanced_accuracy",
            "mcc",
        ]:
            result[metric] = np.nan
        if return_model:
            result["_model"] = None
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
    random_seed: int,
    verbose: bool = False,
    return_trees: bool = False,
    tuned_variants: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        X_tv = np.concatenate([X_train, X_val], axis=0)
        T_tv = np.concatenate([T_train, T_val], axis=0)
        YF_tv = np.concatenate([YF_train, YF_val], axis=0)
        YC_tv = np.concatenate([YC_train, YC_val], axis=0)

        # v3 requirement: min_samples_split = 0.5% of observations for causal forest.
        min_samples_split = max(2, int(np.ceil(0.005 * len(X_tv))))
        cf_params = {
            **config.TWOSTEP_CAUSAL_FOREST_PARAMS,
            "random_state": random_seed,
            "min_samples_split": min_samples_split,
        }
        twostep = TwoStepDivergenceTree(
            causal_forest_params=cf_params,
            classification_tree_params={"random_state": random_seed},
            causal_forest_tune_params={},
        )
        cpu_cf_start = get_cpu_time()
        twostep.fit(
            X_tv,
            T_tv,
            YF_tv,
            YC_tv,
            auto_tune_classification_tree=False,
            fit_classification_tree=False,
            verbose=verbose,
        )
        cpu_cf = get_cpu_time() - cpu_cf_start
        base_ct_params = dict(twostep.classification_tree_params)

        if tuned_variants is None:
            tuned_variants = dict(config.TWOSTEP_TUNED_VARIANTS)
        variant_preds = {}
        variant_n_leaves = {}
        variant_cpu = {}
        variant_metrics = {}
        variant_trees = {}
        for variant_name, scoring in tuned_variants.items():
            twostep.classification_tree_params = dict(base_ct_params)
            cpu_start = get_cpu_time()
            twostep._fit_classification_tree_step(
                X_tv,
                auto_tune_classification_tree=True,
                classification_tree_tune_n_trials=config.TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS,
                classification_tree_tune_n_splits=config.TWOSTEP_CLASSIFICATION_TUNE_N_SPLITS,
                classification_tree_scoring=scoring,
                tune_max_leaf_nodes=True,
                max_leaf_nodes_search_space=config.TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES,
                verbose=verbose,
            )
            pred_test = twostep.predict_region_type(X_test)
            pred_train = twostep.predict_region_type(X_train)
            variant_preds[variant_name] = {
                "train": pred_train,
                "test": pred_test,
            }
            variant_n_leaves[variant_name] = int((twostep.classification_tree_.tree_.children_left == -1).sum())
            variant_cpu[variant_name] = get_cpu_time() - cpu_start
            variant_metrics[variant_name] = compute_all_metrics(region_type_test, pred_test, method_name="")
            variant_trees[variant_name] = copy.deepcopy(twostep.classification_tree_)

        result = {"twostep_causal_forest_cpu_time": cpu_cf}
        for variant_name in tuned_variants:
            result[f"{variant_name}_region_type_pred_train"] = variant_preds[variant_name]["train"]
            result[f"{variant_name}_region_type_pred_test"] = variant_preds[variant_name]["test"]
            result[f"{variant_name}_n_leaves"] = variant_n_leaves[variant_name]
            result[f"{variant_name}_cpu_time"] = variant_cpu[variant_name]
            result[f"{variant_name}_total_cpu_time"] = cpu_cf + variant_cpu[variant_name]
            for k, v in variant_metrics[variant_name].items():
                result[f"{variant_name}_{k}"] = v
        if return_trees:
            result["_classification_trees"] = dict(variant_trees)
    except Exception:
        result = {
            "twostep_tuned_region_type_pred_train": None,
            "twostep_tuned_region_type_pred_test": None,
            "twostep_recall_region_type_pred_train": None,
            "twostep_recall_region_type_pred_test": None,
            "twostep_tuned_n_leaves": np.nan,
            "twostep_recall_n_leaves": np.nan,
            "twostep_causal_forest_cpu_time": np.nan,
            "twostep_tuned_cpu_time": np.nan,
            "twostep_recall_cpu_time": np.nan,
            "twostep_tuned_total_cpu_time": np.nan,
            "twostep_recall_total_cpu_time": np.nan,
        }
        for prefix in ["twostep_tuned", "twostep_recall"]:
            for metric in [
                "accuracy",
                "acc_region_1",
                "acc_region_2",
                "acc_region_3",
                "acc_region_4",
                "fnr_region_2",
                "precision_region_2",
                "recall_region_2",
                "f1_region_1",
                "f1_region_2",
                "f1_region_3",
                "f1_region_4",
                "balanced_accuracy",
                "mcc",
            ]:
                result[f"{prefix}_{metric}"] = np.nan
        if return_trees:
            result["_classification_trees"] = {}
    return result


def run_single_task_with_retry(task, base_dir: str, simulation_function: callable) -> Dict[str, Any]:
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
        except Exception as exc:
            is_mem = (
                isinstance(exc, MemoryError)
                or "MemoryError" in type(exc).__name__
                or "Unable to allocate" in str(exc)
            )
            if is_mem and attempt < max_retries - 1:
                gc.collect()
                time.sleep(60)
                continue
            raise

