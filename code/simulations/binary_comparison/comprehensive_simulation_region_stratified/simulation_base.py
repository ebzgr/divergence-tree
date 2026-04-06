"""
Base utilities for region-stratified comprehensive simulation.
"""

import os
import sys
import time
import gc
import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import lognorm

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
from region_stratified_dgp import generate_region_stratified_data

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


def _sample_bounded_lognormal(
    rng: np.random.Generator,
    low: float,
    high: float,
    mu: float,
    sigma: float,
) -> float:
    """Truncated-lognormal sampling on [low, high] via inverse-CDF."""
    dist = lognorm(s=sigma, scale=np.exp(mu))
    f_low = float(dist.cdf(low))
    f_high = float(dist.cdf(high))
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_high <= f_low:
        return float(rng.uniform(low, high))
    u = rng.uniform(f_low, f_high)
    return float(np.clip(dist.ppf(u), low, high))


def sample_random_aspects(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "noise": float(np.exp(rng.uniform(np.log(config.NOISE_MIN), np.log(config.NOISE_MAX)))),
        "data_size": int(np.exp(rng.uniform(np.log(config.DATA_SIZE_MIN), np.log(config.DATA_SIZE_MAX)))),
        "sparsity": int(rng.choice(config.SPARSITY_VALUES)),
        "dispersion": _sample_bounded_lognormal(
            rng, config.DISPERSION_MIN, config.DISPERSION_MAX, config.LOGNORMAL_MU, config.LOGNORMAL_SIGMA
        ),
        "rareness": _sample_bounded_lognormal(
            rng, config.RARENESS_MIN, config.RARENESS_MAX, config.LOGNORMAL_MU, config.LOGNORMAL_SIGMA
        ),
    }


def generate_data_with_params(
    noise: float,
    sparsity: int,
    dispersion: float,
    rareness: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
):
    n_categories = [60 // sparsity] * sparsity
    n_users_total = n_users_train + n_users_test
    X_all, T_all, YF_all, YC_all, _, _, region_type_all, functional_form = generate_region_stratified_data(
        n_users=n_users_total,
        k=sparsity,
        n_categories=n_categories,
        dispersion_region2=dispersion,
        rareness_region2=rareness,
        intensity=config.DEFAULT_INTENSITY,
        effect_noise_std=noise,
        firm_outcome_noise_std=0.0,
        user_outcome_noise_std=0.0,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    n_train = int(config.TRAIN_FRAC * n_users_total)
    n_val = int(config.VAL_FRAC * n_users_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return (
        X_all[train_indices], T_all[train_indices], YF_all[train_indices], YC_all[train_indices], region_type_all[train_indices],
        X_all[val_indices], T_all[val_indices], YF_all[val_indices], YC_all[val_indices], region_type_all[val_indices],
        X_all[test_indices], T_all[test_indices], YF_all[test_indices], YC_all[test_indices], region_type_all[test_indices],
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
            X_train, T_train, YF_train, YC_train,
            X_val, T_val, YF_val, YC_val,
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
        result.update({
            "region_type_pred_train": pred_train,
            "region_type_pred_test": pred_test,
            "n_leaves": n_leaves,
            "runtime": time.time() - start,
            "cpu_time": get_cpu_time() - cpu_start,
        })
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
        for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                       "fnr_region_2", "precision_region_2", "recall_region_2",
                       "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                       "balanced_accuracy", "mcc"]:
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
    n_leaves_divtree0: int,
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
        twostep = TwoStepDivergenceTree(
            causal_forest_params={**config.TWOSTEP_CAUSAL_FOREST_PARAMS, "random_state": random_seed},
            classification_tree_params={"random_state": random_seed},
            causal_forest_tune_params={},
        )
        cpu_cf_start = get_cpu_time()
        twostep.fit(X_tv, T_tv, YF_tv, YC_tv, auto_tune_classification_tree=False, fit_classification_tree=False, verbose=verbose)
        cpu_cf = get_cpu_time() - cpu_cf_start
        base_ct_params = dict(twostep.classification_tree_params)

        # Tuned variants are parameterized by scoring function.
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

        # cap120
        max_leaf_cap120 = max(2, int(np.ceil(1.2 * n_leaves_divtree0)))
        cpu_cap120_start = get_cpu_time()
        twostep._fit_classification_tree_step(X_tv, auto_tune_classification_tree=False, max_leaf_nodes=max_leaf_cap120, verbose=verbose)
        pred_cap120_test = twostep.predict_region_type(X_test)
        pred_cap120_train = twostep.predict_region_type(X_train)
        n_leaves_cap120 = int((twostep.classification_tree_.tree_.children_left == -1).sum())
        cpu_cap120 = get_cpu_time() - cpu_cap120_start
        metrics_cap120 = compute_all_metrics(region_type_test, pred_cap120_test, method_name="")
        tree_cap120 = copy.deepcopy(twostep.classification_tree_)

        result = {
            "twostep_cap120_region_type_pred_train": pred_cap120_train,
            "twostep_cap120_region_type_pred_test": pred_cap120_test,
            "twostep_cap120_n_leaves": n_leaves_cap120,
            "twostep_causal_forest_cpu_time": cpu_cf,
            "twostep_cap120_cpu_time": cpu_cap120,
            "twostep_cap120_total_cpu_time": cpu_cf + cpu_cap120,
        }
        for variant_name in tuned_variants:
            result[f"{variant_name}_region_type_pred_train"] = variant_preds[variant_name]["train"]
            result[f"{variant_name}_region_type_pred_test"] = variant_preds[variant_name]["test"]
            result[f"{variant_name}_n_leaves"] = variant_n_leaves[variant_name]
            result[f"{variant_name}_cpu_time"] = variant_cpu[variant_name]
            result[f"{variant_name}_total_cpu_time"] = cpu_cf + variant_cpu[variant_name]
            for k, v in variant_metrics[variant_name].items():
                result[f"{variant_name}_{k}"] = v
        for k, v in metrics_cap120.items():
            result[f"twostep_cap120_{k}"] = v
        if return_trees:
            trees_out = dict(variant_trees)
            trees_out["twostep_cap120"] = tree_cap120
            result["_classification_trees"] = trees_out
    except Exception:
        result = {
            "twostep_tuned_region_type_pred_train": None,
            "twostep_tuned_region_type_pred_test": None,
            "twostep_recall_region_type_pred_train": None,
            "twostep_recall_region_type_pred_test": None,
            "twostep_cap120_region_type_pred_train": None,
            "twostep_cap120_region_type_pred_test": None,
            "twostep_tuned_n_leaves": np.nan,
            "twostep_recall_n_leaves": np.nan,
            "twostep_cap120_n_leaves": np.nan,
            "twostep_causal_forest_cpu_time": np.nan,
            "twostep_tuned_cpu_time": np.nan,
            "twostep_recall_cpu_time": np.nan,
            "twostep_cap120_cpu_time": np.nan,
            "twostep_tuned_total_cpu_time": np.nan,
            "twostep_recall_total_cpu_time": np.nan,
            "twostep_cap120_total_cpu_time": np.nan,
        }
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap120"]:
            for metric in ["accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
                           "fnr_region_2", "precision_region_2", "recall_region_2",
                           "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
                           "balanced_accuracy", "mcc"]:
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
            is_mem = isinstance(exc, MemoryError) or "MemoryError" in type(exc).__name__ or "Unable to allocate" in str(exc)
            if is_mem and attempt < max_retries - 1:
                gc.collect()
                time.sleep(60)
                continue
            raise

