"""
Base utilities for region-stratified binary-features simulation.

Method wrappers use DivTree and TwoStep from code/src (pip install -e .).
"""

import gc
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from paths import setup_binary_features_path

setup_binary_features_path()

import config
from dgp import generate_region_stratified_data
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna_holdout
from metrics import compute_all_metrics
from twostepdivtree.tree import TwoStepDivergenceTree
from twostepdivtree.tune import (
    tune_classification_tree_with_optuna_holdout,
    tune_grf_with_optuna_holdout,
)

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False


def get_cpu_time() -> float:
    if _HAS_RESOURCE:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    return time.process_time()


def _explicit_n_categories_for_sparsity(k: int) -> list[int]:
    """
    Explicit category allocations for sparsity settings.

    The total number of categories is fixed at 30, allocated as:
    - k=1  -> [30]
    - k=3  -> [10,10,10]
    - k=6  -> [5]*6
    - k=10 -> [3]*10
    """
    k = int(k)
    if k == 1:
        return [30]
    if k == 3:
        return [10, 10, 10]
    if k == 6:
        return [5] * 6
    if k == 10:
        return [3] * 10
    raise ValueError(
        f"Unsupported sparsity={k}. Expected one of {sorted(config.SPARSITY_GRID)}."
    )


def generate_data_with_params(
    *,
    noise: float,
    sparsity: int,
    rareness: float,
    n_users_train: int,
    n_users_test: int,
    random_seed: int,
):
    n_categories = _explicit_n_categories_for_sparsity(sparsity)
    n_users_total = n_users_train + n_users_test
    X_all, T_all, YF_all, YC_all, _, _, region_type_all, functional_form = generate_region_stratified_data(
        n_users=n_users_total,
        k=sparsity,
        n_categories=n_categories,
        rareness_region2=rareness,
        intensity=config.INTENSITY_FIXED,
        effect_noise_std=config.DEFAULT_EFFECT_NOISE_STD,
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
        else:
            tree = None
            gc.collect()
    except Exception as exc:
        result = {
            "region_type_pred_train": None,
            "region_type_pred_test": None,
            "n_leaves": np.nan,
            "runtime": np.nan,
            "cpu_time": np.nan,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "error_tb": (traceback.format_exc(limit=50) or "")[-8000:],
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
    tuned_variants: Optional[Dict[str, str]] = None,
    causal_forest_params_override: Optional[Dict[str, Any]] = None,
    causal_forest_tune_params_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        cf_fixed = {**config.TWOSTEP_CAUSAL_FOREST_PARAMS, "random_state": random_seed, "inference": False}
        if causal_forest_params_override:
            cf_fixed.update(dict(causal_forest_params_override))

        cf_search = (
            config.TWOSTEP_CAUSAL_FOREST_TUNE_PARAMS.get("params", {})
            if causal_forest_tune_params_override is None
            else dict(causal_forest_tune_params_override).get("params", {})
        )

        cpu_cf_start = get_cpu_time()
        valid_train_F = ~np.isnan(YF_train)
        valid_val_F = ~np.isnan(YF_val)
        best_cf_params_F, _ = tune_grf_with_optuna_holdout(
            X_train=X_train[valid_train_F],
            T_train=T_train[valid_train_F],
            Y_train=YF_train[valid_train_F],
            X_val=X_val[valid_val_F],
            T_val=T_val[valid_val_F],
            Y_val=YF_val[valid_val_F],
            fixed=cf_fixed,
            search_space=cf_search,
            n_trials=int(config.TWOSTEP_CAUSAL_FOREST_TUNE_N_TRIALS),
            random_state=random_seed,
            verbose=verbose,
        )

        valid_train_C = ~np.isnan(YC_train)
        valid_val_C = ~np.isnan(YC_val)
        best_cf_params_C, _ = tune_grf_with_optuna_holdout(
            X_train=X_train[valid_train_C],
            T_train=T_train[valid_train_C],
            Y_train=YC_train[valid_train_C],
            X_val=X_val[valid_val_C],
            T_val=T_val[valid_val_C],
            Y_val=YC_val[valid_val_C],
            fixed=cf_fixed,
            search_space=cf_search,
            n_trials=int(config.TWOSTEP_CAUSAL_FOREST_TUNE_N_TRIALS),
            random_state=random_seed,
            verbose=verbose,
        )

        twostep_train = TwoStepDivergenceTree(
            causal_forest_params_F=best_cf_params_F,
            causal_forest_params_C=best_cf_params_C,
            classification_tree_params={"random_state": random_seed},
        )
        twostep_train.fit(
            X_train,
            T_train,
            YF_train,
            YC_train,
            fit_classification_tree=False,
            drop_forests=False,
            verbose=verbose,
        )
        tauF_train = twostep_train.tauF_
        tauC_train = twostep_train.tauC_
        tauF_val, tauC_val = twostep_train.predict_causal_forest_effects(X_val)
        region_types_train = twostep_train._categorize_region_types(tauF_train, tauC_train)
        region_types_val = twostep_train._categorize_region_types(tauF_val, tauC_val)

        X_tv = np.concatenate([X_train, X_val], axis=0)
        T_tv = np.concatenate([T_train, T_val], axis=0)
        YF_tv = np.concatenate([YF_train, YF_val], axis=0)
        YC_tv = np.concatenate([YC_train, YC_val], axis=0)

        twostep = TwoStepDivergenceTree(
            causal_forest_params_F=best_cf_params_F,
            causal_forest_params_C=best_cf_params_C,
            classification_tree_params={"random_state": random_seed},
        )
        twostep.fit(X_tv, T_tv, YF_tv, YC_tv, fit_classification_tree=False, verbose=verbose)
        cpu_cf = get_cpu_time() - cpu_cf_start
        base_ct_params = dict(twostep.classification_tree_params)

        if tuned_variants is None:
            tuned_variants = dict(config.TWOSTEP_TUNED_VARIANTS)
        variant_preds = {}
        variant_n_leaves = {}
        variant_cpu = {}
        variant_metrics = {}
        for variant_name, scoring in tuned_variants.items():
            twostep.classification_tree_params = dict(base_ct_params)
            cpu_start = get_cpu_time()
            tuned_params = tune_classification_tree_with_optuna_holdout(
                X_train=X_train,
                y_train=region_types_train,
                X_val=X_val,
                y_val=region_types_val,
                base_params=twostep.classification_tree_params,
                n_trials=config.TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS,
                scoring=scoring,
                max_leaf_nodes_search_space=config.TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES,
                random_state=random_seed,
                verbose=verbose,
            )
            twostep.classification_tree_params = tuned_params
            twostep._fit_classification_tree_step(
                X_tv,
                classification_tree_scoring=scoring,
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
            twostep.classification_tree_ = None
            gc.collect()

        result = {
            "twostep_causal_forest_cpu_time": cpu_cf,
        }
        for variant_name in tuned_variants:
            result[f"{variant_name}_region_type_pred_train"] = variant_preds[variant_name]["train"]
            result[f"{variant_name}_region_type_pred_test"] = variant_preds[variant_name]["test"]
            result[f"{variant_name}_n_leaves"] = variant_n_leaves[variant_name]
            result[f"{variant_name}_cpu_time"] = variant_cpu[variant_name]
            result[f"{variant_name}_total_cpu_time"] = cpu_cf + variant_cpu[variant_name]
            for k, v in variant_metrics[variant_name].items():
                result[f"{variant_name}_{k}"] = v

        twostep.tauF_ = None
        twostep.tauC_ = None
        twostep.region_types_ = None
        twostep._fit_data = {}
        twostep = None
        gc.collect()
    except Exception as exc:
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
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "error_tb": (traceback.format_exc(limit=50) or "")[-8000:],
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
    return result
