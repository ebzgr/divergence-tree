"""
Region-stratified lambda + TwoStep comparison simulation.
"""

import gzip
import os
import sys
from pathlib import Path
import pickle
import gc
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from simulation_base import (
    sample_random_aspects,
    generate_data_with_params,
    run_divtree_method,
    run_twostep_method,
    run_single_task_with_retry,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = next(p for p in Path(SCRIPT_DIR).resolve().parents if (p / "pyproject.toml").exists())
sys.path.append(os.path.join(BINARY_COMPARISON_DIR, "comprehensive_simulation"))
import utils


def _save_tree_artifact(path: str, payload: Dict[str, Any]) -> None:
    utils.safe_makedirs(os.path.dirname(path))
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_single_lambda_simulation(
    simulation_id: int,
    aspect_values: Dict[str, Any],
    random_seed: int,
    base_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    result = {"simulation_id": simulation_id, **aspect_values}
    lambda_values = config.LAMBDA_VALUES
    try:
        n_users_train = aspect_values["data_size"]
        n_users_test = n_users_train // 2

        data_dir = os.path.join(base_dir, "data", config.DATA_SUBDIR, f"simulation_{simulation_id:06d}")
        train_data_file = os.path.join(data_dir, "train_data.pkl")
        val_data_file = os.path.join(data_dir, "val_data.pkl")
        test_data_file = os.path.join(data_dir, "test_data.pkl")
        functional_form_file = os.path.join(data_dir, "functional_form.pickle")
        trees_file = os.path.join(data_dir, "trees.pkl.gz")

        if all(os.path.exists(f) for f in [train_data_file, val_data_file, test_data_file, functional_form_file]):
            train_df = pd.read_pickle(train_data_file)
            val_df = pd.read_pickle(val_data_file)
            test_df = pd.read_pickle(test_data_file)
            functional_form = utils.load_data(functional_form_file)["functional_form"]
            X_train = train_df[[c for c in train_df.columns if c.startswith("feature_")]].values
            T_train = train_df["T"].values
            YF_train = train_df["YF"].values
            YC_train = train_df["YC"].values
            X_val = val_df[[c for c in val_df.columns if c.startswith("feature_")]].values
            T_val = val_df["T"].values
            YF_val = val_df["YF"].values
            YC_val = val_df["YC"].values
            X_test = test_df[[c for c in test_df.columns if c.startswith("feature_")]].values
            region_type_test = test_df["region_type_true"].values
        else:
            (
                X_train, T_train, YF_train, YC_train, region_type_train,
                X_val, T_val, YF_val, YC_val, region_type_val,
                X_test, T_test, YF_test, YC_test, region_type_test,
                functional_form,
            ) = generate_data_with_params(
                noise=aspect_values["noise"],
                sparsity=aspect_values["sparsity"],
                dispersion=aspect_values["dispersion"],
                rareness=aspect_values["rareness"],
                n_users_train=n_users_train,
                n_users_test=n_users_test,
                random_seed=random_seed,
            )
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

        # Store realized DGP stats in results table.
        result["dispersion_target_region2"] = functional_form.get("dispersion_region2", np.nan)
        result["rareness_target_region2"] = functional_form.get("rareness_region2", np.nan)
        disp_real = functional_form.get("dispersion_by_region", {})
        rare_real = functional_form.get("rareness_by_region", {})
        for r in [1, 2, 3, 4]:
            result[f"dispersion_real_region{r}"] = float(disp_real.get(r, np.nan))
            result[f"rareness_real_region{r}"] = float(rare_real.get(r, np.nan))

        # Outcome summaries (observed outcomes) by treatment status.
        # We compute these over the combined train+val+test sample for the simulation.
        merged = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
        if {"T", "YF", "YC"}.issubset(set(merged.columns)):
            treated_vals = merged.loc[merged["T"] == 1, "YF"].dropna().astype(float).values
            control_vals = merged.loc[merged["T"] == 0, "YC"].dropna().astype(float).values
            result["outcome_treated_n"] = int(len(treated_vals))
            result["outcome_untreated_n"] = int(len(control_vals))
            result["outcome_treated_mean"] = float(np.mean(treated_vals)) if len(treated_vals) else np.nan
            result["outcome_treated_std"] = float(np.std(treated_vals)) if len(treated_vals) else np.nan
            result["outcome_untreated_mean"] = float(np.mean(control_vals)) if len(control_vals) else np.nan
            result["outcome_untreated_std"] = float(np.std(control_vals)) if len(control_vals) else np.nan
        else:
            result["outcome_treated_n"] = np.nan
            result["outcome_untreated_n"] = np.nan
            result["outcome_treated_mean"] = np.nan
            result["outcome_treated_std"] = np.nan
            result["outcome_untreated_mean"] = np.nan
            result["outcome_untreated_std"] = np.nan

        # Methods
        method_results = []
        divtree_models = {}
        n_leaves_divtree0 = 4
        for lam in lambda_values:
            mr = run_divtree_method(
                X_train, T_train, YF_train, YC_train,
                X_val, T_val, YF_val, YC_val,
                X_test, region_type_test,
                lambda_=float(lam),
                regions_of_interest=None if lam == 0 else [2],
                random_seed=random_seed,
                verbose=False,
                return_model=True,
            )
            model = mr.pop("_model", None)
            method_results.append((lam, mr))
            if model is not None:
                key = "divtree_lambda0" if lam == 0 else f"divtree_lambda{lam}_region2"
                divtree_models[key] = model
            if lam == 0:
                nlv = mr.get("n_leaves")
                if nlv is not None and np.isfinite(nlv):
                    n_leaves_divtree0 = max(2, int(nlv))
            prefix = "divtree_lambda0" if lam == 0 else f"divtree_lambda{lam}_region2"
            for k, v in mr.items():
                if k not in ["region_type_pred_train", "region_type_pred_test"]:
                    result[f"{prefix}_{k}"] = v

        tw = run_twostep_method(
            X_train, T_train, YF_train, YC_train,
            X_val, T_val, YF_val, YC_val,
            X_test, region_type_test,
            n_leaves_divtree0=n_leaves_divtree0,
            random_seed=random_seed,
            verbose=False,
            return_trees=True,
        )
        twostep_trees = tw.pop("_classification_trees", {})
        pred_skip = [
            "twostep_tuned_region_type_pred_train", "twostep_tuned_region_type_pred_test",
            "twostep_recall_region_type_pred_train", "twostep_recall_region_type_pred_test",
            "twostep_cap120_region_type_pred_train", "twostep_cap120_region_type_pred_test",
        ]
        for k, v in tw.items():
            if k not in pred_skip:
                result[k] = v

        for lam, mr in method_results:
            col = "divtree_lambda0_region_pred" if lam == 0 else f"divtree_lambda{lam}_region2_region_pred"
            train_df[col] = mr.get("region_type_pred_train", np.nan)
            test_df[col] = mr.get("region_type_pred_test", np.nan)
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap120"]:
            train_df[f"{prefix}_region_pred"] = tw.get(f"{prefix}_region_type_pred_train", np.nan)
            test_df[f"{prefix}_region_pred"] = tw.get(f"{prefix}_region_type_pred_test", np.nan)
        train_df.to_pickle(train_data_file)
        test_df.to_pickle(test_data_file)

        _save_tree_artifact(
            trees_file,
            {
                "simulation_id": simulation_id,
                "lambda_values": lambda_values,
                "divtree_models": divtree_models,
                "twostep_classification_trees": twostep_trees,
                "meta": {
                    "aspect_values": aspect_values,
                    "n_leaves_divtree0": n_leaves_divtree0,
                },
            },
        )
        result["tree_artifact_file"] = trees_file

    except Exception:
        metrics = [
            "accuracy", "acc_region_1", "acc_region_2", "acc_region_3", "acc_region_4",
            "fnr_region_2", "precision_region_2", "recall_region_2",
            "f1_region_1", "f1_region_2", "f1_region_3", "f1_region_4",
            "balanced_accuracy", "mcc", "n_leaves", "runtime", "cpu_time",
        ]
        result["outcome_treated_n"] = np.nan
        result["outcome_untreated_n"] = np.nan
        result["outcome_treated_mean"] = np.nan
        result["outcome_treated_std"] = np.nan
        result["outcome_untreated_mean"] = np.nan
        result["outcome_untreated_std"] = np.nan
        for method_prefix in ["divtree_lambda0"] + [f"divtree_lambda{v}_region2" for v in [1, 2, 4, 6, 8]]:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
        for prefix in ["twostep_tuned", "twostep_recall", "twostep_cap120"]:
            for metric in [m for m in metrics if m not in ("runtime", "cpu_time")]:
                result[f"{prefix}_{metric}"] = np.nan
            result[f"{prefix}_cpu_time"] = np.nan
            result[f"{prefix}_total_cpu_time"] = np.nan
        result["twostep_causal_forest_cpu_time"] = np.nan
        result["tree_artifact_file"] = None
    finally:
        gc.collect()

    return result


def run_lambda_comparison(
    n_simulations: int,
    base_dir: str,
    n_jobs: int = -1,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
    batch_size: int = 1000,
) -> None:
    if n_jobs <= 0:
        effective_n_jobs = max(1, (os.cpu_count() or 1) - 1)
    else:
        effective_n_jobs = n_jobs
    aggregated_dir = os.path.join(base_dir, "aggregated", config.AGGREGATED_SUBDIR)
    utils.safe_makedirs(aggregated_dir)
    results_file = os.path.join(aggregated_dir, "all_simulations_results.pkl")

    tasks = []
    for i in range(n_simulations):
        simulation_id = i + 1
        seed = base_random_seed + simulation_id * 1000
        tasks.append((simulation_id, sample_random_aspects(seed), seed))

    all_results = []
    n_batches = (n_simulations + batch_size - 1) // batch_size
    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, n_simulations)
        batch_tasks = tasks[lo:hi]
        if verbose:
            print(f"Batch {b+1}/{n_batches}: {lo+1}-{hi}")
        batch_results = Parallel(n_jobs=effective_n_jobs, verbose=10 if verbose else 0)(
            delayed(run_single_task_with_retry)(task, base_dir, run_single_lambda_simulation)
            for task in batch_tasks
        )
        all_results.extend(batch_results)
        pd.DataFrame(all_results).to_pickle(results_file)
        gc.collect()
    if verbose:
        print(f"Completed {len(all_results)} simulations. Results: {results_file}")


if __name__ == "__main__":
    base_dir = str(REPO_ROOT / "outputs" / "simulations" / "comprehensive_simulation_region_stratified")
    run_lambda_comparison(
        n_simulations=config.DEFAULT_N_SIMULATIONS,
        base_dir=base_dir,
        n_jobs=config.DEFAULT_N_JOBS,
        verbose=True,
        batch_size=config.DEFAULT_BATCH_SIZE,
    )

