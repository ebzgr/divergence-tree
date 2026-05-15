"""
Region-stratified lambda + TwoStep comparison simulation (binary features).

This script runs a fixed aspect grid with repeats and saves:
- per-simulation train/val/test dataframes under `outputs/.../data/.../simulation_XXXXXX/`
- an aggregated results table under `outputs/.../aggregated/.../all_simulations_results.pkl`

Key design points
-----------------
- Intensity is fixed via `config.INTENSITY_FIXED` (not part of the grid).
- Each (aspect combination, repeat_id) gets a deterministic seed derived from the aspects.
  This guarantees reproducibility under multiprocessing and ensures repeats are truly distinct.
- This version does not use dataset caching: it always regenerates data and overwrites
  per-simulation dataset files on disk.
- Batching: one batch == all (data_size × repeat_id) tasks for a fixed
  (noise, sparsity, rareness) triple.
- Resume mode: with `--cache`, skip tasks whose aspect keys already appear as **successful** rows in
  `aggregated/.../all_simulations_results.pkl` only (no scan of per-simulation `data/` folders).
"""

import gc
import hashlib
import os
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from paths import repo_root, setup_binary_features_path

setup_binary_features_path()

import config
import utils
from simulation_base import (
    generate_data_with_params,
    run_divtree_method,
    run_twostep_method,
)


def _seed_for(aspect_values: Dict[str, Any], base_random_seed: int) -> int:
    """
    Deterministic per-(combo, repeat) seed.

    We must vary seeds across repeats because multiprocessing would otherwise
    rerun the exact same DGP draw for each repetition.
    """
    payload = (
        f"ds={int(aspect_values['data_size'])}|"
        f"nz={float(aspect_values['noise'])}|"
        f"sp={int(aspect_values['sparsity'])}|"
        f"ra={float(aspect_values['rareness'])}|"
        f"rep={int(aspect_values.get('repeat_id', 0))}"
    ).encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8, person=b"divtreev4").digest()
    # Many sklearn/econml entrypoints ultimately validate seeds as uint32.
    # Keep seeds stable but always within [0, 2**32 - 1].
    seed64 = int(base_random_seed) + int.from_bytes(h, byteorder="little", signed=False)
    return int(seed64 % (2**32))


def run_single_lambda_simulation(
    simulation_id: int,
    aspect_values: Dict[str, Any],
    random_seed: int,
    base_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    result = {"simulation_id": simulation_id, **aspect_values, "intensity": float(config.INTENSITY_FIXED)}
    lambda_values = config.LAMBDA_VALUES

    try:
        n_users_train = int(aspect_values["data_size"])
        n_users_test = n_users_train // 2

        # Always regenerate data (no caching reads) and always write per-simulation datasets.
        data_dir = os.path.join(base_dir, "data", config.DATA_SUBDIR, f"simulation_{simulation_id:06d}")
        train_data_file = os.path.join(data_dir, "train_data.pkl")
        val_data_file = os.path.join(data_dir, "val_data.pkl")
        test_data_file = os.path.join(data_dir, "test_data.pkl")
        functional_form_file = os.path.join(data_dir, "functional_form.pickle")

        (
            X_train,
            T_train,
            YF_train,
            YC_train,
            region_type_train,
            X_val,
            T_val,
            YF_val,
            YC_val,
            region_type_val,
            X_test,
            T_test,
            YF_test,
            YC_test,
            region_type_test,
            functional_form,
        ) = generate_data_with_params(
            noise=float(aspect_values["noise"]),
            sparsity=int(aspect_values["sparsity"]),
            rareness=float(aspect_values["rareness"]),
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
        result["rareness_target_region2"] = functional_form.get("rareness_region2", np.nan)
        combo_share = functional_form.get("combo_share_by_region", {})
        obs_share = functional_form.get("obs_share_by_region", {})
        for r in [1, 2, 3, 4]:
            result[f"combo_share_real_region{r}"] = float(combo_share.get(r, np.nan))
            result[f"rareness_real_region{r}"] = float(obs_share.get(r, np.nan))

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

        method_results = []
        for lam in lambda_values:
            mr = run_divtree_method(
                X_train,
                T_train,
                YF_train,
                YC_train,
                X_val,
                T_val,
                YF_val,
                YC_val,
                X_test,
                region_type_test,
                lambda_=float(lam),
                regions_of_interest=None if lam == 0 else [2],
                random_seed=random_seed,
                verbose=False,
                return_model=False,
            )
            method_results.append((lam, mr))
            prefix = "divtree_lambda0" if lam == 0 else f"divtree_lambda{lam}_region2"
            for k, v in mr.items():
                if k not in ["region_type_pred_train", "region_type_pred_test"]:
                    result[f"{prefix}_{k}"] = v

        tw = run_twostep_method(
            X_train,
            T_train,
            YF_train,
            YC_train,
            X_val,
            T_val,
            YF_val,
            YC_val,
            X_test,
            region_type_test,
            random_seed=random_seed,
            verbose=False,
        )
        pred_skip = [
            "twostep_tuned_region_type_pred_train",
            "twostep_tuned_region_type_pred_test",
            "twostep_recall_region_type_pred_train",
            "twostep_recall_region_type_pred_test",
        ]
        for k, v in tw.items():
            if k not in pred_skip:
                result[k] = v

        for lam, mr in method_results:
            col = "divtree_lambda0_region_pred" if lam == 0 else f"divtree_lambda{lam}_region2_region_pred"
            train_df[col] = mr.get("region_type_pred_train", np.nan)
            test_df[col] = mr.get("region_type_pred_test", np.nan)
        for prefix in ["twostep_tuned", "twostep_recall"]:
            train_df[f"{prefix}_region_pred"] = tw.get(f"{prefix}_region_type_pred_train", np.nan)
            test_df[f"{prefix}_region_pred"] = tw.get(f"{prefix}_region_type_pred_test", np.nan)
        # Always overwrite final dfs (including prediction columns).
        train_df.to_pickle(train_data_file)
        test_df.to_pickle(test_data_file)

    except Exception as exc:
        # Do not fail silently: fast runs often indicate an exception path.
        result["error_type"] = type(exc).__name__
        result["error_msg"] = str(exc)
        # Keep traceback reasonably sized for pickling / viewing.
        tb = traceback.format_exc(limit=50)
        result["error_tb"] = tb[-8000:] if isinstance(tb, str) else None
        metrics = [
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
            "n_leaves",
            "runtime",
            "cpu_time",
        ]
        result["outcome_treated_n"] = np.nan
        result["outcome_untreated_n"] = np.nan
        result["outcome_treated_mean"] = np.nan
        result["outcome_treated_std"] = np.nan
        result["outcome_untreated_mean"] = np.nan
        result["outcome_untreated_std"] = np.nan
        for method_prefix in ["divtree_lambda0"] + [f"divtree_lambda{v}_region2" for v in [1, 2, 4, 8]]:
            for metric in metrics:
                result[f"{method_prefix}_{metric}"] = np.nan
        for prefix in ["twostep_tuned", "twostep_recall"]:
            for metric in [m for m in metrics if m not in ("runtime", "cpu_time")]:
                result[f"{prefix}_{metric}"] = np.nan
            result[f"{prefix}_cpu_time"] = np.nan
            result[f"{prefix}_total_cpu_time"] = np.nan
        result["twostep_causal_forest_cpu_time"] = np.nan
    finally:
        train_df = val_df = test_df = merged = None
        X_train = T_train = YF_train = YC_train = None
        X_val = T_val = YF_val = YC_val = None
        X_test = region_type_test = functional_form = None
        tw = method_results = None
        gc.collect()

    return result


def _append_results(results_file: str, rows: List[Dict[str, Any]]) -> None:
    """
    Append rows to the aggregated results pickle.

    We keep this simple and robust: read existing if present, then write back.
    """
    df_new = pd.DataFrame(rows)
    if os.path.exists(results_file):
        df_old = pd.read_pickle(results_file)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_pickle(results_file)


def _completed_keys_from_results(df: pd.DataFrame) -> set[tuple[float, int, float, int, int]]:
    """
    Return aspect keys considered completed in an existing results df.

    We treat a run as completed if it exists and appears successful:
    - no top-level `error_type`, and
    - at least one core metric is non-NaN (TwoStep or DivTree).
    """
    required = {"noise", "sparsity", "rareness", "data_size", "repeat_id"}
    if not required.issubset(set(df.columns)):
        return set()

    ok = pd.Series(True, index=df.index)
    if "error_type" in df.columns:
        ok &= df["error_type"].isna()

    core_cols = [c for c in ["twostep_tuned_accuracy", "divtree_lambda0_accuracy"] if c in df.columns]
    if core_cols:
        any_metric = pd.Series(False, index=df.index)
        for c in core_cols:
            any_metric |= df[c].notna()
        ok &= any_metric

    sub = df.loc[ok, ["noise", "sparsity", "rareness", "data_size", "repeat_id"]].copy()
    # Normalize types to stable tuple keys.
    keys = set(
        (
            float(r["noise"]),
            int(r["sparsity"]),
            float(r["rareness"]),
            int(r["data_size"]),
            int(r["repeat_id"]),
        )
        for _, r in sub.iterrows()
    )
    return keys


def _iter_batches() -> Tuple[int, int, Dict[str, float | int], List[Dict[str, Any]]]:
    """
    Yield batches in the requested execution order.

    Each yielded batch is defined by a fixed (sparsity, rareness) pair and contains
    all (data_size × repeat_id × noise) tasks.
    """
    n_repeats = int(config.N_REPEATS)
    batch_idx = 0
    n_batches_total = len(config.SPARSITY_GRID) * len(config.RARENESS_GRID)
    for sparsity in sorted(config.SPARSITY_GRID, reverse=True):
        for rareness in config.RARENESS_GRID:
            batch_idx += 1
            batch_key: Dict[str, float | int] = {
                "sparsity": int(sparsity),
                "rareness": float(rareness),
            }
            aspects: List[Dict[str, Any]] = []
            for data_size in sorted(config.DATA_SIZE_GRID, reverse=True):
                for rep in range(n_repeats):
                    for noise in config.NOISE_GRID:
                        aspects.append(
                            {
                                "data_size": int(data_size),
                                "noise": float(noise),
                                "sparsity": int(sparsity),
                                "rareness": float(rareness),
                                "repeat_id": int(rep),
                            }
                        )
            yield batch_idx, n_batches_total, batch_key, aspects


def run_lambda_comparison(
    base_dir: str,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
    use_cache: bool = False,
) -> None:
    # `batch_size` kept for API compatibility with v3, but intentionally unused in v4.
    aggregated_dir = os.path.join(base_dir, "aggregated", config.AGGREGATED_SUBDIR)
    utils.safe_makedirs(aggregated_dir)
    results_file = os.path.join(aggregated_dir, "all_simulations_results.pkl")

    n_jobs = int(config.DEFAULT_N_JOBS)
    n_repeats = int(config.N_REPEATS)
    total = (
        len(config.NOISE_GRID)
        * len(config.SPARSITY_GRID)
        * len(config.RARENESS_GRID)
        * len(config.DATA_SIZE_GRID)
        * n_repeats
    )

    completed_keys: set[tuple[float, int, float, int, int]] = set()
    sim_id_start = 0
    done = 0
    if use_cache and os.path.exists(results_file):
        existing = pd.read_pickle(results_file)
        completed_keys = _completed_keys_from_results(existing)
        if "simulation_id" in existing.columns and existing["simulation_id"].notna().any():
            sim_id_start = int(existing["simulation_id"].max())
        # Count completed tasks for progress visibility (only those we will skip).
        done = int(len(completed_keys))

    sim_id = sim_id_start

    for batch_idx, n_batches_total, batch_key, batch_aspects in _iter_batches():
        if verbose:
            print(
                f"Batch {batch_idx}/{n_batches_total}: "
                f"sparsity={batch_key['sparsity']} rareness={batch_key['rareness']} "
                f"(data_sizes={len(config.DATA_SIZE_GRID)}×repeats={n_repeats}×noises={len(config.NOISE_GRID)} => {len(batch_aspects)} runs) "
                f"| done={done}/{total} | n_jobs={n_jobs} | cache={'on' if use_cache else 'off'}",
                flush=True,
            )

        tasks: List[Tuple[int, Dict[str, Any], int]] = []
        for aspect_values in batch_aspects:
            key = (
                float(aspect_values["noise"]),
                int(aspect_values["sparsity"]),
                float(aspect_values["rareness"]),
                int(aspect_values["data_size"]),
                int(aspect_values["repeat_id"]),
            )
            if use_cache and key in completed_keys:
                continue
            sim_id += 1
            seed = _seed_for(aspect_values, base_random_seed=base_random_seed)
            tasks.append((sim_id, aspect_values, seed))

        if not tasks:
            continue

        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run_single_lambda_simulation)(
                simulation_id=sid,
                aspect_values=av,
                random_seed=sd,
                base_dir=base_dir,
                verbose=False,
            )
            for (sid, av, sd) in tasks
        )
        _append_results(results_file, results)
        done += len(results)
        del results
        gc.collect()

    if verbose:
        print(f"Completed {done} simulations. Results: {results_file}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run lambda + TwoStep grid simulation (binary features).")
    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Resume mode: load all_simulations_results.pkl and skip (noise, sparsity, rareness, "
            "data_size, repeat_id) tasks that already completed successfully (not inferred from data/)."
        ),
    )
    args = parser.parse_args()

    base_dir = str(repo_root() / "outputs" / "simulations" / "binary_features")
    run_lambda_comparison(
        base_dir=base_dir,
        batch_size=config.DEFAULT_BATCH_SIZE,
        base_random_seed=config.BASE_RANDOM_SEED,
        verbose=True,
        use_cache=bool(args.cache),
    )

