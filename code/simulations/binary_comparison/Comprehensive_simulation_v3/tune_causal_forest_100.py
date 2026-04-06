"""
Run tuning-only CausalForest study on 100 random v3 DGP draws.

Outputs:
- per-run table with aspects + selected hyperparameters + average tree stats
- summary table with min/max and value-count frequencies
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import config
from simulation_base import sample_random_aspects, generate_data_with_params

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_COMPARISON_DIR = os.path.dirname(SCRIPT_DIR)
SIMULATIONS_DIR = os.path.dirname(BINARY_COMPARISON_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATIONS_DIR)
REPO_ROOT = next(p for p in Path(SCRIPT_DIR).resolve().parents if (p / "pyproject.toml").exists())
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from econml.dml import CausalForestDML


def _to_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _extract_selected_params(cf_model: Any) -> Dict[str, Any]:
    p = cf_model.get_params(deep=True) if hasattr(cf_model, "get_params") else {}

    def _pick(name: str) -> Any:
        # Prefer direct attribute (tune mutates estimator attributes), then fall back to get_params.
        if hasattr(cf_model, name):
            return getattr(cf_model, name)
        return p.get(name)

    return {
        "max_depth": _pick("max_depth"),
        "min_weight_fraction_leaf": _pick("min_weight_fraction_leaf"),
        "min_var_fraction_leaf": _pick("min_var_fraction_leaf"),
        "min_samples_split": _pick("min_samples_split"),
        "min_samples_leaf": _pick("min_samples_leaf"),
        "max_samples": _pick("max_samples"),
        "min_balancedness_tol": _pick("min_balancedness_tol"),
        "n_estimators": _pick("n_estimators"),
    }


def _iter_candidates(container: Any) -> Iterable[Any]:
    if container is None:
        return []
    if hasattr(container, "estimators_"):
        return list(getattr(container, "estimators_"))
    if hasattr(container, "__iter__") and not isinstance(container, (str, bytes, dict, np.ndarray)):
        try:
            return list(container)
        except Exception:
            return []
    return []


def _collect_tree_objects(obj: Any, out: List[Any], visited: set[int], depth: int = 0, max_depth: int = 5) -> None:
    if obj is None or depth > max_depth:
        return
    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    # sklearn-like tree wrapper
    if hasattr(obj, "tree_"):
        out.append(obj)
        return

    # recurse into common ensemble containers
    if hasattr(obj, "estimators_"):
        for e in getattr(obj, "estimators_", []):
            _collect_tree_objects(e, out, visited, depth + 1, max_depth)
        return

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict, np.ndarray)):
        try:
            for e in obj:
                _collect_tree_objects(e, out, visited, depth + 1, max_depth)
        except Exception:
            pass


def _compute_tree_stats(cf_model: Any) -> Dict[str, float]:
    trees: List[Any] = []
    visited: set[int] = set()

    # Try several likely entrypoints on econml wrappers.
    for candidate in [cf_model, getattr(cf_model, "model_cate", None), getattr(cf_model, "model_final_", None)]:
        _collect_tree_objects(candidate, trees, visited)

    depths: List[float] = []
    splits: List[float] = []
    leaves: List[float] = []

    for t in trees:
        # sklearn tree estimator
        if hasattr(t, "tree_"):
            tr = t.tree_
            d = float(getattr(tr, "max_depth", np.nan))
            n_leaves = float(getattr(tr, "n_leaves", np.nan))
            n_nodes = float(getattr(tr, "node_count", np.nan))
            if np.isfinite(n_nodes) and np.isfinite(n_leaves):
                n_splits = max(0.0, n_nodes - n_leaves)
            else:
                n_splits = np.nan
        else:
            # fallback: some estimators expose methods instead of tree_
            d = _to_float(t.get_depth()) if hasattr(t, "get_depth") else np.nan
            n_leaves = _to_float(t.get_n_leaves()) if hasattr(t, "get_n_leaves") else np.nan
            n_splits = np.nan
            if np.isfinite(n_leaves):
                n_splits = max(0.0, n_leaves - 1.0)

        if np.isfinite(d):
            depths.append(d)
        if np.isfinite(n_splits):
            splits.append(n_splits)
        if np.isfinite(n_leaves):
            leaves.append(n_leaves)

    return {
        "tree_count_detected": float(len(trees)),
        "avg_tree_depth": float(np.mean(depths)) if depths else np.nan,
        "avg_n_splits": float(np.mean(splits)) if splits else np.nan,
        "avg_n_leaves": float(np.mean(leaves)) if leaves else np.nan,
        "max_tree_depth": float(np.max(depths)) if depths else np.nan,
    }


def _format_cf_params(random_seed: int) -> Dict[str, Any]:
    return {
        **config.TWOSTEP_CAUSAL_FOREST_PARAMS,
        "random_state": random_seed,
    }


def _run_single_tuning_simulation(simulation_id: int, seed: int) -> Dict[str, Any]:
    aspects = sample_random_aspects(seed)
    n_users_train = aspects["data_size"]
    n_users_test = n_users_train // 2
    (
        X_train,
        T_train,
        YF_train,
        _YC_train,
        _region_type_train,
        X_val,
        T_val,
        YF_val,
        _YC_val,
        _region_type_val,
        _X_test,
        _T_test,
        _YF_test,
        _YC_test,
        _region_type_test,
        _functional_form,
    ) = generate_data_with_params(
        noise=aspects["noise"],
        sparsity=aspects["sparsity"],
        rareness=aspects["rareness"],
        intensity=aspects["intensity"],
        n_users_train=n_users_train,
        n_users_test=n_users_test,
        random_seed=seed,
    )

    X_tv = np.concatenate([X_train, X_val], axis=0)
    T_tv = np.concatenate([T_train, T_val], axis=0)
    YF_tv = np.concatenate([YF_train, YF_val], axis=0)

    cf_params = _format_cf_params(seed)
    cf_firm = CausalForestDML(**cf_params)

    run_t0 = time.perf_counter()
    valid_f = ~np.isnan(YF_tv)
    if valid_f.sum() < 10:
        raise ValueError("Too few valid observations for firm outcome.")

    tune_cfg = config.TWOSTEP_CAUSAL_FOREST_TUNE_PARAMS or {}
    if tune_cfg:
        cf_firm.tune(
            Y=YF_tv[valid_f],
            T=T_tv[valid_f],
            X=X_tv[valid_f],
            **tune_cfg,
        )
    cf_firm.fit(
        Y=YF_tv[valid_f],
        T=T_tv[valid_f],
        X=X_tv[valid_f],
    )
    run_elapsed = time.perf_counter() - run_t0

    firm_sel = _extract_selected_params(cf_firm)
    firm_stats = _compute_tree_stats(cf_firm)

    return {
        "simulation_id": simulation_id,
        "random_seed": seed,
        **aspects,
        "cf_tune_runtime_sec": float(run_elapsed),
        "x_train_val_n": int(len(X_tv)),
        "firm_max_depth": firm_sel["max_depth"],
        "firm_min_weight_fraction_leaf": firm_sel["min_weight_fraction_leaf"],
        "firm_min_var_fraction_leaf": firm_sel["min_var_fraction_leaf"],
        "firm_min_samples_split": firm_sel["min_samples_split"],
        "firm_tree_count_detected": firm_stats["tree_count_detected"],
        "firm_avg_tree_depth": firm_stats["avg_tree_depth"],
        "firm_max_tree_depth": firm_stats["max_tree_depth"],
        "firm_avg_n_splits": firm_stats["avg_n_splits"],
        "firm_avg_n_leaves": firm_stats["avg_n_leaves"],
    }


def run_tuning_sweep(
    n_simulations: int = 100,
    base_random_seed: int = config.BASE_RANDOM_SEED,
    verbose: bool = True,
) -> Tuple[str, str]:
    t0 = time.perf_counter()
    out_dir = os.path.join(
        str(REPO_ROOT),
        "outputs",
        "simulations",
        "Comprehensive_simulation_v3",
        "aggregated",
        config.AGGREGATED_SUBDIR,
        "tuning",
    )
    os.makedirs(out_dir, exist_ok=True)
    runs_csv = os.path.join(out_dir, f"causal_forest_tuning_{n_simulations}_runs.csv")
    summary_csv = os.path.join(out_dir, f"causal_forest_tuning_{n_simulations}_summary.csv")

    rows: List[Dict[str, Any]] = []
    tasks = [
        (i + 1, base_random_seed + (i + 1) * 1000)
        for i in range(n_simulations)
    ]
    if verbose:
        print("[cf-tune] launching sequential sweep (outer workers=1)", flush=True)
    for i, (sim_id, seed) in enumerate(tasks, start=1):
        if verbose:
            print(f"[cf-tune] {i}/{n_simulations} starting (sim_id={sim_id}, seed={seed})", flush=True)
        row = _run_single_tuning_simulation(sim_id, seed)
        rows.append(row)
        if verbose:
            print(
                f"[cf-tune] {i}/{n_simulations} done (sim_id={sim_id}, seed={seed}, "
                f"noise={row['noise']:.6g}, data_size={int(row['data_size'])}, sparsity={int(row['sparsity'])}, "
                f"rareness={row['rareness']:.6g}, intensity={row['intensity']:.6g}) "
                f"firm(w={row['firm_min_weight_fraction_leaf']}, v={row['firm_min_var_fraction_leaf']}, d={row['firm_max_depth']}), "
                f"firm(avg_depth={row['firm_avg_tree_depth']:.2f}, avg_splits={row['firm_avg_n_splits']:.2f}) "
                f"in {row['cf_tune_runtime_sec']:.1f}s",
                flush=True,
            )

    runs_df = pd.DataFrame(rows)
    if not runs_df.empty and "simulation_id" in runs_df.columns:
        runs_df = runs_df.sort_values("simulation_id")
    runs_df.to_csv(runs_csv, index=False)

    summary_rows: List[Dict[str, Any]] = []
    summary_targets = [
        "firm_max_depth",
        "firm_min_weight_fraction_leaf",
        "firm_min_var_fraction_leaf",
        "firm_avg_tree_depth",
        "firm_avg_n_splits",
    ]
    for col in summary_targets:
        if col not in runs_df.columns:
            continue
        s = runs_df[col].dropna()
        if s.empty:
            continue
        summary_rows.append(
            {
                "metric": col,
                "min": float(np.min(s)),
                "max": float(np.max(s)),
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
                "n": int(s.size),
            }
        )
        vc = s.value_counts().sort_index()
        for val, cnt in vc.items():
            summary_rows.append(
                {
                    "metric": f"{col}__value_count",
                    "value": float(val) if isinstance(val, (int, float, np.floating, np.integer)) else val,
                    "count": int(cnt),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    if verbose:
        print(f"[cf-tune] Wrote per-run table: {runs_csv}", flush=True)
        print(f"[cf-tune] Wrote summary table: {summary_csv}", flush=True)
        print(f"[cf-tune] Total elapsed: {time.perf_counter() - t0:.1f}s", flush=True)

    return runs_csv, summary_csv


if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_tuning_sweep(
        n_simulations=n,
        verbose=True,
    )

