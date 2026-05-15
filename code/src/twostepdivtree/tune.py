"""
Hyperparameter tuning helpers for TwoStepDivergenceTree.

This mirrors `divtree/tune.py` in style: tuning is performed outside the model
class, using Optuna and a held-out validation set.

We intentionally tune:
- GRF causal forest (`econml.grf.CausalForest`) hyperparameters using a proxy
  transformed-outcome loss on the validation set.
- Classification tree (`sklearn.tree.DecisionTreeClassifier`) hyperparameters
  using a validation-set scoring metric (accuracy/recall/fnr for a region).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import optuna.logging
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

try:
    from econml.grf import CausalForest
except ImportError as exc:  # pragma: no cover
    raise ImportError("econml is required for twostepdivtree tuning helpers.") from exc


def _as_1d(pred: Any) -> np.ndarray:
    """Normalize econml GRF predict output to a 1D numpy array."""
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    arr = np.asarray(pred)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(float, copy=False)


def transformed_outcome_holdout_loss(
    *,
    y_val: np.ndarray,
    t_val: np.ndarray,
    tau_hat_val: np.ndarray,
    mask_val: Optional[np.ndarray] = None,
) -> float:
    """
    Proxy loss for CATE tuning under randomized treatment with P(T=1)=0.5.

    Same structure as `divtree.tune.pseudo_outcome_holdout_loss`:
        T_F = (2T - 1) * Y
        loss = mean((T_F - 0.5 * tau_hat)^2)
    """
    y_val = np.asarray(y_val, dtype=float)
    t_val = np.asarray(t_val, dtype=int)
    tau_hat_val = np.asarray(tau_hat_val, dtype=float)
    if mask_val is None:
        mask_val = np.ones_like(y_val, dtype=bool)
    mask_val = np.asarray(mask_val, dtype=bool)
    if mask_val.sum() == 0:
        return float("inf")
    tf = (2 * t_val[mask_val] - 1) * y_val[mask_val]
    err = tf - 0.5 * tau_hat_val[mask_val]
    return float(np.mean(err * err))


def tune_grf_with_optuna_holdout(
    *,
    X_train: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    fixed: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 25,
    random_state: Optional[int] = 123,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """
    Tune econml.grf.CausalForest hyperparameters using validation loss.

    `search_space` supports two formats per parameter:
    - categorical list: {"param": [v1, v2, ...]}
    - ranged float/int spec (DivTree-style):
        {"param": {"low": <num>, "high": <num>, "log": <bool optional>, "type": "int" optional}}
    """
    fixed = dict(fixed or {})
    search_space = dict(search_space or {})

    # We only need point estimates of CATE for region labeling.
    fixed.setdefault("inference", False)

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed)
        for name, spec in search_space.items():
            # Ranged spec
            if isinstance(spec, dict) and "low" in spec and "high" in spec:
                low = spec["low"]
                high = spec["high"]
                is_log = bool(spec.get("log", False))
                is_int = spec.get("type") == "int" or (isinstance(low, int) and isinstance(high, int))
                if is_int:
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high), log=is_log)
                continue

            # Categorical spec (list/tuple/etc.)
            params[name] = trial.suggest_categorical(name, list(spec))

        try:
            cf = CausalForest(**params)
            cf.fit(X_train, T_train, Y_train)
            tau_hat_val = _as_1d(cf.predict(X_val))
            loss = transformed_outcome_holdout_loss(
                y_val=Y_val, t_val=T_val, tau_hat_val=tau_hat_val, mask_val=None
            )
            return loss if np.isfinite(loss) else 1e12
        except Exception:
            return 1e12

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=verbose)

    if len(study.trials) == 0 or study.best_trial is None:
        raise RuntimeError("No successful GRF tuning trials completed.")

    best_params = dict(fixed)
    best_params.update(study.best_trial.params)
    return best_params, float(study.best_value)


def score_region_predictions(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))

    if scoring.startswith("recall_region_"):
        target_region = int(scoring.split("_")[-1])
        true_mask = y_true == target_region
        if true_mask.sum() == 0:
            return 0.0
        tp = ((y_pred == target_region) & true_mask).sum()
        fn = ((y_pred != target_region) & true_mask).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if scoring.startswith("fnr_region_"):
        target_region = int(scoring.split("_")[-1])
        true_mask = y_true == target_region
        if true_mask.sum() == 0:
            return 0.0
        fnr = (y_pred[true_mask] != target_region).sum() / true_mask.sum()
        return float(-fnr)

    raise ValueError(
        f"Invalid scoring function: {scoring}. Expected 'accuracy', "
        "'recall_region_X', or 'fnr_region_X' where X is 1-4."
    )


def tune_classification_tree_with_optuna_holdout(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: Dict[str, Any],
    n_trials: int,
    scoring: str,
    max_leaf_nodes_search_space: Dict[str, int],
    random_state: Optional[int],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Tune DecisionTreeClassifier params using a held-out validation set.

    This replaces the v4-local `_tune_twostep_classification_tree_holdout`.
    """
    fixed = dict(base_params)
    search_space: Dict[str, Dict[str, int]] = {}

    if "max_depth" not in fixed:
        search_space["max_depth"] = {"low": 2, "high": 15}
    if "min_samples_split" not in fixed:
        search_space["min_samples_split"] = {"low": 2, "high": 20}
    if "min_samples_leaf" not in fixed:
        search_space["min_samples_leaf"] = {"low": 1, "high": 10}
    if "max_leaf_nodes" not in fixed:
        search_space["max_leaf_nodes"] = dict(max_leaf_nodes_search_space)

    if len(search_space) == 0:
        return fixed

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed)

        if "min_samples_leaf" in search_space:
            params["min_samples_leaf"] = trial.suggest_int(
                "min_samples_leaf",
                search_space["min_samples_leaf"]["low"],
                search_space["min_samples_leaf"]["high"],
            )

        if "min_samples_split" in search_space:
            min_split_low = search_space["min_samples_split"]["low"]
            if "min_samples_leaf" in params:
                min_split_low = max(min_split_low, 2 * int(params["min_samples_leaf"]))
            min_split_high = search_space["min_samples_split"]["high"]
            if min_split_low > min_split_high:
                return -1.0 if scoring.startswith("fnr_region_") else 0.0
            params["min_samples_split"] = trial.suggest_int("min_samples_split", min_split_low, min_split_high)

        for name, bounds in search_space.items():
            if name in {"min_samples_leaf", "min_samples_split"}:
                continue
            params[name] = trial.suggest_int(name, bounds["low"], bounds["high"])

        if "max_leaf_nodes" in params:
            params["max_depth"] = None
        if random_state is not None:
            params["random_state"] = random_state

        try:
            clf = DecisionTreeClassifier(**params)
            clf.fit(X_train, y_train)
            pred_val = clf.predict(X_val)
            score = score_region_predictions(y_val, pred_val, scoring=scoring)
            if not np.isfinite(score):
                return -1.0 if scoring.startswith("fnr_region_") else 0.0
            return float(score)
        except Exception:
            return -1.0 if scoring.startswith("fnr_region_") else 0.0

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=verbose)

    best_params = dict(fixed)
    best_params.update(study.best_trial.params)
    if "max_leaf_nodes" in best_params:
        best_params["max_depth"] = None
    if random_state is not None:
        best_params["random_state"] = random_state
    return best_params

