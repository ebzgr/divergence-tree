"""
Hyperparameter tuning for DivergenceTree using Optuna.

This module provides functions for tuning the divergence tree hyperparameters
using cross-validated pseudo-outcome loss. The tuning process uses Optuna's
TPE sampler to efficiently search the hyperparameter space.

Hyperparameters tuned:
- max_partitions: Maximum number of leaves to grow
- min_improvement_ratio: Minimum improvement ratio for keeping splits during pruning

The evaluation metric is K-fold cross-validated pseudo-outcome MSE loss,
which combines firm and consumer outcome prediction errors.
"""

import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, Any, Optional, Tuple
import optuna
import optuna.logging

from .tree import DivergenceTree


def pseudo_outcome_cv_loss(
    X: np.ndarray,
    T: np.ndarray,
    YF: np.ndarray,
    YC: np.ndarray,
    params: Dict[str, Any],
    n_splits: int = 5,
    random_state: Optional[int] = 123,
) -> float:
    """
    Compute K-fold cross-validated pseudo-outcome MSE loss.

    Uses the pseudo-outcome approach from Wager & Athey (2018) to evaluate
    treatment effect predictions. The loss combines firm and consumer outcome
    prediction errors. YC is only observed for subscribers (YF == 1), so
    non-subscribers are masked out of the consumer-outcome loss.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    T : np.ndarray of shape (n_samples,)
        Treatment indicator (0 or 1).
    YF : np.ndarray of shape (n_samples,)
        Firm outcome (binary, observed for all units).
    YC : np.ndarray of shape (n_samples,)
        Consumer outcome (continuous, NaN where YF == 0).
    params : dict
        Hyperparameters for DivergenceTree.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for KFold shuffling.

    Returns
    -------
    float
        Mean cross-validated pseudo-outcome MSE loss across all folds.
    """
    # Input validation
    n = X.shape[0]
    if len(T) != n or len(YF) != n or len(YC) != n:
        raise ValueError(
            f"Input arrays must have matching lengths: "
            f"X={n}, T={len(T)}, YF={len(YF)}, YC={len(YC)}"
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses = []

    mask_C = ~np.isnan(YC)
    T_F = (2 * T - 1) * YF
    T_C = (2 * T - 1) * np.nan_to_num(YC, nan=0.0)

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        T_train, T_val = T[train_idx], T[val_idx]
        YF_train, YF_val = YF[train_idx], YF[val_idx]
        YC_train, YC_val = YC[train_idx], YC[val_idx]

        try:
            tree = DivergenceTree(**params)
            tree.fit(X_train, T_train, YF_train, YC_train)

            leaves_val = tree.predict_leaf(X_val)
            tauF_hat = np.nan_to_num([leaf.tauF for leaf in leaves_val], nan=0.0)
            tauC_hat = np.nan_to_num([leaf.tauC for leaf in leaves_val], nan=0.0)
        except Exception:
            # If tree fitting fails, return a large loss
            losses.append(1e6)
            continue

        # firm outcome loss (all)
        err_F = (T_F[val_idx] - 0.5 * tauF_hat) ** 2
        firm_loss = np.mean(err_F)

        # consumer outcome loss (only subscribers)
        valid_mask = mask_C[val_idx]
        if np.any(valid_mask):
            err_C = (T_C[val_idx][valid_mask] - 0.5 * tauC_hat[valid_mask]) ** 2
            consumer_loss = np.mean(err_C)
            fold_loss = firm_loss + consumer_loss
        else:
            fold_loss = firm_loss

        losses.append(fold_loss)

    return float(np.mean(losses))


def tune_with_optuna(
    X: np.ndarray,
    T: np.ndarray,
    YF: np.ndarray,
    YC: np.ndarray,
    fixed: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: Optional[int] = 123,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """
    Hyperparameter tuning using Optuna with K-fold CV pseudo-outcome loss.

    Tunes `max_partitions` and `min_improvement_ratio` by default. If a
    search_space is provided, it will be used instead of the defaults.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    T : np.ndarray of shape (n_samples,)
        Treatment indicator (0 or 1).
    YF : np.ndarray of shape (n_samples,)
        Firm outcome (binary, observed for all units).
    YC : np.ndarray of shape (n_samples,)
        Consumer outcome (continuous, NaN where YF == 0).
    fixed : dict, optional
        Fixed hyperparameters for DivergenceTree. These will be
        used in all trials. Common examples: lambda_, n_quantiles, co_movement.
    search_space : dict, optional
        Custom search space specification. If None, defaults to tuning
        max_partitions (2-20) and min_improvement_ratio (0.001-0.1, log scale).
        Format: {"param_name": {"low": value, "high": value, "log": bool}}
    n_trials : int, default=50
        Number of Optuna optimization trials.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducibility (KFold and Optuna sampler).
    verbose : bool, default=True
        Whether to show Optuna progress bar and logging output.
        Set to False to suppress output during parallel execution.

    Returns
    -------
    best_params : dict
        Best hyperparameters found (combines fixed and tuned parameters).
    best_loss : float
        Best cross-validated pseudo-outcome MSE loss achieved.

    Raises
    ------
    RuntimeError
        If Optuna study fails to complete any successful trials.
    """
    fixed = dict(fixed or {})
    search_space = dict(search_space or {}) if search_space is not None else {}

    # Default search space for the two main hyperparameters
    if "max_partitions" not in search_space and "max_partitions" not in fixed:
        search_space["max_partitions"] = {"low": 2, "high": 20}
    if (
        "min_improvement_ratio" not in search_space
        and "min_improvement_ratio" not in fixed
    ):
        search_space["min_improvement_ratio"] = {"low": 0.001, "high": 0.1, "log": True}

    def objective(trial):
        params = dict(fixed)

        # Suggest hyperparameters from search space
        for name, spec in search_space.items():
            if name in fixed:
                continue  # Skip if already in fixed params

            if "log" in spec and spec["log"]:
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=True
                )
            elif isinstance(spec["low"], float) or isinstance(spec["high"], float):
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            else:
                step = spec.get("step", 1)
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], step=step
                )

        loss = pseudo_outcome_cv_loss(
            X, T, YF, YC, params, n_splits=n_splits, random_state=random_state
        )
        return loss if np.isfinite(loss) else 1e6

    # Set Optuna logging verbosity
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Check if any trials completed successfully
    if len(study.trials) == 0 or study.best_trial is None:
        raise RuntimeError(
            "No successful trials completed. Check that tree fitting "
            "succeeds with the provided parameters."
        )

    tuned = {k: v for k, v in study.best_trial.params.items() if k != "random_state"}
    best_params = dict(fixed)
    best_params.update(tuned)
    best_loss = study.best_value

    return best_params, best_loss
