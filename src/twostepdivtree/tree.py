"""
Two-Step Divergence Tree using Causal Forest and Classification Tree.

This implementation uses a two-step approach:
1. Step 1: Use causal forest to estimate treatment effects for YF and YC separately
2. Step 2: Categorize observations into 4 region types, then train a classification tree

The algorithm:
1. Fit separate causal forests for firm (YF) and consumer (YC) outcomes
2. Estimate tauF and tauC for each observation
3. Categorize observations into 4 region types based on effect signs:
   - Region 1: tauF > 0 and tauC > 0 (both positive)
   - Region 2: tauF > 0 and tauC <= 0 (firm+, customer-)
   - Region 3: tauF <= 0 and tauC > 0 (firm-, customer+)
   - Region 4: tauF <= 0 and tauC <= 0 (both negative)
4. Train a classification tree to predict region types from features
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import optuna

try:
    from econml.dml import CausalForestDML
except ImportError:
    raise ImportError(
        "econml is required for TwoStepDivergenceTree. "
        "Install it with: pip install econml"
    )


class TwoStepDivergenceTree:
    """
    Two-step divergence tree using causal forest and classification tree.

    This class implements an alternative approach to divergence tree estimation:
    1. Uses causal forest to estimate treatment effects separately for each outcome
    2. Categorizes observations into 4 region types based on effect signs
    3. Trains a classification tree to predict region types

    Parameters
    ----------
    causal_forest_params : dict, optional
        Parameters for causal forest models. Common parameters:
        - n_estimators: int, default=100
        - max_depth: int, default=None
        - min_samples_split: int, default=10
        - min_samples_leaf: int, default=5
        - n_jobs: int, default=None
            Number of parallel jobs to run. None means 1, -1 means use all processors.
            Causal forests support parallelization for faster training.
        - random_state: int, optional
    classification_tree_params : dict, optional
        Parameters for classification tree. Common parameters:
        - max_depth: int, default=None
        - min_samples_split: int, default=2
        - min_samples_leaf: int, default=1
        - random_state: int, optional
        Note: DecisionTreeClassifier (single tree) does not support parallelization.
    causal_forest_tune_params : dict, optional
        Parameters for causal forest tuning. The causal forest will be tuned
        during fit() using econml's built-in tune() method.
        - params: str or dict, default="auto"
            If "auto", uses econml's default grid. Otherwise, provide a dict
            with grid search space: {"param_name": [value1, value2, ...], ...}

    Attributes
    ----------
    causal_forest_F_ : CausalForestDML
        Fitted causal forest for firm outcome (YF).
    causal_forest_C_ : CausalForestDML
        Fitted causal forest for consumer outcome (YC).
    classification_tree_ : DecisionTreeClassifier
        Fitted classification tree for region type prediction.
    tauF_ : np.ndarray
        Estimated treatment effects for firm outcome.
    tauC_ : np.ndarray
        Estimated treatment effects for consumer outcome.
    region_types_ : np.ndarray
        Region type labels (1-4) for training data.
    """

    def __init__(
        self,
        causal_forest_params: Optional[Dict[str, Any]] = None,
        classification_tree_params: Optional[Dict[str, Any]] = None,
        causal_forest_tune_params: Optional[Dict[str, Any]] = None,
    ):
        self.causal_forest_params = dict(causal_forest_params or {})
        self.classification_tree_params = dict(classification_tree_params or {})
        self.causal_forest_tune_params = dict(causal_forest_tune_params or {})

        # Set defaults for causal forest
        if "n_estimators" not in self.causal_forest_params:
            self.causal_forest_params["n_estimators"] = 100
        if "max_depth" not in self.causal_forest_params:
            self.causal_forest_params["max_depth"] = None
        if "min_samples_split" not in self.causal_forest_params:
            self.causal_forest_params["min_samples_split"] = 10
        if "min_samples_leaf" not in self.causal_forest_params:
            self.causal_forest_params["min_samples_leaf"] = 5
        # n_jobs defaults to None (1 job) if not specified - users can set it to -1 for all CPUs

        # Set defaults for classification tree
        if "max_depth" not in self.classification_tree_params:
            self.classification_tree_params["max_depth"] = None
        if "min_samples_split" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_split"] = 2
        if "min_samples_leaf" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_leaf"] = 1

        # Will be set during fit
        self.causal_forest_F_: Optional[CausalForestDML] = None
        self.causal_forest_C_: Optional[CausalForestDML] = None
        self.classification_tree_: Optional[DecisionTreeClassifier] = None
        self.tauF_: Optional[np.ndarray] = None
        self.tauC_: Optional[np.ndarray] = None
        self.region_types_: Optional[np.ndarray] = None
        self._fit_data: Dict[str, Any] = {}
    
    def _fit_classification_tree_step(
        self,
        X: np.ndarray,
        auto_tune_classification_tree: Optional[bool] = None,
        classification_tree_tune_n_trials: Optional[int] = None,
        classification_tree_tune_n_splits: Optional[int] = None,
        classification_tree_scoring: str = "accuracy",
        verbose: bool = True,
    ) -> None:
        """
        Internal method to fit the classification tree (step 2) when CausalForests are already set.
        
        This is used when CausalForests have been pre-built and we only need to train
        the classification tree. Requires that causal_forest_F_ and causal_forest_C_
        are already set and fitted.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        auto_tune_classification_tree : bool, optional
            If True or None, automatically tunes the classification tree hyperparameters.
            If False, uses the provided `classification_tree_params`.
        classification_tree_tune_n_trials : int, optional
            Number of Optuna trials for classification tree tuning. Default: 30.
        classification_tree_tune_n_splits : int, optional
            Number of CV folds for classification tree tuning. Default: 5.
        classification_tree_scoring : str, default="accuracy"
            Scoring function for classification tree tuning.
        verbose : bool, default=True
            Whether to show progress output.
        """
        if self.causal_forest_F_ is None or self.causal_forest_C_ is None:
            raise ValueError("CausalForests must be set before calling _fit_classification_tree_step")
        
        X = np.asarray(X)
        
        # Store fit data
        if "X" not in self._fit_data:
            self._fit_data["X"] = X
        
        # Default to True if None
        if auto_tune_classification_tree is None:
            auto_tune_classification_tree = True
        
        # Predict treatment effects for all observations
        if verbose:
            print("Predicting treatment effects using pre-built CausalForests...")
        self.tauF_ = self.causal_forest_F_.effect(X)
        self.tauC_ = self.causal_forest_C_.effect(X)
        
        # Categorize observations into 4 region types
        if verbose:
            print("Categorizing observations into region types...")
        self.region_types_ = self._categorize_region_types(self.tauF_, self.tauC_)
        
        # Train classification tree (with optional auto-tuning)
        if auto_tune_classification_tree:
            if verbose:
                print("Auto-tuning classification tree hyperparameters...")
            ct_params, ct_score = self._tune_classification_tree(
                X, 
                self.region_types_,
                n_trials=classification_tree_tune_n_trials,
                n_splits=classification_tree_tune_n_splits,
                scoring=classification_tree_scoring,
                verbose=verbose,
            )
            if verbose:
                if classification_tree_scoring == "accuracy":
                    print(f"  Best classification tree accuracy: {ct_score:.6f}")
                else:
                    print(f"  Best classification tree {classification_tree_scoring}: {ct_score:.6f}")
            # Update classification_tree_params with tuned values
            self.classification_tree_params.update(ct_params)
        else:
            if verbose:
                print("Training classification tree with provided parameters...")
        
        self.classification_tree_ = DecisionTreeClassifier(
            **self.classification_tree_params
        )
        self.classification_tree_.fit(X, self.region_types_)
        
        if verbose:
            print("Classification tree fitting complete!")

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        YF: np.ndarray,
        YC: np.ndarray,
        auto_tune_classification_tree: Optional[bool] = None,
        classification_tree_tune_n_trials: Optional[int] = None,
        classification_tree_tune_n_splits: Optional[int] = None,
        classification_tree_scoring: str = "accuracy",
        treatment_probability: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "TwoStepDivergenceTree":
        """
        Fit the two-step divergence tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment indicator (0 or 1).
        YF : np.ndarray of shape (n_samples,)
            Firm outcome (binary or continuous, may contain NaN).
        YC : np.ndarray of shape (n_samples,)
            Consumer outcome (binary or continuous, may contain NaN).
        auto_tune_classification_tree : bool, optional
            If True or None, automatically tunes the classification tree hyperparameters.
            If False, uses the provided `classification_tree_params`.
        classification_tree_tune_n_trials : int, optional
            Number of Optuna trials for classification tree tuning. Default: 30.
        classification_tree_tune_n_splits : int, optional
            Number of CV folds for classification tree tuning. Default: 5.
        classification_tree_scoring : str, default="accuracy"
            Scoring function for classification tree tuning. Options:
            - "accuracy": Classification accuracy (maximize)
            - "fnr_region_1", "fnr_region_2", "fnr_region_3", "fnr_region_4": 
              False Negative Rate for the specified region (minimize)
        treatment_probability : np.ndarray of shape (n_samples,), optional
            Treatment assignment probability (propensity score) for each observation.
            If provided, will be passed to CausalForestDML.fit() methods.
            If None, CausalForestDML will estimate it from the data.
        verbose : bool, default=True
            Whether to show progress output. Set to False to suppress output.

        Returns
        -------
        self : TwoStepDivergenceTree
            Returns self for method chaining.
        """
        X = np.asarray(X)
        T = np.asarray(T)
        YF = np.asarray(YF)
        YC = np.asarray(YC)

        # Input validation
        n = X.shape[0]
        if len(T) != n or len(YF) != n or len(YC) != n:
            raise ValueError(
                f"Input arrays must have matching lengths: "
                f"X={n}, T={len(T)}, YF={len(YF)}, YC={len(YC)}"
            )

        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must be in {0,1}.")

        # Validate treatment probability if provided
        if treatment_probability is not None:
            treatment_probability = np.asarray(treatment_probability)
            if treatment_probability.ndim != 1 or len(treatment_probability) != n:
                raise ValueError(
                    f"treatment_probability must be 1D array of length {n}, "
                    f"got shape {treatment_probability.shape}"
                )
            if np.any(treatment_probability < 0) or np.any(treatment_probability > 1):
                raise ValueError(
                    "treatment_probability must be between 0 and 1 for all values"
                )

        # Store fit data
        self._fit_data = dict(X=X, T=T, YF=YF, YC=YC)

        # Default to True if None
        if auto_tune_classification_tree is None:
            auto_tune_classification_tree = True

        # Step 1: Fit firm causal forest (with optional tuning)
        if verbose:
            print("Fitting causal forest for firm outcome (YF)...")
        
        # Handle NaN values for YF
        valid_F = ~np.isnan(YF)
        if valid_F.sum() < 10:
            raise ValueError("Too few valid observations for firm outcome.")

        # Create causal forest for firm outcome
        self.causal_forest_F_ = CausalForestDML(**self.causal_forest_params)
        
        # Prepare treatment probability for firm causal forest if provided
        fit_kwargs_F = {}
        tune_kwargs_F = {}
        if treatment_probability is not None:
            # Pass treatment probability to fit and tune methods
            # In econml, CausalForestDML can accept propensity scores through the W parameter
            # W is typically for confounders, but can include propensity scores
            # Reshape to (n_samples, 1) as econml expects 2D array for W
            fit_kwargs_F["W"] = treatment_probability[valid_F].reshape(-1, 1)
            tune_kwargs_F["W"] = treatment_probability[valid_F].reshape(-1, 1)
        
        # Tune if tune params are provided
        if self.causal_forest_tune_params:
            if verbose:
                print("  Tuning firm causal forest hyperparameters...")
            self.causal_forest_F_.tune(
                Y=YF[valid_F],
                T=T[valid_F],
                X=X[valid_F],
                **{**self.causal_forest_tune_params, **tune_kwargs_F}
            )
        
        # Fit firm causal forest
        self.causal_forest_F_.fit(
            Y=YF[valid_F],
            T=T[valid_F],
            X=X[valid_F],
            **fit_kwargs_F
        )

        # Step 2: Fit user causal forest (with optional tuning)
        if verbose:
            print("Fitting causal forest for consumer outcome (YC)...")
        
        # Handle NaN values for YC
        valid_C = ~np.isnan(YC)
        if valid_C.sum() < 10:
            raise ValueError("Too few valid observations for consumer outcome.")

        # Create causal forest for user outcome
        self.causal_forest_C_ = CausalForestDML(**self.causal_forest_params)
        
        # Prepare treatment probability for consumer causal forest if provided
        fit_kwargs_C = {}
        tune_kwargs_C = {}
        if treatment_probability is not None:
            # Pass treatment probability to fit and tune methods
            fit_kwargs_C["W"] = treatment_probability[valid_C].reshape(-1, 1)
            tune_kwargs_C["W"] = treatment_probability[valid_C].reshape(-1, 1)
        
        # Tune if tune params are provided
        if self.causal_forest_tune_params:
            if verbose:
                print("  Tuning user causal forest hyperparameters...")
            self.causal_forest_C_.tune(
                Y=YC[valid_C],
                T=T[valid_C],
                X=X[valid_C],
                **{**self.causal_forest_tune_params, **tune_kwargs_C}
            )
        
        # Fit user causal forest
        self.causal_forest_C_.fit(
            Y=YC[valid_C],
            T=T[valid_C],
            X=X[valid_C],
            **fit_kwargs_C
        )

        # Predict treatment effects for all observations
        if verbose:
            print("Predicting treatment effects...")
        self.tauF_ = self.causal_forest_F_.effect(X)
        self.tauC_ = self.causal_forest_C_.effect(X)

        # Step 2: Categorize observations into 4 region types
        if verbose:
            print("Categorizing observations into region types...")
        self.region_types_ = self._categorize_region_types(self.tauF_, self.tauC_)

        # Step 3: Train classification tree (with optional auto-tuning)
        if auto_tune_classification_tree:
            if verbose:
                print("Auto-tuning classification tree hyperparameters...")
            ct_params, ct_score = self._tune_classification_tree(
                X, 
                self.region_types_,
                n_trials=classification_tree_tune_n_trials,
                n_splits=classification_tree_tune_n_splits,
                scoring=classification_tree_scoring,
                verbose=verbose,
            )
            if verbose:
                if classification_tree_scoring == "accuracy":
                    print(f"  Best classification tree accuracy: {ct_score:.6f}")
                else:
                    print(f"  Best classification tree {classification_tree_scoring}: {ct_score:.6f}")
            # Update classification_tree_params with tuned values
            self.classification_tree_params.update(ct_params)
        else:
            if verbose:
                print("Training classification tree with provided parameters...")

        self.classification_tree_ = DecisionTreeClassifier(
            **self.classification_tree_params
        )
        self.classification_tree_.fit(X, self.region_types_)

        if verbose:
            print("Two-step divergence tree fitting complete!")
        return self

    def _categorize_region_types(
        self, tauF: np.ndarray, tauC: np.ndarray
    ) -> np.ndarray:
        """
        Categorize observations into 4 region types based on treatment effect signs.

        Parameters
        ----------
        tauF : np.ndarray
            Treatment effects for firm outcome.
        tauC : np.ndarray
            Treatment effects for consumer outcome.

        Returns
        -------
        np.ndarray
            Region type labels (1-4).
        """
        region_types = np.zeros(len(tauF), dtype=int)

        # Handle NaN values by treating them as 0
        tauF_clean = np.nan_to_num(tauF, nan=0.0)
        tauC_clean = np.nan_to_num(tauC, nan=0.0)

        # Region 1: both positive
        mask1 = (tauF_clean > 0) & (tauC_clean > 0)
        region_types[mask1] = 1

        # Region 2: firm positive, customer negative
        mask2 = (tauF_clean > 0) & (tauC_clean <= 0)
        region_types[mask2] = 2

        # Region 3: firm negative, customer positive
        mask3 = (tauF_clean <= 0) & (tauC_clean > 0)
        region_types[mask3] = 3

        # Region 4: both negative
        mask4 = (tauF_clean <= 0) & (tauC_clean <= 0)
        region_types[mask4] = 4

        return region_types

    def _region_type_cv_score(
        self,
        X: np.ndarray,
        region_types: np.ndarray,
        params: Dict[str, Any],
        scoring: str = "accuracy",
        n_splits: int = 5,
        random_state: Optional[int] = None,
    ) -> float:
        """
        Compute K-fold cross-validated score for region type classification.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        region_types : np.ndarray
            Region type labels (1-4).
        params : dict
            Hyperparameters for DecisionTreeClassifier.
        scoring : str, default="accuracy"
            Scoring function. Options:
            - "accuracy": Classification accuracy
            - "fnr_region_1", "fnr_region_2", "fnr_region_3", "fnr_region_4": 
              False Negative Rate for the specified region
        n_splits : int, default=5
            Number of folds for cross-validation.
        random_state : int, optional
            Random seed for KFold shuffling.

        Returns
        -------
        float
            Mean cross-validated score across all folds.
            For accuracy: higher is better (0-1 range).
            For FNR: lower is better (0-1 range), returned as negative value for minimization.
        """
        n = X.shape[0]
        if len(region_types) != n:
            raise ValueError(
                f"Input arrays must have matching lengths: X={n}, region_types={len(region_types)}"
            )

        # Parse scoring function
        if scoring == "accuracy":
            compute_fnr = False
            target_region = None
        elif scoring.startswith("fnr_region_"):
            compute_fnr = True
            try:
                target_region = int(scoring.split("_")[-1])
                if target_region not in [1, 2, 3, 4]:
                    raise ValueError(f"Invalid region: {target_region}. Must be 1, 2, 3, or 4.")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid scoring function: {scoring}. Expected 'accuracy' or 'fnr_region_X' where X is 1-4.")
        else:
            raise ValueError(f"Invalid scoring function: {scoring}. Expected 'accuracy' or 'fnr_region_X' where X is 1-4.")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, val_idx in kf.split(X):
            try:
                clf = DecisionTreeClassifier(**params)
                clf.fit(X[train_idx], region_types[train_idx])
                pred = clf.predict(X[val_idx])
                
                if compute_fnr:
                    # Compute False Negative Rate for target region
                    true_region_mask = region_types[val_idx] == target_region
                    if true_region_mask.sum() > 0:
                        # FNR = (true region but predicted as other) / (total true region)
                        fnr = (pred[true_region_mask] != target_region).sum() / true_region_mask.sum()
                        scores.append(fnr)
                    else:
                        # No samples of target region in validation fold, use 0.0 (perfect FNR)
                        scores.append(0.0)
                else:
                    # Compute accuracy
                    acc = accuracy_score(region_types[val_idx], pred)
                    scores.append(acc)
            except Exception:
                # On error, use worst possible score
                if compute_fnr:
                    scores.append(1.0)  # Worst FNR
                else:
                    scores.append(0.0)  # Worst accuracy

        mean_score = float(np.mean(scores)) if scores else (1.0 if compute_fnr else 0.0)
        
        # For FNR, return negative value so we can maximize (minimize FNR)
        if compute_fnr:
            return -mean_score
        else:
            return mean_score

    def _tune_classification_tree(
        self,
        X: np.ndarray,
        region_types: np.ndarray,
        n_trials: Optional[int] = None,
        n_splits: Optional[int] = None,
        scoring: str = "accuracy",
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters for classification tree using Optuna.

        Uses Optuna's TPE sampler to optimize the specified scoring function via
        K-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        region_types : np.ndarray
            Region type labels (1-4).
        n_trials : int, optional
            Number of Optuna trials. Default: 30.
        n_splits : int, optional
            Number of CV folds. Default: 5.
        scoring : str, default="accuracy"
            Scoring function. Options:
            - "accuracy": Classification accuracy (maximize)
            - "fnr_region_1", "fnr_region_2", "fnr_region_3", "fnr_region_4": 
              False Negative Rate for the specified region (minimize)
        verbose : bool, default=True
            Whether to show Optuna progress output.

        Returns
        -------
        best_params : dict
            Best hyperparameters found (combines fixed and tuned parameters).
        best_score : float
            Best cross-validated score (accuracy or FNR, depending on scoring).
        """
        fixed = {}
        if "random_state" in self.classification_tree_params:
            fixed["random_state"] = self.classification_tree_params["random_state"]

        # Default search space
        search_space = {}
        if "max_depth" not in self.classification_tree_params:
            search_space["max_depth"] = {"low": 2, "high": 15}
        if "min_samples_split" not in self.classification_tree_params:
            search_space["min_samples_split"] = {"low": 2, "high": 20}
        if "min_samples_leaf" not in self.classification_tree_params:
            search_space["min_samples_leaf"] = {"low": 1, "high": 10}

        if len(search_space) == 0:
            # No parameters to tune, return current params
            return dict(self.classification_tree_params), 0.0

        # Set defaults
        if n_trials is None:
            n_trials = 30
        if n_splits is None:
            n_splits = 5

        random_state = self.classification_tree_params.get("random_state")

        # Set Optuna logging verbosity
        import optuna.logging
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(fixed)

            # Handle min_samples_split and min_samples_leaf with constraint
            if "min_samples_split" in search_space:
                if "min_samples_leaf" in search_space:
                    # Both are being tuned: suggest min_samples_leaf first, then constrain min_samples_split
                    min_samples_leaf = trial.suggest_int(
                        "min_samples_leaf",
                        search_space["min_samples_leaf"]["low"],
                        search_space["min_samples_leaf"]["high"],
                    )
                    params["min_samples_leaf"] = min_samples_leaf

                    # Constrain min_samples_split to be at least 2 * min_samples_leaf
                    min_split_low = max(
                        search_space["min_samples_split"]["low"],
                        2 * min_samples_leaf,
                    )
                    if min_split_low > search_space["min_samples_split"]["high"]:
                        return 0.0
                    params["min_samples_split"] = trial.suggest_int(
                        "min_samples_split",
                        min_split_low,
                        search_space["min_samples_split"]["high"],
                    )
                else:
                    params["min_samples_split"] = trial.suggest_int(
                        "min_samples_split",
                        search_space["min_samples_split"]["low"],
                        search_space["min_samples_split"]["high"],
                    )
            elif "min_samples_leaf" in search_space:
                params["min_samples_leaf"] = trial.suggest_int(
                    "min_samples_leaf",
                    search_space["min_samples_leaf"]["low"],
                    search_space["min_samples_leaf"]["high"],
                )

            # Suggest other hyperparameters from search space
            for name, spec in search_space.items():
                if name in fixed or name in ["min_samples_split", "min_samples_leaf"]:
                    continue
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])

            # Add random_state if provided
            if random_state is not None:
                params["random_state"] = random_state

            score = self._region_type_cv_score(
                X, region_types, params, scoring=scoring, n_splits=n_splits, random_state=random_state
            )
            # Return worst possible score on error (0.0 for accuracy, 1.0 for FNR which becomes -1.0)
            if not np.isfinite(score):
                if scoring == "accuracy":
                    return 0.0
                else:
                    return -1.0
            return score

        sampler = optuna.samplers.TPESampler(seed=random_state)
        # Always maximize (FNR is returned as negative value)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        if len(study.trials) == 0 or study.best_trial is None:
            raise RuntimeError(
                "No successful trials completed for classification tree optimization."
            )

        best_params = dict(fixed)
        best_params.update(study.best_trial.params)
        if random_state is not None:
            best_params["random_state"] = random_state

        best_score = study.best_value
        # Convert back to positive FNR if using FNR scoring
        if scoring.startswith("fnr_region_"):
            best_score = -best_score

        return best_params, best_score

    def predict_region_type(self, X: np.ndarray) -> np.ndarray:
        """
        Predict region types for new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted region type labels (1-4).
        """
        if self.classification_tree_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        return self.classification_tree_.predict(X)

    def predict_treatment_effects(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict treatment effects for new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        tauF : np.ndarray of shape (n_samples,)
            Predicted treatment effects for firm outcome.
        tauC : np.ndarray of shape (n_samples,)
            Predicted treatment effects for consumer outcome.
        """
        if self.causal_forest_F_ is None or self.causal_forest_C_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        tauF = self.causal_forest_F_.effect(X)
        tauC = self.causal_forest_C_.effect(X)

        return tauF, tauC

    def get_training_treatment_effects(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the estimated treatment effects for the training data.

        Returns
        -------
        tauF : np.ndarray of shape (n_samples,)
            Estimated treatment effects for firm outcome on training data.
        tauC : np.ndarray of shape (n_samples,)
            Estimated treatment effects for consumer outcome on training data.
        """
        if self.tauF_ is None or self.tauC_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return self.tauF_, self.tauC_

    def leaf_effects(self) -> Dict[str, Any]:
        """
        Return summary of leaf effects from the classification tree.

        For each leaf in the classification tree, compute average treatment
        effects for observations in that leaf.

        Returns
        -------
        dict
            Dictionary with 'leaves' key containing list of leaf dictionaries.
            Each leaf dict has: leaf_id, region_type, tauF, tauC, n.
        """
        if self.classification_tree_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Get leaf assignments for training data
        leaf_ids = self.classification_tree_.apply(self._fit_data["X"])
        unique_leaves = np.unique(leaf_ids)

        leaves = []
        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n = mask.sum()

            # Get average treatment effects for this leaf
            tauF_leaf = np.mean(self.tauF_[mask]) if n > 0 else 0.0
            tauC_leaf = np.mean(self.tauC_[mask]) if n > 0 else 0.0

            # Get most common region type in this leaf
            region_type_leaf = (
                np.bincount(self.region_types_[mask]).argmax() if n > 0 else 0
            )

            leaves.append(
                {
                    "leaf_id": int(leaf_id),
                    "region_type": int(region_type_leaf),
                    "tauF": float(tauF_leaf),
                    "tauC": float(tauC_leaf),
                    "n": int(n),
                }
            )

        return {"leaves": leaves}
