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

import gc
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:
    # We intentionally use GRF's CausalForest (no DML wrapper) for speed and
    # because most simulation DGPs in this repo use randomized treatment.
    from econml.grf import CausalForest
except ImportError:
    raise ImportError(
        "econml is required for TwoStepDivergenceTree. "
        "Install it with: pip install econml"
    )


def _as_1d_effect(pred: Any) -> np.ndarray:
    """
    Normalize econml GRF predict/effect output to a 1D numpy array.

    econml.grf.CausalForest.predict may return:
    - array-like of shape (n,)
    - array-like of shape (n, 1)
    - a tuple where the first element is the point estimate
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred0 = pred[0]
    else:
        pred0 = pred
    arr = np.asarray(pred0)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(float, copy=False)


 


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
    Attributes
    ----------
    causal_forest_F_ : CausalForest
        Fitted causal forest for firm outcome (YF).
    causal_forest_C_ : CausalForest
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
        causal_forest_params_F: Optional[Dict[str, Any]] = None,
        causal_forest_params_C: Optional[Dict[str, Any]] = None,
        classification_tree_params: Optional[Dict[str, Any]] = None,
    ):
        base = dict(causal_forest_params or {})
        self.causal_forest_params_F = {**base, **dict(causal_forest_params_F or {})}
        self.causal_forest_params_C = {**base, **dict(causal_forest_params_C or {})}
        self.classification_tree_params = dict(classification_tree_params or {})

        self.causal_forest_params_F = self._normalize_grf_params(self.causal_forest_params_F)
        self.causal_forest_params_C = self._normalize_grf_params(self.causal_forest_params_C)

        # Set defaults for classification tree
        if "max_depth" not in self.classification_tree_params:
            self.classification_tree_params["max_depth"] = None
        if "min_samples_split" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_split"] = 2
        if "min_samples_leaf" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_leaf"] = 1

        # Will be set during fit
        self.causal_forest_F_: Optional[CausalForest] = None
        self.causal_forest_C_: Optional[CausalForest] = None
    @staticmethod
    def _normalize_grf_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply repo defaults and econml GRF constraints."""
        params = dict(params)
        params.setdefault("n_estimators", 100)
        params.setdefault("max_depth", None)
        params.setdefault("min_samples_split", 10)
        params.setdefault("min_samples_leaf", 5)
        # We only need point estimates of CATE for region labeling.
        params.setdefault("inference", False)

        # econml.grf requires: n_estimators divisible by subforest_size, and
        # subforest_size >= 2 if inference=True.
        n_estimators = int(params.get("n_estimators", 100))
        subforest_size = params.get("subforest_size", None)
        if subforest_size is not None:
            subforest_size = int(subforest_size)
            if subforest_size <= 0:
                raise ValueError("subforest_size must be positive")
            if params.get("inference", False) and subforest_size < 2:
                raise ValueError("subforest_size must be at least 2 when inference=True")
            if n_estimators % subforest_size != 0:
                raise ValueError(
                    f"n_estimators={n_estimators} must be divisible by subforest_size={subforest_size}"
                )
        else:
            # Avoid econml default subforest_size=4 constraint when using tiny n_estimators in tests.
            if n_estimators % 4 != 0:
                params["subforest_size"] = 2 if params.get("inference", False) else 1
        return params
        self.classification_tree_: Optional[DecisionTreeClassifier] = None
        self.tauF_: Optional[np.ndarray] = None
        self.tauC_: Optional[np.ndarray] = None
        self.region_types_: Optional[np.ndarray] = None
        self._fit_data: Dict[str, Any] = {}
    
    def _fit_classification_tree_step(
        self,
        X: np.ndarray,
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
        classification_tree_scoring : str, default="accuracy"
            Stored only for compatibility with downstream callers/logging.
        verbose : bool, default=True
            Whether to show progress output.
        """
        X = np.asarray(X)
        
        # Store fit data
        if "X" not in self._fit_data:
            self._fit_data["X"] = X
        
        # Use cached treatment effects when forests are dropped.
        if self.tauF_ is None or self.tauC_ is None:
            if self.causal_forest_F_ is None or self.causal_forest_C_ is None:
                raise ValueError(
                    "Treatment effects unavailable. Fit the model first so tauF_/tauC_ are computed."
                )
            if verbose:
                print("Predicting treatment effects using pre-built CausalForests...")
            self.tauF_ = _as_1d_effect(self.causal_forest_F_.predict(X))
            self.tauC_ = _as_1d_effect(self.causal_forest_C_.predict(X))
        
        # Categorize observations into 4 region types
        if verbose:
            print("Categorizing observations into region types...")
        self.region_types_ = self._categorize_region_types(self.tauF_, self.tauC_)
        
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
        fit_classification_tree: bool = True,
        classification_tree_scoring: str = "accuracy",
        drop_forests: bool = True,
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
        fit_classification_tree : bool, default=True
            If True, fit the classification tree after fitting CausalForests.
            If False, only fit CausalForests (useful when calling _fit_classification_tree_step
            separately with custom parameters).
        classification_tree_scoring : str, default="accuracy"
            Scoring function for classification tree tuning. Options:
            - "accuracy": Classification accuracy (maximize)
            - "fnr_region_1", "fnr_region_2", "fnr_region_3", "fnr_region_4": 
              False Negative Rate for the specified region (minimize)
        verbose : bool, default=True
            Whether to show progress output. Set to False to suppress output.
        drop_forests : bool, default=True
            If True, drop fitted causal forest objects after computing treatment effects
            to reduce memory usage. Set to False when you need to call
            `predict_causal_forest_effects(X_new)` on new data (e.g. validation).

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

        # Keep only X for downstream leaf summaries.
        self._fit_data = dict(X=X)

        # Step 1: Fit GRF causal forests for firm (YF) and consumer (YC).
        if verbose:
            print("Fitting causal forest for firm outcome (YF)...")
        
        # Handle NaN values for YF
        valid_F = ~np.isnan(YF)
        if valid_F.sum() < 10:
            raise ValueError("Too few valid observations for firm outcome.")

        self.causal_forest_F_ = CausalForest(**self.causal_forest_params_F)
        self.causal_forest_F_.fit(X[valid_F], T[valid_F], YF[valid_F])
        self.tauF_ = _as_1d_effect(self.causal_forest_F_.predict(X))
        if drop_forests:
            self.causal_forest_F_ = None
            gc.collect()

        # Step 2: Fit user causal forest (with optional tuning)
        if verbose:
            print("Fitting causal forest for consumer outcome (YC)...")
        
        # Handle NaN values for YC
        valid_C = ~np.isnan(YC)
        if valid_C.sum() < 10:
            raise ValueError("Too few valid observations for consumer outcome.")

        self.causal_forest_C_ = CausalForest(**self.causal_forest_params_C)
        self.causal_forest_C_.fit(X[valid_C], T[valid_C], YC[valid_C])
        self.tauC_ = _as_1d_effect(self.causal_forest_C_.predict(X))
        if drop_forests:
            self.causal_forest_C_ = None
            gc.collect()

        if fit_classification_tree:
            # Step 2: Categorize observations into 4 region types
            if verbose:
                print("Categorizing observations into region types...")
            self.region_types_ = self._categorize_region_types(self.tauF_, self.tauC_)

            # Step 3: Train classification tree (with optional auto-tuning)
            if verbose:
                print("Training classification tree with provided parameters...")

            self.classification_tree_ = DecisionTreeClassifier(
                **self.classification_tree_params
            )
            self.classification_tree_.fit(X, self.region_types_)

        if verbose:
            print("Two-step divergence tree fitting complete!")
        return self

    def predict_causal_forest_effects(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict treatment effects (tauF, tauC) for new observations using fitted forests.

        Requires `fit(..., drop_forests=False)` so the forest objects are retained.
        """
        if self.causal_forest_F_ is None or self.causal_forest_C_ is None:
            raise ValueError(
                "CausalForest models are not available. Fit with drop_forests=False first."
            )
        X = np.asarray(X)
        tauF = _as_1d_effect(self.causal_forest_F_.predict(X))
        tauC = _as_1d_effect(self.causal_forest_C_.predict(X))
        return tauF, tauC

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
        raise ValueError(
            "CausalForest models are dropped after fit to save memory. "
            "Use get_training_treatment_effects() for training data effects."
        )

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
