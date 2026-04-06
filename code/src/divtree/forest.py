"""
DivTree Forest: bagging of DivergenceTree with a classification tree step.

Fits multiple DivTrees on bootstrap samples (and optional feature subsamples),
aggregates (tauF, tauC) per observation, categorizes into region types,
then fits a single classification tree for final prediction.
Mirrors the CausalForest + TwoStep structure for apple-to-apple comparison.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.tree import DecisionTreeClassifier

from .tree import DivergenceTree


def _categorize_region_types(tauF: np.ndarray, tauC: np.ndarray) -> np.ndarray:
    """Categorize observations into 4 region types based on treatment effect signs."""
    region_types = np.zeros(len(tauF), dtype=int)
    tauF_clean = np.nan_to_num(tauF, nan=0.0)
    tauC_clean = np.nan_to_num(tauC, nan=0.0)
    region_types[(tauF_clean > 0) & (tauC_clean > 0)] = 1
    region_types[(tauF_clean > 0) & (tauC_clean <= 0)] = 2
    region_types[(tauF_clean <= 0) & (tauC_clean > 0)] = 3
    region_types[(tauF_clean <= 0) & (tauC_clean <= 0)] = 4
    return region_types


class DivTreeForest:
    """
    Forest of DivergenceTrees with a classification tree for region prediction.

    Step 1: Fit n_estimators DivTrees, each on a bootstrap sample (and optional
    feature subsample). Step 2: Average (tauF, tauC) per observation, categorize
    into region types, fit a DecisionTreeClassifier. Predictions use the
    classification tree.

    Parameters
    ----------
    n_estimators : int
        Number of DivTrees in the forest.
    lambda_ : float
        Lambda parameter for each DivTree.
    regions_of_interest : list of int, optional
        Region IDs to focus on (e.g. [2]). None means all regions.
    max_samples : float or int, default=0.63
        Fraction (float) or count (int) of samples to draw per tree (with replacement).
    max_features : float or int, default=1.0
        Fraction (float) or count (int) of features to draw per tree. 1.0 = use all.
    random_state : int, optional
        Random seed for bootstrap and feature subsampling.
    **divtree_params
        Passed to each DivergenceTree (e.g. max_partitions, min_improvement_ratio,
        n_quantiles, eps_scale).
    """

    def __init__(
        self,
        n_estimators: int,
        lambda_: float,
        regions_of_interest: Optional[List[int]] = None,
        max_samples: float = 0.63,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
        **divtree_params: Any,
    ):
        self.n_estimators = int(n_estimators)
        self.lambda_ = float(lambda_)
        self.regions_of_interest = regions_of_interest
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.divtree_params = dict(divtree_params)

        self.estimators_: List[DivergenceTree] = []
        self.feature_indices_: List[np.ndarray] = []  # column indices used per tree
        self.classification_tree_: Optional[DecisionTreeClassifier] = None
        self._rng: Optional[np.random.Generator] = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)
        return self._rng

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        YF: np.ndarray,
        YC: np.ndarray,
    ) -> "DivTreeForest":
        """
        Fit the forest and the classification tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment indicator (0 or 1).
        YF : np.ndarray of shape (n_samples,)
            Firm outcome.
        YC : np.ndarray of shape (n_samples,)
            Consumer outcome.

        Returns
        -------
        self : DivTreeForest
        """
        X = np.asarray(X)
        T = np.asarray(T)
        YF = np.asarray(YF)
        YC = np.asarray(YC)
        n, p = X.shape

        if len(T) != n or len(YF) != n or len(YC) != n:
            raise ValueError("X, T, YF, YC must have matching lengths.")
        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must be in {0,1}.")

        rng = self._get_rng()

        n_samples = (
            int(self.max_samples * n)
            if isinstance(self.max_samples, float)
            else min(int(self.max_samples), n)
        )
        n_samples = max(2, n_samples)

        n_features = (
            max(1, int(self.max_features * p))
            if isinstance(self.max_features, float)
            else min(int(self.max_features), p)
        )

        self.estimators_ = []
        self.feature_indices_ = []

        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n_samples)
            feat_idx = (
                rng.choice(p, size=n_features, replace=False)
                if n_features < p
                else np.arange(p)
            )
            feat_idx = np.sort(feat_idx)

            X_sub = X[np.ix_(boot_idx, feat_idx)]
            T_sub = T[boot_idx]
            YF_sub = YF[boot_idx]
            YC_sub = YC[boot_idx]

            tree = DivergenceTree(
                lambda_=self.lambda_,
                regions_of_interest=self.regions_of_interest,
                random_state=self.random_state,
                **self.divtree_params,
            )
            tree.fit(X_sub, T_sub, YF_sub, YC_sub)
            self.estimators_.append(tree)
            self.feature_indices_.append(feat_idx)

        # Aggregate (tauF, tauC) on full training data
        tauF_list = []
        tauC_list = []
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            X_f = X[:, feat_idx]
            tauF, tauC = tree.predict_treatment_effects(X_f)
            tauF_list.append(tauF)
            tauC_list.append(tauC)

        tauF_agg = np.mean(tauF_list, axis=0)
        tauC_agg = np.mean(tauC_list, axis=0)
        region_types = _categorize_region_types(tauF_agg, tauC_agg)

        # Fit classification tree
        self.classification_tree_ = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        self.classification_tree_.fit(X, region_types)

        return self

    def predict_region_type(self, X: np.ndarray) -> np.ndarray:
        """
        Predict region types (1-4) for new observations.

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
        return self.classification_tree_.predict(X).astype(int)
