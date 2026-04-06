"""
Region-stratified synthetic data generator for binary comparison simulation.

Key properties:
- Region 2 combination share (dispersion) is fixed by input.
- Region 2 observation share (rareness) is fixed by input.
- Regions 1/3/4 split only the remaining mass.
- Outcome noise is intentionally zero; only treatment-effect noise is used.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _generate_all_combinations(n_categories: List[int]) -> List[Tuple[int, ...]]:
    ranges = [range(n) for n in n_categories]
    return [tuple(combo) for combo in product(*ranges)]


def _largest_remainder_counts(total: int, probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if total < 0:
        raise ValueError("total must be non-negative")
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("probs must be a non-empty 1D array")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    s = probs.sum()
    if s <= 0:
        raise ValueError("probs sum must be > 0")
    probs = probs / s
    raw = probs * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def generate_region_stratified_data(
    n_users: int,
    k: int,
    n_categories: List[int],
    dispersion_region2: float,
    rareness_region2: float,
    similarity_unused: Optional[float] = None,
    intensity: float = 1.0,
    effect_noise_std: float = 0.1,
    firm_outcome_noise_std: float = 0.0,
    user_outcome_noise_std: float = 0.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate data by explicitly assigning combinations and observations to 4 regions.

    Parameters
    ----------
    n_users : int
        Number of observations.
    k : int
        Number of categorical variables.
    n_categories : list[int]
        Category count per variable.
    dispersion_region2 : float
        Target fraction of combinations in region 2.
    rareness_region2 : float
        Target fraction of observations in region 2 combinations.
    similarity_unused : Optional[float]
        Ignored placeholder kept for API compatibility.
    intensity : float, default=1.0
        Base treatment-effect magnitude.
    effect_noise_std : float, default=0.1
        Std of noise added to treatment effects.
    firm_outcome_noise_std : float, default=0.0
        Must be 0 for this DGP variant.
    user_outcome_noise_std : float, default=0.0
        Must be 0 for this DGP variant.
    random_seed : Optional[int], default=None
        RNG seed.
    """
    if n_users < 1:
        raise ValueError("n_users must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(n_categories) != k:
        raise ValueError(f"n_categories must have length {k}")
    if any(c < 2 for c in n_categories):
        raise ValueError("Each categorical variable must have at least 2 categories")
    if not (0.0 < dispersion_region2 < 1.0):
        raise ValueError("dispersion_region2 must be in (0,1)")
    if not (0.0 < rareness_region2 < 1.0):
        raise ValueError("rareness_region2 must be in (0,1)")
    if intensity <= 0:
        raise ValueError("intensity must be > 0")
    if effect_noise_std < 0:
        raise ValueError("effect_noise_std must be >= 0")
    if firm_outcome_noise_std != 0.0 or user_outcome_noise_std != 0.0:
        raise ValueError("This DGP requires zero outcome noise (firm/user outcome noise must be 0).")

    rng = np.random.default_rng(random_seed)
    all_combinations = _generate_all_combinations(n_categories)
    total_combinations = len(all_combinations)
    if total_combinations < 4:
        raise ValueError("Need at least 4 combinations to assign all four regions.")

    # Combination-level target shares (p2 fixed exactly at probability level).
    q = rng.dirichlet(np.ones(3))
    p2 = float(dispersion_region2)
    p_rest = (1.0 - p2) * q
    combo_probs = np.array([p_rest[0], p2, p_rest[1], p_rest[2]], dtype=float)
    combo_counts = _largest_remainder_counts(total_combinations, combo_probs)
    # Ensure each region has at least one combination.
    for i in range(4):
        if combo_counts[i] == 0:
            j = int(np.argmax(combo_counts))
            if combo_counts[j] <= 1:
                raise ValueError("Unable to allocate at least one combination per region.")
            combo_counts[j] -= 1
            combo_counts[i] += 1

    perm = rng.permutation(total_combinations)
    s1_idx = perm[: combo_counts[0]]
    s2_idx = perm[combo_counts[0] : combo_counts[0] + combo_counts[1]]
    s3_idx = perm[combo_counts[0] + combo_counts[1] : combo_counts[0] + combo_counts[1] + combo_counts[2]]
    s4_idx = perm[combo_counts[0] + combo_counts[1] + combo_counts[2] :]

    region_combo_sets = {
        1: [all_combinations[i] for i in s1_idx],
        2: [all_combinations[i] for i in s2_idx],
        3: [all_combinations[i] for i in s3_idx],
        4: [all_combinations[i] for i in s4_idx],
    }

    # Observation-level target shares (w2 fixed exactly at probability level).
    q_obs = rng.dirichlet(np.ones(3))
    w2 = float(rareness_region2)
    w_rest = (1.0 - w2) * q_obs
    obs_probs = np.array([w_rest[0], w2, w_rest[1], w_rest[2]], dtype=float)
    obs_counts = _largest_remainder_counts(n_users, obs_probs)
    # Ensure each region has at least one observation where possible.
    if n_users >= 4:
        for i in range(4):
            if obs_counts[i] == 0:
                j = int(np.argmax(obs_counts))
                if obs_counts[j] <= 1:
                    break
                obs_counts[j] -= 1
                obs_counts[i] += 1

    # Allocate observations to combinations by region.
    categorical_data = np.zeros((n_users, k), dtype=int)
    region_type = np.zeros(n_users, dtype=int)
    start = 0
    for region_id in [1, 2, 3, 4]:
        n_r = int(obs_counts[region_id - 1])
        combos_r = region_combo_sets[region_id]
        chosen_idx = rng.choice(len(combos_r), size=n_r, replace=True)
        for local_i, combo_idx in enumerate(chosen_idx):
            global_i = start + local_i
            categorical_data[global_i, :] = np.asarray(combos_r[combo_idx], dtype=int)
            region_type[global_i] = region_id
        start += n_r

    # Shuffle observation order.
    obs_perm = rng.permutation(n_users)
    categorical_data = categorical_data[obs_perm]
    region_type = region_type[obs_perm]

    # One-hot X.
    n_features = sum(n_categories)
    X = np.zeros((n_users, n_features), dtype=int)
    offset = 0
    for j in range(k):
        for i in range(n_users):
            X[i, offset + categorical_data[i, j]] = 1
        offset += n_categories[j]

    # Baselines.
    baseline_coef_F = rng.standard_normal(n_features)
    baseline_coef_C = rng.standard_normal(n_features)
    baseline_F = X @ baseline_coef_F
    baseline_C = X @ baseline_coef_C

    # Clean treatment effects by region design.
    tauF_clean = np.where(np.isin(region_type, [1, 2]), intensity, -intensity).astype(float)
    tauC_clean = np.where(np.isin(region_type, [1, 3]), intensity, -intensity).astype(float)

    # Add treatment-effect noise only.
    if effect_noise_std > 0:
        tauF = tauF_clean + rng.normal(0.0, effect_noise_std, size=n_users)
        tauC = tauC_clean + rng.normal(0.0, effect_noise_std, size=n_users)
    else:
        tauF = tauF_clean.copy()
        tauC = tauC_clean.copy()

    # Outcomes with zero additive outcome noise.
    T = rng.binomial(1, 0.5, size=n_users)
    YF = baseline_F + tauF * T
    YC = baseline_C + tauC * T

    dispersion_by_region = {
        r: float(len(region_combo_sets[r]) / total_combinations) for r in [1, 2, 3, 4]
    }
    rareness_by_region = {
        r: float((region_type == r).mean()) for r in [1, 2, 3, 4]
    }

    functional_form: Dict[str, Any] = {
        "k": k,
        "n_categories": list(n_categories),
        "intensity": float(intensity),
        "effect_noise_std": float(effect_noise_std),
        "firm_outcome_noise_std": 0.0,
        "user_outcome_noise_std": 0.0,
        "dispersion_region2": float(dispersion_region2),
        "rareness_region2": float(rareness_region2),
        "combo_target_by_region": {
            1: float(combo_probs[0]),
            2: float(combo_probs[1]),
            3: float(combo_probs[2]),
            4: float(combo_probs[3]),
        },
        "obs_target_by_region": {
            1: float(obs_probs[0]),
            2: float(obs_probs[1]),
            3: float(obs_probs[2]),
            4: float(obs_probs[3]),
        },
        "dispersion_by_region": dispersion_by_region,
        "rareness_by_region": rareness_by_region,
        "combo_counts_by_region": {r: int(len(region_combo_sets[r])) for r in [1, 2, 3, 4]},
        "obs_counts_by_region": {r: int((region_type == r).sum()) for r in [1, 2, 3, 4]},
        "region_combinations": {
            "region_1": region_combo_sets[1],
            "region_2": region_combo_sets[2],
            "region_3": region_combo_sets[3],
            "region_4": region_combo_sets[4],
        },
        "baseline_coef_F": baseline_coef_F,
        "baseline_coef_C": baseline_coef_C,
        "random_seed": random_seed,
    }

    return X, T, YF, YC, tauF, tauC, region_type, functional_form

