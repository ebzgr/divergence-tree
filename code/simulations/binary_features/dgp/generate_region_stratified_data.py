"""
Region-stratified synthetic data generator (v3) for binary comparison simulation.

v3 changes:
- Single aspect `rareness_region2` controls both region-2 combination share and
  region-2 observation share.
- Remaining regions (1, 3, 4) are split equally (reduced variance).
- Outcome noise is allowed and used; treatment-effect noise is optional.
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


def _total_combinations(n_categories: List[int]) -> int:
    total = 1
    for c in n_categories:
        total *= int(c)
    return int(total)


def _sample_unique_combinations(
    n_categories: List[int],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """
    Sample unique categorical combinations without enumerating the full cartesian product.

    This is critical when the combination space is enormous (e.g., 2^30).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    k = len(n_categories)
    target = int(n_samples)
    seen: set[Tuple[int, ...]] = set()
    # Safety to avoid infinite loops when total space is small.
    max_tries = max(10_000, target * 50)
    tries = 0
    while len(seen) < target and tries < max_tries:
        tries += 1
        combo = tuple(int(rng.integers(0, n_categories[j])) for j in range(k))
        seen.add(combo)
    if len(seen) < target:
        # Fallback: enumerate (space is likely small enough if uniqueness is failing).
        all_combos = _generate_all_combinations(n_categories)
        rng.shuffle(all_combos)
        return all_combos[:target]
    return list(seen)


def generate_region_stratified_data(
    n_users: int,
    k: int,
    n_categories: List[int],
    rareness_region2: float,
    intensity: float = 1.0,
    effect_noise_std: float = 0.0,
    firm_outcome_noise_std: float = 0.1,
    user_outcome_noise_std: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate data by explicitly assigning combinations and observations to 4 regions.
    """
    if n_users < 1:
        raise ValueError("n_users must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(n_categories) != k:
        raise ValueError(f"n_categories must have length {k}")
    if any(c < 2 for c in n_categories):
        raise ValueError("Each categorical variable must have at least 2 categories")
    if not (0.0 < rareness_region2 < 1.0):
        raise ValueError("rareness_region2 must be in (0,1)")
    if intensity <= 0:
        raise ValueError("intensity must be > 0")
    if effect_noise_std < 0:
        raise ValueError("effect_noise_std must be >= 0")
    if firm_outcome_noise_std < 0 or user_outcome_noise_std < 0:
        raise ValueError("outcome noise std values must be >= 0")

    rng = np.random.default_rng(random_seed)
    total_combinations = _total_combinations(n_categories)
    if total_combinations < 4:
        raise ValueError("Need at least 4 combinations to assign all four regions.")

    # If the full combination space is huge, sample a manageable pool of combinations.
    # This makes k=30 feasible while keeping the region-allocation logic intact.
    max_combo_pool = 1000
    if total_combinations > max_combo_pool:
        all_combinations = _sample_unique_combinations(
            n_categories, n_samples=max_combo_pool, rng=rng
        )
        total_combinations_for_allocation = len(all_combinations)
    else:
        all_combinations = _generate_all_combinations(n_categories)
        total_combinations_for_allocation = len(all_combinations)

    # Region 2 fixed by rareness, all others equal.
    p2 = float(rareness_region2)
    p_other = (1.0 - p2) / 3.0
    combo_probs = np.array([p_other, p2, p_other, p_other], dtype=float)
    combo_counts = _largest_remainder_counts(total_combinations_for_allocation, combo_probs)
    for i in range(4):
        if combo_counts[i] == 0:
            j = int(np.argmax(combo_counts))
            if combo_counts[j] <= 1:
                raise ValueError("Unable to allocate at least one combination per region.")
            combo_counts[j] -= 1
            combo_counts[i] += 1

    perm = rng.permutation(total_combinations_for_allocation)
    s1_idx = perm[: combo_counts[0]]
    s2_idx = perm[combo_counts[0] : combo_counts[0] + combo_counts[1]]
    s3_idx = perm[combo_counts[0] + combo_counts[1] : combo_counts[0] + combo_counts[1] + combo_counts[2]]
    s4_idx = perm[combo_counts[0] + combo_counts[1] + combo_counts[2] :]

    region_combo_sets = {
        1: [all_combinations[int(i)] for i in s1_idx],
        2: [all_combinations[int(i)] for i in s2_idx],
        3: [all_combinations[int(i)] for i in s3_idx],
        4: [all_combinations[int(i)] for i in s4_idx],
    }

    obs_probs = combo_probs.copy()
    obs_counts = _largest_remainder_counts(n_users, obs_probs)
    if n_users >= 4:
        for i in range(4):
            if obs_counts[i] == 0:
                j = int(np.argmax(obs_counts))
                if obs_counts[j] <= 1:
                    break
                obs_counts[j] -= 1
                obs_counts[i] += 1

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

    obs_perm = rng.permutation(n_users)
    categorical_data = categorical_data[obs_perm]
    region_type = region_type[obs_perm]

    n_features = sum(n_categories)
    X = np.zeros((n_users, n_features), dtype=int)
    offset = 0
    for j in range(k):
        for i in range(n_users):
            X[i, offset + categorical_data[i, j]] = 1
        offset += n_categories[j]

    baseline_coef_F = rng.standard_normal(n_features)
    baseline_coef_C = rng.standard_normal(n_features)
    baseline_F = X @ baseline_coef_F
    baseline_C = X @ baseline_coef_C

    tauF_clean = np.where(np.isin(region_type, [1, 2]), intensity, -intensity).astype(float)
    tauC_clean = np.where(np.isin(region_type, [1, 3]), intensity, -intensity).astype(float)

    if effect_noise_std > 0:
        tauF = tauF_clean + rng.normal(0.0, effect_noise_std, size=n_users)
        tauC = tauC_clean + rng.normal(0.0, effect_noise_std, size=n_users)
    else:
        tauF = tauF_clean.copy()
        tauC = tauC_clean.copy()

    # v3: noise affects outcomes directly.
    eps_F = rng.normal(0.0, firm_outcome_noise_std, size=n_users) if firm_outcome_noise_std > 0 else 0.0
    eps_C = rng.normal(0.0, user_outcome_noise_std, size=n_users) if user_outcome_noise_std > 0 else 0.0
    T = rng.binomial(1, 0.5, size=n_users)
    YF = baseline_F + tauF * T + eps_F
    YC = baseline_C + tauC * T + eps_C

    # Share is with respect to the allocation pool when sampling is used.
    combo_share_by_region = {
        r: float(len(region_combo_sets[r]) / total_combinations_for_allocation) for r in [1, 2, 3, 4]
    }
    obs_share_by_region = {
        r: float((region_type == r).mean()) for r in [1, 2, 3, 4]
    }

    functional_form: Dict[str, Any] = {
        "k": k,
        "n_categories": list(n_categories),
        "intensity": float(intensity),
        "effect_noise_std": float(effect_noise_std),
        "firm_outcome_noise_std": float(firm_outcome_noise_std),
        "user_outcome_noise_std": float(user_outcome_noise_std),
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
        "combo_share_by_region": combo_share_by_region,
        "obs_share_by_region": obs_share_by_region,
        "combo_counts_by_region": {r: int(len(region_combo_sets[r])) for r in [1, 2, 3, 4]},
        "obs_counts_by_region": {r: int((region_type == r).sum()) for r in [1, 2, 3, 4]},
        "combo_allocation_total_combinations": int(total_combinations),
        "combo_allocation_pool_size": int(total_combinations_for_allocation),
        "region_combinations": {
            "region_1": region_combo_sets[1],
            "region_2": region_combo_sets[2],
            "region_3": region_combo_sets[3],
            "region_4": region_combo_sets[4],
        },
        "baseline_coef_F": baseline_coef_F,
        "baseline_coef_C": baseline_coef_C,
    }

    return X, T, YF, YC, tauF, tauC, region_type, functional_form

