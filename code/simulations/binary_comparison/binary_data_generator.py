"""
Binary Data Generator for Comparison Simulation

Generates synthetic data with binary features (one-hot encoded categorical variables):
- k categorical variables with specified number of categories each
- Treatment effects activated only for specific combinations of categories
- Half of activated combinations have positive sign, half have negative sign
- Inactive combinations get small random effects (mean 0, configurable variance)
- Standard DGP structure: Y = μ(X) + τ(X) × T + ε
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.stats import norm


def generate_binary_comparison_data(
    n_users: int = 10000,
    k: int = 3,
    n_categories: List[int] = [3, 4, 5],
    m_firm: int = 5,
    m_user: Optional[int] = None,
    similarity: float = 0.5,
    intensity: float = 2.0,
    effect_noise_std: float = 0.1,
    firm_outcome_noise_std: float = 1.0,
    user_outcome_noise_std: float = 1.0,
    positive_ratio: float = 0.5,
    random_seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Generate synthetic data with binary features and categorical activation patterns.

    Follows standard data generating process: Y = μ(X) + τ(X) × T + ε
    where:
    - μ(X) is baseline outcome function (linear combination of binary features)
    - τ(X) is treatment effect function (activated for specific category combinations)
    - T is treatment indicator (random assignment with probability 0.5)
    - ε is outcome noise

    Parameters
    ----------
    n_users : int, default=10000
        Number of observations to generate.
    k : int, default=3
        Number of categorical variables.
    n_categories : list of int, default=[3, 4, 5]
        List of k integers, where n_categories[i] = number of categories in i-th variable.
    m_firm : int, default=5
        Number of combinations that activate firm treatment effect.
    m_user : int, optional
        Number of combinations that activate user treatment effect.
        If None, uses m_firm.
    similarity : float, default=0.5
        Proportion of combinations shared between firm and user effects.
        Must be in [0, 1]. For example, if m_firm=4 and similarity=0.5,
        then 2 combinations are shared.
    intensity : float, default=2.0
        Treatment effect intensity. Activated combinations get +intensity,
        non-activated combinations get -intensity.
    effect_noise_std : float, default=0.1
        Standard deviation of noise added to all treatment effects (mean=0).
    firm_outcome_noise_std : float, default=1.0
        Standard deviation of noise for firm outcomes.
    user_outcome_noise_std : float, default=1.0
        Standard deviation of noise for user outcomes.
    positive_ratio : float, default=0.5
        Minimum proportion of observations in activating combinations that should
        have positive treatment effects (after noise is added). Must be in [0, 1].
        If noise causes some effects to become negative, the base intensity will
        be adjusted to ensure at least this ratio remains positive.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_users, sum(n_categories))
        Binary feature matrix (one-hot encoded).
    T : np.ndarray of shape (n_users,)
        Treatment assignment (binary 0/1, random with probability 0.5).
    YF : np.ndarray of shape (n_users,)
        Firm outcome (continuous, always observed).
    YC : np.ndarray of shape (n_users,)
        User outcome (continuous, always observed).
    tauF : np.ndarray of shape (n_users,)
        Firm treatment effects.
    tauC : np.ndarray of shape (n_users,)
        User treatment effects.
    region_type : np.ndarray of shape (n_users,)
        Region type labels (1-4):
        - 1: both positive (tauF > 0, tauC > 0)
        - 2: firm+ customer- (tauF > 0, tauC <= 0)
        - 3: firm- customer+ (tauF <= 0, tauC > 0)
        - 4: both negative (tauF <= 0, tauC <= 0)
    functional_form : dict
        Dictionary containing the functional form information:
        - 'baseline_coef_F': Baseline coefficients for firm outcome (all binary features)
        - 'baseline_coef_C': Baseline coefficients for user outcome (all binary features)
        - 'firm_combinations': List of tuples (combinations) for firm effects
        - 'user_combinations': List of tuples (combinations) for user effects
        - 'firm_signs': List of signs (always +1, kept for backward compatibility)
        - 'user_signs': List of signs (always +1, kept for backward compatibility)
        - 'k': Number of categorical variables
        - 'n_categories': List of category counts
        - 'm_firm': Number of firm combinations
        - 'm_user': Number of user combinations
        - 'similarity': Similarity parameter used
        - 'intensity': Intensity parameter used
        - 'effect_noise_std': Effect noise std parameter used
    """
    # Input validation
    if n_users < 1:
        raise ValueError("n_users must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(n_categories) != k:
        raise ValueError(
            f"n_categories must have length k={k}, got {len(n_categories)}"
        )
    if any(n < 2 for n in n_categories):
        raise ValueError("Each categorical variable must have at least 2 categories")
    if m_firm < 1:
        raise ValueError("m_firm must be >= 1")
    if m_user is None:
        m_user = m_firm
    if m_user < 1:
        raise ValueError("m_user must be >= 1")
    if not (0 <= similarity <= 1):
        raise ValueError("similarity must be in [0, 1]")
    if intensity <= 0:
        raise ValueError("intensity must be > 0")
    if effect_noise_std < 0:
        raise ValueError("effect_noise_std must be >= 0")
    if not (0 <= positive_ratio <= 1):
        raise ValueError("positive_ratio must be in [0, 1]")

    # Check that we have enough possible combinations
    total_combinations = np.prod(n_categories)
    if m_firm > total_combinations:
        raise ValueError(
            f"m_firm={m_firm} exceeds total possible combinations={total_combinations}"
        )
    if m_user > total_combinations:
        raise ValueError(
            f"m_user={m_user} exceeds total possible combinations={total_combinations}"
        )

    # Set random seed
    rng = np.random.default_rng(random_seed)

    # Step 1: Generate activation combinations FIRST
    # This must be done before generating observations so we can force some observations into combinations
    # Generate all possible combinations
    all_combinations = []
    for combo in _generate_all_combinations(n_categories):
        all_combinations.append(tuple(combo))

    # Randomly select m_firm combinations for firm effects
    firm_combinations = rng.choice(len(all_combinations), size=m_firm, replace=False)
    firm_combinations = [all_combinations[i] for i in firm_combinations]

    # All activated combinations get positive sign (for backward compatibility)
    firm_signs = np.ones(m_firm, dtype=int)

    # Handle similarity: share some combinations with user
    n_shared = int(np.round(m_user * similarity))
    if n_shared > 0:
        # Share first n_shared combinations (with their signs)
        shared_combinations = firm_combinations[: min(n_shared, len(firm_combinations))]
        shared_signs = firm_signs[: min(n_shared, len(firm_signs))]

        # Generate remaining unique combinations for user
        remaining_combinations = [
            combo for combo in all_combinations if combo not in firm_combinations
        ]
        n_unique_user = m_user - len(shared_combinations)

        if n_unique_user > 0:
            if len(remaining_combinations) < n_unique_user:
                raise ValueError(
                    f"Not enough remaining combinations. "
                    f"Need {n_unique_user}, have {len(remaining_combinations)}"
                )
            unique_indices = rng.choice(
                len(remaining_combinations), size=n_unique_user, replace=False
            )
            unique_combinations = [remaining_combinations[i] for i in unique_indices]

            user_combinations = shared_combinations + unique_combinations
        else:
            user_combinations = shared_combinations

        # Assign signs to user combinations (all positive for activated)
        user_signs = list(shared_signs)
        n_unique_user = len(user_combinations) - len(shared_combinations)
        if n_unique_user > 0:
            unique_signs = np.ones(n_unique_user, dtype=int)
            user_signs.extend(unique_signs.tolist())
        user_signs = np.array(user_signs)
    else:
        # No overlap, generate all unique combinations for user
        remaining_combinations = [
            combo for combo in all_combinations if combo not in firm_combinations
        ]
        if len(remaining_combinations) < m_user:
            raise ValueError(
                f"Not enough remaining combinations. "
                f"Need {m_user}, have {len(remaining_combinations)}"
            )
        unique_indices = rng.choice(
            len(remaining_combinations), size=m_user, replace=False
        )
        user_combinations = [remaining_combinations[i] for i in unique_indices]

        # All activated combinations get positive sign
        user_signs = np.ones(m_user, dtype=int)

    # Step 2: Generate categorical data
    # First, force positive_ratio of observations to be in activating combinations
    # Then force the rest to be in non-activating combinations (negative regions)
    categorical_data = np.zeros((n_users, k), dtype=int)

    # Calculate how many observations should be forced into activating combinations
    forced_obs = int(np.ceil(positive_ratio * n_users))

    # Collect all unique activating combinations (union of firm and user)
    all_activating_combinations = list(set(firm_combinations + user_combinations))

    # For the forced observations, randomly assign each to one of the activating combinations
    if forced_obs > 0 and len(all_activating_combinations) > 0:
        # Randomly select which activating combination each forced observation belongs to
        selected_combo_indices = rng.choice(
            len(all_activating_combinations), size=forced_obs, replace=True
        )
        for i, combo_idx in enumerate(selected_combo_indices):
            if i < n_users:
                combo = all_activating_combinations[combo_idx]
                categorical_data[i, :] = np.array(combo)

    # For the remaining observations, force them into non-activating combinations
    if forced_obs < n_users:
        # Reuse all_combinations already generated in Step 1
        # Identify non-activating combinations (all combinations NOT in activating set)
        non_activating_combinations = [
            combo for combo in all_combinations 
            if combo not in all_activating_combinations
        ]
        
        if len(non_activating_combinations) == 0:
            raise ValueError(
                "No non-activating combinations available. "
                "All possible combinations are activating. "
                "This can happen if m_firm + m_user >= total_combinations."
            )
        
        # Randomly assign remaining observations to non-activating combinations
        n_remaining = n_users - forced_obs
        selected_combo_indices = rng.choice(
            len(non_activating_combinations), size=n_remaining, replace=True
        )
        for i, combo_idx in enumerate(selected_combo_indices):
            obs_idx = forced_obs + i
            if obs_idx < n_users:
                combo = non_activating_combinations[combo_idx]
                categorical_data[obs_idx, :] = np.array(combo)

    # Step 3: Create binary feature matrix (one-hot encoded)
    n_features = sum(n_categories)
    X = np.zeros((n_users, n_features), dtype=int)

    feature_offset = 0
    for i in range(k):
        # For each observation, set the corresponding category to 1
        for j in range(n_users):
            category_idx = categorical_data[j, i]
            X[j, feature_offset + category_idx] = 1
        feature_offset += n_categories[i]

    # Step 4: Generate baseline outcomes μ(X)
    # Random coefficients for all binary features
    baseline_coef_F = rng.standard_normal(n_features)
    baseline_coef_C = rng.standard_normal(n_features)

    baseline_F = X @ baseline_coef_F
    baseline_C = X @ baseline_coef_C

    # Step 5: Generate treatment effects τ(X)
    # For each observation, check if it matches any activation combination
    # Activated combinations get +intensity, non-activated get -intensity
    # Use float dtype to allow multiplication by float adjustment_factor later
    tauF_clean = np.full(n_users, -float(intensity), dtype=float)  # Start with -intensity for all
    tauC_clean = np.full(n_users, -float(intensity), dtype=float)  # Start with -intensity for all

    # Convert categorical_data to list of tuples for easy comparison
    obs_combinations = [tuple(categorical_data[i, :]) for i in range(n_users)]

    # Firm treatment effects: set activated combinations to +intensity
    firm_activated_mask = np.zeros(n_users, dtype=bool)
    for i, combo in enumerate(firm_combinations):
        matches = [obs_combo == combo for obs_combo in obs_combinations]
        matches = np.array(matches)
        if matches.any():
            tauF_clean[matches] = float(intensity)
            firm_activated_mask[matches] = True

    # User treatment effects: set activated combinations to +intensity
    user_activated_mask = np.zeros(n_users, dtype=bool)
    for i, combo in enumerate(user_combinations):
        matches = [obs_combo == combo for obs_combo in obs_combinations]
        matches = np.array(matches)
        if matches.any():
            tauC_clean[matches] = float(intensity)
            user_activated_mask[matches] = True

    # Step 5a: Compute region type based on CLEAN treatment effects (before noise)
    # This ensures region assignment is based on true underlying effects, not noisy versions
    region_type = np.zeros(n_users, dtype=int)
    both_positive = (tauF_clean > 0) & (tauC_clean > 0)
    firm_pos_cust_neg = (tauF_clean > 0) & (tauC_clean <= 0)
    firm_neg_cust_pos = (tauF_clean <= 0) & (tauC_clean > 0)
    both_negative = (tauF_clean <= 0) & (tauC_clean <= 0)

    region_type[both_positive] = 1
    region_type[firm_pos_cust_neg] = 2
    region_type[firm_neg_cust_pos] = 3
    region_type[both_negative] = 4

    # Step 5b: Add noise to treatment effects AFTER region assignment
    # This noise affects outcome generation but not region type assignment
    # Adjust intensity to ensure at least positive_ratio of activated observations remain positive
    if effect_noise_std > 0:
        # Calculate required intensity to ensure positive_ratio remains positive
        # For a normal distribution, P(X > 0) where X ~ N(mu, sigma^2) = Phi(mu/sigma)
        # We want P(X > 0) >= positive_ratio, so mu/sigma >= Phi^{-1}(positive_ratio)

        # For firm effects
        if firm_activated_mask.sum() > 0:
            # Calculate minimum intensity needed
            z_score = norm.ppf(positive_ratio)
            min_intensity_F = max(
                intensity, z_score * effect_noise_std + 0.01
            )  # Add small buffer
            # Adjust tauF_clean for activated observations
            tauF_clean[firm_activated_mask] = min_intensity_F

        # For user effects
        if user_activated_mask.sum() > 0:
            z_score = norm.ppf(positive_ratio)
            min_intensity_C = max(
                intensity, z_score * effect_noise_std + 0.01
            )  # Add small buffer
            # Adjust tauC_clean for activated observations
            tauC_clean[user_activated_mask] = min_intensity_C

        tauF = tauF_clean + rng.normal(0, effect_noise_std, size=n_users)
        tauC = tauC_clean + rng.normal(0, effect_noise_std, size=n_users)

        # Verify that at least positive_ratio of activated observations have positive effects
        if firm_activated_mask.sum() > 0:
            firm_positive_ratio_actual = (tauF[firm_activated_mask] > 0).mean()
            if firm_positive_ratio_actual < positive_ratio:
                # If still not enough, increase intensity further
                adjustment_factor = 1.1  # Increase by 10%
                while (
                    firm_positive_ratio_actual < positive_ratio
                    and adjustment_factor < 10
                ):
                    tauF_clean[firm_activated_mask] *= adjustment_factor
                    tauF = tauF_clean + rng.normal(0, effect_noise_std, size=n_users)
                    firm_positive_ratio_actual = (tauF[firm_activated_mask] > 0).mean()
                    adjustment_factor += 0.1

        if user_activated_mask.sum() > 0:
            user_positive_ratio_actual = (tauC[user_activated_mask] > 0).mean()
            if user_positive_ratio_actual < positive_ratio:
                # If still not enough, increase intensity further
                adjustment_factor = 1.1  # Increase by 10%
                while (
                    user_positive_ratio_actual < positive_ratio
                    and adjustment_factor < 10
                ):
                    tauC_clean[user_activated_mask] *= adjustment_factor
                    tauC = tauC_clean + rng.normal(0, effect_noise_std, size=n_users)
                    user_positive_ratio_actual = (tauC[user_activated_mask] > 0).mean()
                    adjustment_factor += 0.1
    else:
        tauF = tauF_clean
        tauC = tauC_clean

    # Step 6: Generate outcomes Y = μ(X) + τ(X) × T + ε
    # Generate treatment assignment (fully random with probability 1/2)
    T = rng.binomial(1, 0.5, size=n_users)

    noise_F = rng.normal(0, firm_outcome_noise_std, size=n_users)
    noise_C = rng.normal(0, user_outcome_noise_std, size=n_users)

    YF = baseline_F + tauF * T + noise_F
    YC = baseline_C + tauC * T + noise_C

    # Store functional form information
    functional_form = {
        "baseline_coef_F": baseline_coef_F,
        "baseline_coef_C": baseline_coef_C,
        "firm_combinations": firm_combinations,
        "user_combinations": user_combinations,
        "firm_signs": firm_signs.tolist(),
        "user_signs": user_signs.tolist(),
        "k": k,
        "n_categories": n_categories,
        "m_firm": m_firm,
        "m_user": m_user,
        "similarity": similarity,
        "intensity": intensity,
        "effect_noise_std": effect_noise_std,
        "positive_ratio": positive_ratio,
    }

    return X, T, YF, YC, tauF, tauC, region_type, functional_form


def _generate_all_combinations(n_categories: List[int]) -> List[List[int]]:
    """
    Generate all possible combinations of category indices.

    Parameters
    ----------
    n_categories : list of int
        Number of categories for each variable.

    Returns
    -------
    list of list
        All possible combinations, each as a list of category indices.
    """
    from itertools import product

    ranges = [range(n) for n in n_categories]
    combinations = list(product(*ranges))
    return [list(combo) for combo in combinations]


def get_data_summary(
    X: np.ndarray, T: np.ndarray, YF: np.ndarray, YC: np.ndarray
) -> Dict[str, Any]:
    """
    Get summary statistics of the generated data.

    Parameters
    ----------
    X : np.ndarray
        Binary feature matrix.
    T : np.ndarray
        Treatment assignment.
    YF : np.ndarray
        Firm outcome (continuous).
    YC : np.ndarray
        User outcome (continuous).

    Returns
    -------
    dict
        Dictionary with summary statistics.
    """
    n_users = X.shape[0]
    n_treated = T.sum()
    n_control = n_users - n_treated

    # Firm outcome statistics
    firm_treated_mean = YF[T == 1].mean() if n_treated > 0 else np.nan
    firm_control_mean = YF[T == 0].mean() if n_control > 0 else np.nan
    firm_ate = (
        firm_treated_mean - firm_control_mean
        if not (np.isnan(firm_treated_mean) or np.isnan(firm_control_mean))
        else np.nan
    )

    # User outcome statistics
    user_treated_mean = YC[T == 1].mean() if n_treated > 0 else np.nan
    user_control_mean = YC[T == 0].mean() if n_control > 0 else np.nan
    user_ate = (
        user_treated_mean - user_control_mean
        if not (np.isnan(user_treated_mean) or np.isnan(user_control_mean))
        else np.nan
    )

    return {
        "n_users": n_users,
        "n_treated": n_treated,
        "n_control": n_control,
        "treatment_rate": n_treated / n_users,
        "firm_treated_mean": firm_treated_mean,
        "firm_control_mean": firm_control_mean,
        "firm_ate": firm_ate,
        "firm_std": YF.std(),
        "user_treated_mean": user_treated_mean,
        "user_control_mean": user_control_mean,
        "user_ate": user_ate,
        "user_std": YC.std(),
    }


if __name__ == "__main__":
    # Example usage
    print("Generating binary comparison data...")

    X, T, YF, YC, tauF, tauC, region_type, functional_form = (
        generate_binary_comparison_data(
            n_users=10000,
            k=3,
            n_categories=[3, 4, 5],
            m_firm=5,
            m_user=5,
            similarity=0.5,
            intensity=2.0,
            effect_noise_std=0.1,
            firm_outcome_noise_std=1.0,
            user_outcome_noise_std=1.0,
            random_seed=42,
        )
    )

    print(
        f"Generated data shape: X={X.shape}, T={T.shape}, YF={YF.shape}, YC={YC.shape}"
    )
    print(f"YF has NaN: {np.isnan(YF).any()}")
    print(f"YC has NaN: {np.isnan(YC).any()}")
    print(f"\nRegion type distribution:")
    for rt in [1, 2, 3, 4]:
        count = (region_type == rt).sum()
        print(
            f"  Region {rt}: {count} observations ({100*count/len(region_type):.2f}%)"
        )

    # Get summary statistics
    summary = get_data_summary(X, T, YF, YC)
    print("\nData Summary:")
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )
