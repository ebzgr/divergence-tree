"""
Data Generator for Comparison Simulation (Continuous Outcomes)

Generates synthetic data with:
- N users with F features (uniformly distributed 0-10)
- Random 50/50 treatment assignment
- Firm outcome (continuous) with treatment effects based on X1, X2
- User outcome (continuous) with treatment effects based on X1, X2
- Both outcomes are always observed (no NaN values)
- Specific treatment effect structure as specified
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def generate_comparison_data(
    n_users: int = 10000,
    n_features: int = 10,
    firm_outcome_base: float = 0.0,
    firm_outcome_noise_std: float = 1.0,
    user_outcome_base: float = 0.0,
    user_outcome_noise_std: float = 1.0,
    random_seed: Optional[int] = None,
    firm_effect_strength: float = 1,
    user_effect_strength: float = 2.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Generate synthetic data for comparison simulation with continuous outcomes.

    Args:
        n_users: Number of users to generate
        n_features: Number of features per user (must be >= 4)
        firm_outcome_base: Base value for firm outcome (without treatment)
        firm_outcome_noise_std: Standard deviation of firm outcome noise
        user_outcome_base: Base value for user outcome (without treatment)
        user_outcome_noise_std: Standard deviation of user outcome noise
        random_seed: Random seed for reproducibility
        firm_effect_strength: Direct strength of firm treatment effects (default 0.2)
        user_effect_strength: Direct strength of user treatment effects (default 2.0)

    Returns:
        X: Feature matrix (n_users, n_features)
        T: Treatment assignment (n_users,) - binary 0/1
        YF: Firm outcome (n_users,) - continuous, always observed
        YC: User outcome (n_users,) - continuous, always observed
        tauF: Firm treatment effects (n_users,)
        tauC: User treatment effects (n_users,)
        region_type: Region type (n_users,) - 1: both positive, 2: firm+ customer-,
                    3: firm- customer+, 4: both negative
    """

    if n_features < 4:
        raise ValueError("n_features must be >= 4")

    # Set random seed
    rng = np.random.default_rng(random_seed)

    # Generate features: uniform distribution between 0 and 10
    X = rng.uniform(0, 10, size=(n_users, n_features))

    # Generate treatment assignment (random 50/50)
    T = rng.binomial(1, 0.5, size=n_users)

    # Calculate treatment effects based on X1, X2 with direct strength values
    tauF = calculate_firm_treatment_effects(X, firm_effect_strength)
    tauC = calculate_user_treatment_effects(X, user_effect_strength)

    # Generate firm outcomes (continuous)
    # Base value + treatment effect (only for treated users) + noise
    firm_treatment_effect = tauF * T
    firm_outcome_noise = rng.normal(0, firm_outcome_noise_std, size=n_users)
    YF = firm_outcome_base + firm_treatment_effect + firm_outcome_noise

    # Generate user outcomes (continuous)
    # Base value + treatment effect (only for treated users) + noise
    user_treatment_effect = tauC * T
    user_outcome_noise = rng.normal(0, user_outcome_noise_std, size=n_users)
    YC = user_outcome_base + user_treatment_effect + user_outcome_noise

    # Compute region type based on signs of treatment effects
    # 1: both positive, 2: firm+ customer-, 3: firm- customer+, 4: both negative
    region_type = np.zeros(n_users, dtype=int)
    both_positive = (tauF > 0) & (tauC > 0)
    firm_pos_cust_neg = (tauF > 0) & (tauC <= 0)
    firm_neg_cust_pos = (tauF <= 0) & (tauC > 0)
    both_negative = (tauF <= 0) & (tauC <= 0)

    region_type[both_positive] = 1
    region_type[firm_pos_cust_neg] = 2
    region_type[firm_neg_cust_pos] = 3
    region_type[both_negative] = 4

    # Both outcomes are always observed (no NaN values)
    return X, T, YF, YC, tauF, tauC, region_type


def calculate_firm_treatment_effects(
    X: np.ndarray, strength: float = 0.1
) -> np.ndarray:
    """
    Calculate firm outcome treatment effects based on X1, X2.

    Treatment effect structure (using direct strength values):
    0<=X1<5 & 0<=X2<5 : tau_F = -strength
    0<=X1<5 & 5<=X2<=10 : tau_F = +strength
    5<=X1<9 & 0<=X2<=5 : tau_F = -strength
    5<=X1<9 & 5<=X2<=10 : tau_F = +strength
    9<=X1<=10 & 0<=X2<=5 : tau_F = +strength
    9<=X1<=10 & 5<=X2<=10 : tau_F = +strength

    Args:
        X: Feature matrix (n_users, n_features)
        strength: Direct strength of treatment effects

    Returns:
        tauF: Firm treatment effects (n_users,)
    """
    n_users = X.shape[0]
    tauF = np.zeros(n_users)

    X1 = X[:, 0]
    X2 = X[:, 1]

    # Define regions based on X1, X2
    region1 = (X1 >= 0) & (X1 < 5) & (X2 >= 0) & (X2 < 5)
    region2 = (X1 >= 0) & (X1 < 5) & (X2 >= 5) & (X2 <= 10)
    region3 = (X1 >= 5) & (X1 <= 10) & (X2 >= 0) & (X2 < 1)
    region4 = (X1 >= 5) & (X1 <= 10) & (X2 >= 1) & (X2 <= 5)
    region5 = (X1 >= 5) & (X1 <= 10) & (X2 >= 5) & (X2 < 9)
    region6 = (X1 >= 5) & (X1 <= 10) & (X2 >= 9) & (X2 <= 10)

    # Assign treatment effects with direct strength values
    tauF[region1] = -strength
    tauF[region2] = +strength
    tauF[region3] = -strength
    tauF[region4] = -strength
    tauF[region5] = +strength
    tauF[region6] = +strength

    return tauF


def calculate_user_treatment_effects(
    X: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    """
    Calculate user outcome treatment effects based on X1, X2.

    Treatment effect structure (using direct strength values):
    0<=X1<5 & 0<=X2<5 : tau_C = -strength
    0<=X1<5 & 5<=X2<=10 : tau_C = -strength
    5<=X1<9 & 0<=X2<=5 : tau_C = +strength
    5<=X1<9 & 5<=X2<=10 : tau_C = +strength
    9<=X1<=10 & 0<=X2<=5 : tau_C = +strength
    9<=X1<=10 & 5<=X2<=10 : tau_C = -strength

    Args:
        X: Feature matrix (n_users, n_features)
        strength: Direct strength of treatment effects

    Returns:
        tauC: User treatment effects (n_users,)
    """
    n_users = X.shape[0]
    tauC = np.zeros(n_users)

    X1 = X[:, 0]
    X2 = X[:, 1]

    # Define regions based on X1, X2
    region1 = (X1 >= 0) & (X1 < 5) & (X2 >= 0) & (X2 < 5)
    region2 = (X1 >= 0) & (X1 < 5) & (X2 >= 5) & (X2 <= 10)
    region3 = (X1 >= 5) & (X1 <= 10) & (X2 >= 0) & (X2 < 1)
    region4 = (X1 >= 5) & (X1 <= 10) & (X2 >= 1) & (X2 <= 5)
    region5 = (X1 >= 5) & (X1 <= 10) & (X2 >= 5) & (X2 < 9)
    region6 = (X1 >= 5) & (X1 <= 10) & (X2 >= 9) & (X2 <= 10)

    # Assign treatment effects with direct strength values
    tauC[region1] = -strength
    tauC[region2] = -strength
    tauC[region3] = -strength
    tauC[region4] = +strength
    tauC[region5] = +strength
    tauC[region6] = -strength

    return tauC


def get_data_summary(
    X: np.ndarray, T: np.ndarray, YF: np.ndarray, YC: np.ndarray
) -> Dict[str, Any]:
    """
    Get summary statistics of the generated data.

    Args:
        X: Feature matrix
        T: Treatment assignment
        YF: Firm outcome (continuous)
        YC: User outcome (continuous)

    Returns:
        Dictionary with summary statistics
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
    print("Generating comparison data with continuous outcomes...")

    X, T, YF, YC, tauF, tauC, region_type = generate_comparison_data(
        n_users=10000,
        n_features=10,
        firm_outcome_base=0.0,
        firm_outcome_noise_std=1.0,
        user_outcome_base=0.0,
        user_outcome_noise_std=1.0,
        random_seed=42,
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
