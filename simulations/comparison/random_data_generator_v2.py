"""
Random Data Generator V2 for Comparison Simulation (Continuous Outcomes)

Generates synthetic data using standard DGP structure: Y = μ(X) + τ(X) × T + ε
- Random coefficients from N(0,1) following literature standard
- Configurable feature overlap between firm and user treatment effects
- Intensity scaling for treatment effects
- Optional noise on treatment effects
- Fully random treatment assignment with probability 1/2
- Patchy treatment effects with region-based masking using hyper-rectangles
- Logit transitions for smooth activation/deactivation
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List


def _generate_hyper_rectangles(
    n_regions: int,
    n_features: int,
    feature_bounds: Tuple[float, float] = (0.0, 10.0),
    min_size: float = 1.0,
    max_size: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate random hyper-rectangles in feature space.
    
    Parameters
    ----------
    n_regions : int
        Number of hyper-rectangles to generate.
    n_features : int
        Number of feature dimensions.
    feature_bounds : tuple, default=(0.0, 10.0)
        (min, max) bounds for feature space.
    min_size : float, default=1.0
        Minimum size of rectangle in each dimension.
    max_size : float, default=5.0
        Maximum size of rectangle in each dimension.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    list of dict
        Each dict contains 'min_bounds' and 'max_bounds' arrays of shape (n_features,).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    rectangles = []
    min_bound, max_bound = feature_bounds
    
    for _ in range(n_regions):
        # Randomly sample center for each dimension
        centers = rng.uniform(
            min_bound + max_size / 2,
            max_bound - max_size / 2,
            size=n_features
        )
        
        # Randomly sample size for each dimension
        sizes = rng.uniform(min_size, max_size, size=n_features)
        
        # Compute min and max bounds
        min_bounds = centers - sizes / 2
        max_bounds = centers + sizes / 2
        
        # Clip to feature space bounds
        min_bounds = np.clip(min_bounds, min_bound, max_bound)
        max_bounds = np.clip(max_bounds, min_bound, max_bound)
        
        rectangles.append({
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
        })
    
    return rectangles


def _compute_region_mask(
    X: np.ndarray,
    rectangles: List[Dict[str, np.ndarray]],
    transition_steepness: float = 10.0,
) -> np.ndarray:
    """
    Compute region mask using logit transitions.
    
    For each observation, computes how "inside" each region it is,
    then applies a logit function to create smooth transitions.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_obs, n_features)
        Feature matrix.
    rectangles : list of dict
        List of hyper-rectangles, each with 'min_bounds' and 'max_bounds'.
    transition_steepness : float, default=10.0
        Steepness of logit transition. Higher = sharper, lower = smoother.
    
    Returns
    -------
    np.ndarray of shape (n_obs,)
        Mask values in [0, 1] for each observation.
    """
    n_obs = X.shape[0]
    n_regions = len(rectangles)
    
    if n_regions == 0:
        # No regions means all effects are active
        return np.ones(n_obs)
    
    # For each observation and each region, compute "inside-ness" score
    # Score = minimum distance to boundaries across all dimensions
    # Positive score = inside, negative = outside
    region_scores = np.zeros((n_obs, n_regions))
    
    for i, rect in enumerate(rectangles):
        min_bounds = rect['min_bounds']  # shape: (n_features,)
        max_bounds = rect['max_bounds']  # shape: (n_features,)
        
        # For each dimension, compute distance to boundaries
        # Distance is positive if inside, negative if outside
        dist_to_min = X - min_bounds  # shape: (n_obs, n_features)
        dist_to_max = max_bounds - X  # shape: (n_obs, n_features)
        
        # Inside-ness: minimum of distances to both boundaries
        # This gives positive values when inside, negative when outside
        inside_scores = np.minimum(dist_to_min, dist_to_max)  # shape: (n_obs, n_features)
        
        # Take minimum across dimensions (most constraining dimension)
        region_scores[:, i] = np.min(inside_scores, axis=1)  # shape: (n_obs,)
    
    # Apply logit function to each region's score
    # sigmoid(steepness * score) gives smooth transition
    # Higher steepness = sharper transition
    region_masks = 1.0 / (1.0 + np.exp(-transition_steepness * region_scores))  # shape: (n_obs, n_regions)
    
    # Combine masks from multiple regions (use max to get union)
    # This means effect is active if inside ANY region
    combined_mask = np.max(region_masks, axis=1)  # shape: (n_obs,)
    
    return combined_mask


def generate_random_comparison_data(
    n_users: int = 10000,
    n_features: int = 10,
    n_active_features: int = 5,
    overlap: float = 0.5,
    intensity: float = 1.0,
    effect_noise_std: float = 0.0,
    firm_outcome_noise_std: float = 1.0,
    user_outcome_noise_std: float = 1.0,
    random_seed: Optional[int] = None,
    n_regions_firm: Optional[int] = None,
    n_regions_user: Optional[int] = None,
    region_overlap: float = 0.0,
    transition_steepness: float = 10.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
]:
    """
    Generate synthetic data using random coefficients and standard DGP structure.
    
    Follows standard data generating process: Y = μ(X) + τ(X) × T + ε
    where:
    - μ(X) is baseline outcome function (linear combination of all features)
    - τ(X) is treatment effect function (linear combination of selected features)
    - T is treatment indicator (random assignment with probability 0.5)
    - ε is outcome noise
    
    Treatment effects can be patchy (active only in certain regions) if n_regions_firm
    or n_regions_user are specified.
    
    Parameters
    ----------
    n_users : int, default=10000
        Number of observations to generate.
    n_features : int, default=10
        Total number of features.
    n_active_features : int, default=5
        Number of features used for generating each treatment effect (shared for both outcomes).
    overlap : float, default=0.5
        Proportion of features that are shared between firm and user treatment effects.
        Must be in [0, 1]. For example, if n_active_features=4 and overlap=0.5,
        then 2 features are shared.
    intensity : float, default=1.0
        Scales the treatment effects after normalization to [-1, 1] range.
    effect_noise_std : float, default=0.0
        Standard deviation of noise added to treatment effects (optional).
    firm_outcome_noise_std : float, default=1.0
        Standard deviation of noise for firm outcomes.
    user_outcome_noise_std : float, default=1.0
        Standard deviation of noise for user outcomes.
    random_seed : int, optional
        Random seed for reproducibility.
    n_regions_firm : int, optional
        Number of active regions for firm treatment effects. If None, effects are active everywhere.
    n_regions_user : int, optional
        Number of active regions for user treatment effects. If None, effects are active everywhere.
    region_overlap : float, default=0.0
        Proportion of regions that overlap between firm and user effects.
        Must be in [0, 1]. For example, if n_regions_firm=4 and region_overlap=0.5,
        then 2 regions are shared.
    transition_steepness : float, default=10.0
        Controls sharpness of logit transition for region boundaries.
        Higher values = sharper transitions (more binary), lower = smoother transitions.
    
    Returns
    -------
    X : np.ndarray of shape (n_users, n_features)
        Feature matrix.
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
        - 'baseline_coef_F': Baseline coefficients for firm outcome (all features)
        - 'baseline_coef_C': Baseline coefficients for user outcome (all features)
        - 'effect_coef_F': Treatment effect coefficients for firm (selected features)
        - 'effect_coef_C': Treatment effect coefficients for user (selected features)
        - 'firm_selected_features': Feature indices used for firm treatment effects
        - 'user_selected_features': Feature indices used for user treatment effects
        - 'shared_features': Feature indices shared between firm and user effects
        - 'n_active_features': Number of active features
        - 'overlap': Overlap parameter used
        - 'intensity': Intensity parameter used
        - 'n_regions_firm': Number of regions for firm effects (if patchy)
        - 'n_regions_user': Number of regions for user effects (if patchy)
        - 'region_overlap': Region overlap parameter (if patchy)
        - 'transition_steepness': Transition steepness parameter (if patchy)
        - 'firm_rectangles': Firm region rectangles (if patchy)
        - 'user_rectangles': User region rectangles (if patchy)
    """
    # Input validation
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if n_active_features < 1:
        raise ValueError("n_active_features must be >= 1")
    if n_active_features > n_features:
        raise ValueError("n_active_features must be <= n_features")
    if not (0 <= overlap <= 1):
        raise ValueError("overlap must be in [0, 1]")
    if n_regions_firm is not None and n_regions_firm < 0:
        raise ValueError("n_regions_firm must be >= 0")
    if n_regions_user is not None and n_regions_user < 0:
        raise ValueError("n_regions_user must be >= 0")
    if not (0 <= region_overlap <= 1):
        raise ValueError("region_overlap must be in [0, 1]")
    if transition_steepness <= 0:
        raise ValueError("transition_steepness must be > 0")
    
    # Set random seed
    rng = np.random.default_rng(random_seed)
    
    # Generate features: uniform distribution between 0 and 10
    X = rng.uniform(0, 10, size=(n_users, n_features))
    
    # Generate treatment assignment (fully random with probability 1/2)
    T = rng.binomial(1, 0.5, size=n_users)
    
    # Step 1: Feature selection for treatment effects
    n_shared = int(np.round(n_active_features * overlap))
    n_unique_F = n_active_features - n_shared
    n_unique_C = n_active_features - n_shared
    
    # Select features for firm treatment effects
    all_feature_indices = np.arange(n_features)
    firm_selected = rng.choice(all_feature_indices, size=n_active_features, replace=False)
    
    # Select shared features
    shared_features = firm_selected[:n_shared] if n_shared > 0 else np.array([], dtype=int)
    
    # Select unique features for user treatment effects
    remaining_features = np.setdiff1d(all_feature_indices, firm_selected)
    if n_unique_C > 0 and len(remaining_features) > 0:
        n_unique_to_select = min(n_unique_C, len(remaining_features))
        user_unique = rng.choice(remaining_features, size=n_unique_to_select, replace=False)
    else:
        user_unique = np.array([], dtype=int)
    
    # Combine shared and unique features for user effects
    user_selected = np.concatenate([shared_features, user_unique]) if len(user_unique) > 0 else shared_features
    
    # Step 2: Generate random coefficients (N(0,1) following literature standard)
    # Baseline coefficients (use all features)
    baseline_coef_F = rng.standard_normal(n_features)
    baseline_coef_C = rng.standard_normal(n_features)
    
    # Treatment effect coefficients (use selected features)
    effect_coef_F = rng.standard_normal(len(firm_selected))
    effect_coef_C = rng.standard_normal(len(user_selected))
    
    # Step 3: Compute baseline outcomes μ(X)
    baseline_F = X @ baseline_coef_F
    baseline_C = X @ baseline_coef_C
    
    # Step 4: Compute treatment effects τ(X)
    # Raw linear combinations
    tauF_raw = X[:, firm_selected] @ effect_coef_F
    tauC_raw = X[:, user_selected] @ effect_coef_C
    
    # Step 4a: Apply region-based masking if regions are specified
    firm_rectangles = None
    user_rectangles = None
    mask_F = None
    mask_C = None
    
    if n_regions_firm is not None and n_regions_firm > 0:
        # Generate hyper-rectangles for firm effects
        firm_rectangles = _generate_hyper_rectangles(
            n_regions=n_regions_firm,
            n_features=n_features,
            feature_bounds=(0.0, 10.0),
            min_size=1.0,
            max_size=5.0,
            rng=rng,
        )
        
        # Compute region mask
        mask_F = _compute_region_mask(X, firm_rectangles, transition_steepness)
        
        # Apply mask to raw treatment effects
        tauF_raw = tauF_raw * mask_F
    
    if n_regions_user is not None and n_regions_user > 0:
        # Generate hyper-rectangles for user effects
        # Handle region overlap
        n_shared_regions = int(np.round(n_regions_user * region_overlap)) if n_regions_user > 0 else 0
        
        if n_shared_regions > 0 and firm_rectangles is not None:
            # Share first n_shared_regions rectangles
            shared_rectangles = firm_rectangles[:min(n_shared_regions, len(firm_rectangles))]
            n_unique_regions = n_regions_user - len(shared_rectangles)
            
            if n_unique_regions > 0:
                unique_rectangles = _generate_hyper_rectangles(
                    n_regions=n_unique_regions,
                    n_features=n_features,
                    feature_bounds=(0.0, 10.0),
                    min_size=1.0,
                    max_size=5.0,
                    rng=rng,
                )
                user_rectangles = shared_rectangles + unique_rectangles
            else:
                user_rectangles = shared_rectangles
        else:
            # No overlap or no firm regions, generate all unique
            user_rectangles = _generate_hyper_rectangles(
                n_regions=n_regions_user,
                n_features=n_features,
                feature_bounds=(0.0, 10.0),
                min_size=1.0,
                max_size=5.0,
                rng=rng,
            )
        
        # Compute region mask
        mask_C = _compute_region_mask(X, user_rectangles, transition_steepness)
        
        # Apply mask to raw treatment effects
        tauC_raw = tauC_raw * mask_C
    
    # Normalize to [-1, 1] range
    tauF_min, tauF_max = tauF_raw.min(), tauF_raw.max()
    tauC_min, tauC_max = tauC_raw.min(), tauC_raw.max()
    
    if tauF_max - tauF_min > 1e-10:  # Avoid division by zero
        tauF_normalized = 2 * (tauF_raw - tauF_min) / (tauF_max - tauF_min) - 1
    else:
        tauF_normalized = np.zeros_like(tauF_raw)
    
    if tauC_max - tauC_min > 1e-10:  # Avoid division by zero
        tauC_normalized = 2 * (tauC_raw - tauC_min) / (tauC_max - tauC_min) - 1
    else:
        tauC_normalized = np.zeros_like(tauC_raw)
    
    # Scale by intensity
    tauF = intensity * tauF_normalized
    tauC = intensity * tauC_normalized
    
    # Add noise to treatment effects (optional)
    if effect_noise_std > 0:
        tauF_final = tauF + rng.normal(0, effect_noise_std, size=n_users)
        tauC_final = tauC + rng.normal(0, effect_noise_std, size=n_users)
    else:
        tauF_final = tauF
        tauC_final = tauC
    
    # Step 5: Generate outcomes Y = μ(X) + τ(X) × T + ε
    noise_F = rng.normal(0, firm_outcome_noise_std, size=n_users)
    noise_C = rng.normal(0, user_outcome_noise_std, size=n_users)
    
    YF = baseline_F + tauF_final * T + noise_F
    YC = baseline_C + tauC_final * T + noise_C
    
    # Compute region type based on signs of treatment effects
    region_type = np.zeros(n_users, dtype=int)
    both_positive = (tauF_final > 0) & (tauC_final > 0)
    firm_pos_cust_neg = (tauF_final > 0) & (tauC_final <= 0)
    firm_neg_cust_pos = (tauF_final <= 0) & (tauC_final > 0)
    both_negative = (tauF_final <= 0) & (tauC_final <= 0)
    
    region_type[both_positive] = 1
    region_type[firm_pos_cust_neg] = 2
    region_type[firm_neg_cust_pos] = 3
    region_type[both_negative] = 4
    
    # Store functional form information (coefficients and feature indices)
    functional_form = {
        'baseline_coef_F': baseline_coef_F,
        'baseline_coef_C': baseline_coef_C,
        'effect_coef_F': effect_coef_F,
        'effect_coef_C': effect_coef_C,
        'firm_selected_features': firm_selected,
        'user_selected_features': user_selected,
        'shared_features': shared_features,
        'n_active_features': n_active_features,
        'overlap': overlap,
        'intensity': intensity,
    }
    
    # Add region information if patchy effects are used
    if n_regions_firm is not None or n_regions_user is not None:
        functional_form['n_regions_firm'] = n_regions_firm
        functional_form['n_regions_user'] = n_regions_user
        functional_form['region_overlap'] = region_overlap
        functional_form['transition_steepness'] = transition_steepness
        if firm_rectangles is not None:
            functional_form['firm_rectangles'] = firm_rectangles
        if user_rectangles is not None:
            functional_form['user_rectangles'] = user_rectangles
    
    # Both outcomes are always observed (no NaN values)
    return X, T, YF, YC, tauF_final, tauC_final, region_type, functional_form


def get_data_summary(
    X: np.ndarray, T: np.ndarray, YF: np.ndarray, YC: np.ndarray
) -> Dict[str, Any]:
    """
    Get summary statistics of the generated data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
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
    print("Generating random comparison data with patchy treatment effects...")
    
    X, T, YF, YC, tauF, tauC, region_type, functional_form = generate_random_comparison_data(
        n_users=10000,
        n_features=10,
        n_active_features=5,
        overlap=0.5,
        intensity=1.0,
        effect_noise_std=0.0,
        firm_outcome_noise_std=1.0,
        user_outcome_noise_std=1.0,
        random_seed=42,
        n_regions_firm=2,
        n_regions_user=2,
        region_overlap=0.5,
        transition_steepness=10.0,
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

