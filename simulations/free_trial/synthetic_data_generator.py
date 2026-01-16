import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes numerically stable sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input array

    Returns:
        Sigmoid values in range (0, 1)
    """
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def _gate_value(
    X: np.ndarray,
    indices: Union[int, List[int]],
    threshold: Union[float, List[float]],
) -> np.ndarray:
    """
    Computes gate values using AND logic: g = AND(X[j] > threshold[j] for all j)

    The gate activates only when ALL selected features exceed their respective thresholds.
    This creates conjunctive behavioral switches that require multiple conditions.

    Args:
        X: Feature matrix (n_samples, n_features)
        indices: Feature index or list of indices to use for gate
        threshold: Single threshold or list of thresholds (one per feature)

    Returns:
        Binary gate values: 1 if ALL conditions met, 0 otherwise
    """
    if isinstance(indices, int):
        # Single feature case
        if isinstance(threshold, (list, np.ndarray)):
            threshold = threshold[0]
        return (X[:, indices] > threshold).astype(float)
    else:
        # Multiple features case - use AND logic
        if isinstance(threshold, (int, float)):
            # Use same threshold for all features
            thresholds = [threshold] * len(indices)
        else:
            # Use individual thresholds
            thresholds = threshold

        # Check that all features exceed their thresholds
        gate_conditions = np.ones(X.shape[0], dtype=bool)
        for i, (idx, thresh) in enumerate(zip(indices, thresholds)):
            gate_conditions &= X[:, idx] > thresh

        return gate_conditions.astype(float)


def _latent_construct(
    X: np.ndarray,
    g: np.ndarray,
    m: float,
    v: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Constructs latent psychological variables from gate values.

    The formula creates bipolar behavior: (2g-1)*m transforms binary gates [0,1]
    into bipolar signals [-m, m], then adds noise and applies sigmoid for final output.

    Args:
        X: Feature matrix (unused, kept for interface consistency)
        g: Binary gate values [0, 1]
        m: Bipolarity strength - controls magnitude of high/low distinction
        v: Noise scale - adds individual variation
        rng: Random number generator

    Returns:
        Latent variable values in range (0, 1)
    """
    eps = rng.normal(0.0, 1.0, size=X.shape[0])
    k_star = (2.0 * g - 1.0) * m + v * eps
    return _sigmoid(k_star)


def generate_data(
    n: int = 50_000,
    p: int = 100,
    v: float = 1.0,
    m_U: float = 1.0,
    m_N: float = 1.0,
    m_S: float = 1.0,
    gate_indices: Optional[Dict[str, Union[int, List[int]]]] = None,
    gate_thresholds: Optional[Dict[str, float]] = None,
    a_u: float = 0.3,
    a_s: float = 0.9,
    a0: float = 0.0,
    a_p: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generates synthetic data for studying sunk cost bias and novelty effects in trial subscriptions.

    The model creates three psychological constructs:
    - U (Usefulness): Actual product value perception
    - N (Novelty): Appeal of trying something new
    - S (Sunk Cost): Psychological investment from time spent

    Decision logic:
    - Short trials: Subscribe if high Novelty OR weighted Usefulness
    - Long trials: Subscribe if high Usefulness OR weighted Sunk Cost

    Args:
        n: Number of samples
        p: Number of features
        v: Noise scale for latent constructs
        m_U, m_N, m_S: Bipolarity strength for each construct
        gate_indices: Which features control each construct
        gate_thresholds: Activation thresholds for gates (can be single values or lists for multi-feature gates)
        a_u: Weight of Usefulness in short trial decisions
        a_s: Weight of Sunk Cost in long trial decisions
        a0, a_p: Parameters for subscription probability
        seed: Random seed for reproducibility

    Returns:
        X: Feature matrix (n, p)
        L: Dictionary of latent variables {'U', 'N', 'S'}
        T: Treatment assignment (0=Short, 1=Long)
        YF: Firm outcome - subscription decision
        YC: Consumer outcome - satisfaction (NaN if not subscribed)
        P_short: Perceived value for short trials
        P_long: Perceived value for long trials
        pi: Subscription probability
    """
    rng = np.random.default_rng(seed)

    # 1) Generate random features
    X = rng.normal(0.0, 1.0, size=(n, p))

    # 2) Configure gates for each psychological construct
    if gate_indices is None:
        # Default: use one random feature per construct
        if p >= 3:
            idxs = rng.choice(p, size=3, replace=False)
            gate_indices = {
                "U": [int(idxs[0])],  # Usefulness gate
                "N": [int(idxs[1])],  # Novelty gate
                "S": [int(idxs[2])],  # Sunk Cost gate
            }
        else:
            # Fallback: use same feature for all constructs
            gate_indices = {"U": [0], "N": [0], "S": [0]}
    else:
        gate_indices = dict(gate_indices)  # Create copy to avoid modifying input

    if gate_thresholds is None:
        gate_thresholds = {"U": 0.0, "N": 0.0, "S": 0.0}
    else:
        gate_thresholds = dict(gate_thresholds)

    # 3) Compute gate values for each psychological construct
    gU = _gate_value(X, gate_indices["U"], gate_thresholds.get("U", 0.0))
    gN = _gate_value(X, gate_indices["N"], gate_thresholds.get("N", 0.0))
    gS = _gate_value(X, gate_indices["S"], gate_thresholds.get("S", 0.0))

    # 4) Construct latent psychological variables
    U = _latent_construct(X, gU, m_U, v, rng)  # Usefulness
    N = _latent_construct(X, gN, m_N, v, rng)  # Novelty
    S = _latent_construct(X, gS, m_S, v, rng)  # Sunk Cost
    L = {"U": U, "N": N, "S": S}

    # 5) Assign treatment: Short (0) vs Long (1) trials
    T = rng.binomial(1, 0.5, size=n).astype(int)

    # 6) Calculate perceived value based on trial type
    # Short trials: driven by Novelty OR weighted Usefulness
    P_short = np.logaddexp(N, a_u * U)

    # Long trials: driven by Usefulness OR weighted Sunk Cost
    P_long = np.logaddexp(U, a_s * S)

    # Combine based on treatment assignment
    P = np.where(T == 0, P_short, P_long)

    # 7) Generate subscription decisions
    pi = np.clip(a0 + a_p * P, 0, 1)  # Subscription probability
    YF = (rng.random(size=n) < pi).astype(int)  # Binary subscription outcome

    # 8) Generate consumer satisfaction (only observed for subscribers)
    YC = np.full(n, np.nan, dtype=float)
    YC[YF == 1] = U[YF == 1]  # Satisfaction equals actual usefulness

    return X, L, T, YF, YC, P_short, P_long, pi
