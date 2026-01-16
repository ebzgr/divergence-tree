"""
Configuration file for the free trial simulation.

This uses separate configuration classes for different components:
- DataGenerationConfig: Parameters for synthetic data generation
- AnalysisConfig: Parameters for divergence tree analysis
- FreeTrialConfig: Main configuration that combines both

Just modify the parameters in each config class to experiment with different scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataGenerationConfig:
    """
    Configuration for synthetic data generation.

    Controls how the free trial data is created:
    - Number of users and features
    - Psychological construct bipolarity
    - Treatment effect parameters
    - Noise and variation levels
    """

    # ===================== DATA GENERATION =====================
    N_USERS: int = 100000  # Number of synthetic users
    N_FEATURES: int = 100  # Number of features per user
    RANDOM_SEED: int = 0  # For reproducible results

    # ===================== PSYCHOLOGICAL CONSTRUCTS =====================
    # Which features control each psychological construct
    GATE_INDICES: Dict[str, List[int]] = None
    # Activation thresholds for each construct
    GATE_THRESHOLDS: Dict[str, float] = None

    # ===================== MODEL PARAMETERS =====================
    # Noise and variation
    NOISE_SCALE: float = 2.0  # Higher = more individual variation

    # Psychological construct bipolarity
    # 0 = unipolar (all values near 0.5), 1 = almost uniform, >1 = bipolar, 3 = highly bipolar
    USEFULNESS_BIPOLARITY: float = 2.0  # Controls how extreme usefulness values become
    NOVELTY_BIPOLARITY: float = 2.0  # Controls how extreme novelty values become
    SUNK_COST_BIPOLARITY: float = 2.0  # Controls how extreme sunk cost values become

    # Treatment effect parameters
    USEFULNESS_SHORT_WEIGHT: float = 0.5  # How much usefulness affects short-term value
    SUNK_COST_LONG_WEIGHT: float = 1.0  # How much sunk cost affects long-term value

    # Subscription model parameters
    BASELINE_SUBSCRIPTION: float = -0.8  # Baseline log-odds of subscription
    VALUE_SUBSCRIPTION_WEIGHT: float = (
        0.9  # How much perceived value affects subscription
    )

    def __post_init__(self):
        """Initialize default values for complex types."""
        if self.GATE_INDICES is None:
            self.GATE_INDICES = {
                "U": [0],  # Usefulness - controlled by feature 0
                "N": [1],  # Novelty - controlled by feature 1
                "S": [2],  # Sunk Cost - controlled by feature 2
            }

        if self.GATE_THRESHOLDS is None:
            self.GATE_THRESHOLDS = {
                "U": 0.0,  # Usefulness activates when feature 0 > 0
                "N": 0.0,  # Novelty activates when feature 1 > 0
                "S": 0.0,  # Sunk cost activates when feature 2 > 0
            }

    def validate(self) -> None:
        """Validate data generation parameters."""
        if self.N_USERS <= 0:
            raise ValueError("N_USERS must be positive")
        if self.N_FEATURES <= 0:
            raise ValueError("N_FEATURES must be positive")
        if self.NOISE_SCALE < 0:
            raise ValueError("NOISE_SCALE must be non-negative")
        if any(
            m < 0
            for m in [
                self.USEFULNESS_BIPOLARITY,
                self.NOVELTY_BIPOLARITY,
                self.SUNK_COST_BIPOLARITY,
            ]
        ):
            raise ValueError("All bipolarity parameters must be non-negative")

        print("Data generation configuration validated")


@dataclass
class AnalysisConfig:
    """
    Configuration for divergence tree analysis.

    Controls how the segmentation analysis is performed:
    - Hyperparameter tuning settings
    - Tree construction parameters
    - Validation and evaluation settings
    """

    # ===================== HYPERPARAMETER TUNING =====================
    N_TUNING_TRIALS: int = 20  # Number of hyperparameter optimization trials
    VALID_FRACTION: float = 0.2  # Fraction of data for validation

    # ===================== FIXED PARAMETERS =====================
    HONEST: bool = True  # Use honest estimation
    MIN_LEAF_TREATED: int = 1  # Minimum treated samples per leaf
    MIN_LEAF_CONTROL: int = 1  # Minimum control samples per leaf
    MIN_LEAF_CONV_TREATED: int = 1  # Minimum converted treated samples per leaf
    MIN_LEAF_CONV_CONTROL: int = 1  # Minimum converted control samples per leaf
    RANDOM_STATE: int = 0  # Random seed for reproducibility
    CO_MOVEMENT: str = "both"  # Focus on convergence, divergence, or both
    LAMBDA_: float = 1.0  # Weight for cross-movement term

    def validate(self) -> None:
        """Validate analysis parameters."""
        if self.N_TUNING_TRIALS <= 0:
            raise ValueError("N_TUNING_TRIALS must be positive")
        if not 0 < self.VALID_FRACTION < 1:
            raise ValueError("VALID_FRACTION must be between 0 and 1")
        if self.CO_MOVEMENT not in ["convergence", "divergence", "both"]:
            raise ValueError(
                "CO_MOVEMENT must be 'convergence', 'divergence', or 'both'"
            )

        print("Analysis configuration validated")


@dataclass
class FreeTrialConfig:
    """
    Main configuration for the free trial simulation.

    Combines data generation and analysis configurations.
    This is the primary configuration class that users should modify.
    """

    data_generation: DataGenerationConfig = None
    analysis: AnalysisConfig = None

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.data_generation is None:
            self.data_generation = DataGenerationConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()

    def validate(self) -> None:
        """Validate all configurations."""
        self.data_generation.validate()
        self.analysis.validate()
        print("Free trial configuration validated")


# Create default configuration instances
data_generation_config = DataGenerationConfig()
analysis_config = AnalysisConfig()
free_trial_config = FreeTrialConfig()

# For backward compatibility, create aliases
config = free_trial_config  # Main config instance
