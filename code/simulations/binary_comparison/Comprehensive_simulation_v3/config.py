"""
Configuration for region-stratified comprehensive simulation v3.
"""

# Aspect ranges
NOISE_MIN = 0.01
NOISE_MAX = 10.0
DATA_SIZE_MIN = 10000
DATA_SIZE_MAX = 500000
SPARSITY_VALUES = [1, 2, 3, 4, 5, 6]
RARENESS_MIN = 0.01
RARENESS_MAX = 0.5
INTENSITY_MIN = 0.1
INTENSITY_MAX = 1.0

# DGP defaults
DEFAULT_EFFECT_NOISE_STD = 0.0  # v3 noise should affect outcomes, not treatment effects

# Train/val/test
TRAIN_FRAC = 0.5
VAL_FRAC = 0.25
TEST_FRAC = 0.25

# DivTree
DIVTREE_FIXED_PARAMS = {
    "n_quantiles": 20,
    "eps_scale": 1e-8,
}
DIVTREE_SEARCH_SPACE = {
    "max_partitions": {"low": 2, "high": 100},
    "min_improvement_ratio": {"low": 0.001, "high": 0.1, "log": True},
}
DIVTREE_N_TRIALS = 30

# TwoStep
TWOSTEP_CAUSAL_FOREST_PARAMS = {
    "n_jobs": 1,
    "n_estimators": 100,
    "max_depth": None,
    # Fixed split/leaf constraints (do not override dynamically).
    "min_samples_split": 10,
    "min_samples_leaf": 5,
}
# Enable econml CausalForestDML self-tuning before fit.
# Keep depth fixed at None; tune leaf fraction controls.
TWOSTEP_CAUSAL_FOREST_TUNE_PARAMS = {
    "params": {
        "min_weight_fraction_leaf": [0.0005, 0.001, 0.005, 0.01, 0.05],
        "min_var_fraction_leaf": [0.001, 0.005, 0.01, 0.05 , 0.1],
    }
}
TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES = {"low": 2, "high": 200}
TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS = 30
TWOSTEP_TUNED_VARIANTS = {
    "twostep_tuned": "accuracy",
    "twostep_recall": "recall_region_2",
}

# Simulation
LAMBDA_VALUES = [0, 1, 2, 4, 8]
BASE_RANDOM_SEED = 0
DEFAULT_N_SIMULATIONS = 10000
DEFAULT_N_JOBS = 33
DEFAULT_BATCH_SIZE = 100

# Output folders
DATA_SUBDIR = "v3_lambda_twostep_comparison"
AGGREGATED_SUBDIR = "v3_lambda_twostep_comparison"

