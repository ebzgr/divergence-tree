"""
Configuration for region-stratified binary-features simulation.

Fixed aspect grid with repeats; batching by (noise, sparsity, rareness).
"""

# Fixed intensity (treatment-effect magnitude/sign structure remains the same)
INTENSITY_FIXED = 1.0

# Train/val/test
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

# DGP defaults
DEFAULT_EFFECT_NOISE_STD = 0.0  # noise affects outcomes, not treatment effects

# Targeted grid values (hardcoded).
# Each combination is run N_REPEATS times with different random seeds.
DATA_SIZE_GRID = [10000, 20000, 40000, 80000, 160000, 320000]
RARENESS_GRID = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
# Sparsity values are tied to explicit category allocations in `simulation_base.generate_data_with_params`.
SPARSITY_GRID = [1, 3, 6, 10]
# Run higher noise values first.
NOISE_GRID = [10.0, 1.0, 0.1, 0.01]
N_REPEATS = 10

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
    "min_samples_split": 10,
    "min_samples_leaf": 5,
}
TWOSTEP_CAUSAL_FOREST_TUNE_PARAMS = {
    "params": {
        "min_weight_fraction_leaf": {"low": 0.0005, "high": 0.05, "log": True},
        "min_var_fraction_leaf": {"low": 0.001, "high": 0.1, "log": True},
    }
}
TWOSTEP_CAUSAL_FOREST_TUNE_N_TRIALS = 30
TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES = {"low": 2, "high": 200}
TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS = 30
TWOSTEP_TUNED_VARIANTS = {
    "twostep_tuned": "accuracy",
    "twostep_recall": "recall_region_2",
}

# Simulation
LAMBDA_VALUES = [0, 1, 2, 4, 8]
BASE_RANDOM_SEED = 0

DEFAULT_N_JOBS = 60
DEFAULT_BATCH_SIZE = 100

# Output folders (under outputs/simulations/binary_features/)
DATA_SUBDIR = "lambda_twostep_comparison"
AGGREGATED_SUBDIR = "lambda_twostep_comparison"
