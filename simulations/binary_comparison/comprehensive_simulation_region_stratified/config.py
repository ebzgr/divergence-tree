"""
Configuration for region-stratified comprehensive simulation.
"""

# Aspect ranges
NOISE_MIN = 0.001
NOISE_MAX = 10.0
DATA_SIZE_MIN = 1000
DATA_SIZE_MAX = 100000
SPARSITY_VALUES = [1, 2, 3, 4, 5, 6]

# Region-2 aspects sampled with bounded lognormal-like distribution
DISPERSION_MIN = 0.01
DISPERSION_MAX = 0.9
RARENESS_MIN = 0.01
RARENESS_MAX = 0.9
# Shape parameters for base lognormal before truncation/mapping
LOGNORMAL_MU = -1.2
LOGNORMAL_SIGMA = 1.0

# DGP defaults
DEFAULT_INTENSITY = 1.0

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
    "max_partitions": {"low": 2, "high": 200},
    "min_improvement_ratio": {"low": 0.001, "high": 0.1, "log": True},
}
DIVTREE_N_TRIALS = 30

# TwoStep
TWOSTEP_CAUSAL_FOREST_PARAMS = {
    "n_jobs": 1,
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
}
TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES = {"low": 2, "high": 200}
TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS = 30
TWOSTEP_CLASSIFICATION_TUNE_N_SPLITS = 5
TWOSTEP_TUNED_VARIANTS = {
    "twostep_tuned": "accuracy",
    "twostep_recall": "recall_region_2",
}

# Simulation
LAMBDA_VALUES = [0, 1, 2, 4, 6, 8]
BASE_RANDOM_SEED = 0
DEFAULT_N_SIMULATIONS = 10000
DEFAULT_N_JOBS = 20
DEFAULT_BATCH_SIZE = 100

# Output folders
DATA_SUBDIR = "region_stratified_lambda_twostep_comparison"
AGGREGATED_SUBDIR = "region_stratified_lambda_twostep_comparison"

