"""
Configuration for comprehensive simulation framework.

Defines search spaces for 5 aspects of data generation and default baseline values.
"""

# ===================== SEARCH SPACES FOR ASPECTS =====================

# Complexity: m_firm = m_user values (uniform between 1 and 30)
COMPLEXITY_MIN = 1
COMPLEXITY_MAX = 30

# Noise: effect_noise_std values
# Logarithmic (log-uniform) distribution between NOISE_MIN and NOISE_MAX
# Note: Using 0.001 instead of 0 because log(0) is undefined
NOISE_MIN = 0.001
NOISE_MAX = 10.0

# Data Size: number of training users (test will be half of this)
# Logarithmic (log-uniform) distribution between DATA_SIZE_MIN and DATA_SIZE_MAX
DATA_SIZE_MIN = 1000
DATA_SIZE_MAX = 100000

# Sparsity: k values (with n_categories = [60//k]*k to keep total features = 60)
SPARSITY_VALUES = [1, 2, 3, 4, 5, 6]

# Rareness: positive_ratio values (uniform between 0.01 and 0.99)
RARENESS_MIN = 0.01
RARENESS_MAX = 0.99

# Covariance: similarity values (uniform between 0 and 1)
COVARIANCE_MIN = 0.0
COVARIANCE_MAX = 1.0

# ===================== DEFAULT BASELINE VALUES =====================
# From simulate_binary.py step1_generate_and_save_data()

DEFAULT_N_USERS_TRAIN = 20000
DEFAULT_N_USERS_TEST = 10000
DEFAULT_K = 6
DEFAULT_N_CATEGORIES = [10] * 6
DEFAULT_M_FIRM = 4
DEFAULT_M_USER = 4
DEFAULT_SIMILARITY = 0.5
DEFAULT_INTENSITY = 1
DEFAULT_EFFECT_NOISE_STD = 1
DEFAULT_FIRM_OUTCOME_NOISE_STD = 1
DEFAULT_USER_OUTCOME_NOISE_STD = 1
DEFAULT_POSITIVE_RATIO = 0.5

# ===================== TRAIN / VALIDATION / TEST SPLIT =====================
TRAIN_FRAC = 0.5
VAL_FRAC = 0.25
TEST_FRAC = 0.25

# ===================== DEFAULT NUMBER OF REPLICATIONS =====================
DEFAULT_N_REPLICATIONS = 50

# ===================== FIXED HYPERPARAMETERS =====================
# From simulate_binary.py step2 and step3

# DivergenceTree hyperparameters
DIVTREE_FIXED_PARAMS = {
    "n_quantiles": 20,
    "eps_scale": 1e-8,
}

DIVTREE_SEARCH_SPACE = {
    "max_partitions": {"low": 2, "high": 200},
    "min_improvement_ratio": {"low": 0.001, "high": 0.1, "log": True},
}

DIVTREE_N_TRIALS = 30
DIVTREE_N_SPLITS = 2

# TwoStepDivergenceTree hyperparameters
TWOSTEP_CAUSAL_FOREST_PARAMS = {
    "n_jobs": 1,  # Single-threaded for reproducible CPU-time measurement (works on Linux)
    "n_estimators": 100,
    "max_depth": 20,  # Limit tree depth to reduce memory (None = unbounded, can cause OOM)
    "min_samples_split": 10,
    "min_samples_leaf": 5,
}

TWOSTEP_CLASSIFICATION_TUNE_MAX_LEAF_NODES = {"low": 2, "high": 200}

TWOSTEP_CLASSIFICATION_TUNE_N_TRIALS = 30
TWOSTEP_CLASSIFICATION_TUNE_N_SPLITS = 5

# DivTreeForest (bagging + classification tree)
DIVTREE_FOREST_LAMBDA = 2
DIVTREE_FOREST_N_ESTIMATORS = 50
DIVTREE_FOREST_MAX_SAMPLES = 0.63
DIVTREE_FOREST_MAX_FEATURES = 1.0

# ===================== BASE RANDOM SEED =====================
BASE_RANDOM_SEED = 0

