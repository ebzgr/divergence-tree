# Comprehensive Simulation Framework for Divergence Tree Comparison

## Overview

This simulation framework systematically compares different **DivergenceTree** and **TwoStepDivergenceTree** configurations across different data generation settings. The framework supports multiple analysis modes:

1. **One-at-a-time Analysis** (`run_simulation.py`): Vary one aspect at a time while keeping others fixed at baseline values
2. **Unified Comparison** (`unified_comparison.py`): **[RECOMMENDED]** Compare all 8 methods on the same datasets (combines lambda and method comparisons)
3. **Lambda Comparison** (`lambda_comparison.py`): Compare DivergenceTree with different lambda values (0, 1, 2, 3, 4) using random space sampling
4. **Method Comparison** (`method_comparison.py`): Compare DivergenceTree and TwoStepDivergenceTree variants using random space sampling
5. **Random Space Comparison** (`random_space_comparison.py`): [DEPRECATED] Legacy script for lambda comparison (missing lambda=3). Use `lambda_comparison.py` instead.

## Purpose

The simulation is designed to:

- Compare the performance of different DivergenceTree configurations (varying lambda and regions_of_interest) across various data generation settings
- Understand how different aspects of data generation (complexity, noise, sparsity, rareness, covariance, data size) affect algorithm performance
- Provide statistical confidence through multiple replications (Monte Carlo approach)
- Generate comprehensive evaluation metrics beyond simple accuracy
- Enable robust statistical comparison between different method configurations

## Data Generation Aspects

The simulation varies 6 key aspects of data generation:

### 1. Complexity
- **Parameter**: `m_firm = m_user` (number of activating combinations)
- **Range**: Uniform between 1 and 30
- **Effect**: As complexity increases, more combinations activate treatment effects

### 2. Noise
- **Parameter**: `effect_noise_std` (noise in treatment effects)
- **Range**: Log-uniform between 0.001 and 10.0
- **Effect**: Higher noise makes treatment effects harder to detect

### 3. Data Size
- **Parameter**: `n_users_train` (number of training samples)
- **Range**: Log-uniform between 1000 and 200000
- **Effect**: Larger datasets provide more information for learning

### 4. Sparsity
- **Parameter**: `k` (number of categorical variables)
- **Values**: [1, 2, 3, 4, 5, 6]
- **Effect**: As k increases, sparsity increases. `n_categories = [60//k]*k` to keep total features = 60

### 5. Rareness
- **Parameter**: `positive_ratio` (proportion of observations in activating combinations)
- **Range**: Uniform between 0.01 and 0.99
- **Effect**: Lower values mean activating combinations are rarer

### 6. Covariance
- **Parameter**: `similarity` (proportion of combinations shared between firm and user effects)
- **Range**: Uniform between 0.0 and 1.0
- **Effect**: Higher similarity means more overlap between firm and user activating combinations

## Default Baseline Values

When running one-at-a-time analysis, other aspects are kept at these baseline values:

- `n_users_train = 20000`
- `n_users_test = 10000`
- `k = 6`
- `n_categories = [10]*6`
- `m_firm = 4`
- `m_user = 4`
- `similarity = 0.5`
- `intensity = 1`
- `effect_noise_std = 1`
- `firm_outcome_noise_std = 1`
- `user_outcome_noise_std = 1`
- `positive_ratio = 0.5`

## Evaluation Metrics

The simulation computes comprehensive evaluation metrics for each algorithm:

### Classification Metrics
- **Overall Accuracy**: Proportion of correct predictions
- **Per-Region Accuracy**: Accuracy for each region type (1, 2, 3, 4)
- **False Negative Rate (FNR) for Region 2**: Proportion of true region 2 observations incorrectly predicted as other regions
- **F1 Score per Region**: F1 score for each region type (1, 2, 3, 4)
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Matthews Correlation Coefficient (MCC)**: Correlation between true and predicted classes

### Information-Theoretic Metrics
- **RIG (Relative Information Gain)**: 
  - RIG = (H_baseline - H_model) / H_baseline
  - Where H is entropy: H = -Σ p_i * log(p_i)
  - Baseline: uniform distribution (1/4 for each class)
  - Model: predicted class distribution

### Complexity Metrics
- **Number of Leaves**: Tree complexity (number of terminal nodes)
- **Runtime**: Computation time in seconds

## Methods

### 1. One-at-a-time Analysis (`run_simulation.py`)

Varies one aspect at a time while keeping others fixed at baseline values.

**Usage:**
```bash
python run_simulation.py --mode one_at_a_time --n_replications 50
```

**Output:**
- `output/data/one_at_a_time/`: Individual simulation data
- `output/aggregated/one_at_a_time/`: Aggregated results per aspect
  - `{aspect}_results.pkl`: DataFrame with all replications for each aspect value

### 2. Unified Comparison (`unified_comparison.py`) **[RECOMMENDED]**

Compares all 8 methods on the same generated datasets, allowing for comprehensive comparison across all methods. This is the recommended approach as it ensures fair comparison by using identical datasets for all methods.

**Methods (8 total):**
- DivTree with λ=0
- DivTree with λ=1, regions_of_interest=[2]
- DivTree with λ=2, regions_of_interest=[2]
- DivTree with λ=3, regions_of_interest=[2]
- DivTree with λ=4, regions_of_interest=[2]
- TwoStepDivergenceTree (unconstrained, no max leaves)
- TwoStepDivergenceTree (constrained, max leaves = DivTree λ=0 leaves)
- TwoStepDivergenceTree (constrained + FNR scoring, max leaves = DivTree λ=0 leaves)

**Usage:**
```bash
python unified_comparison.py
```

**Configuration:**
- Number of simulations: 10000 (default, configurable in script)
- Batch size: 1000 (saves incrementally)
- Parallel jobs: All CPUs minus 1 (leaves 1 core free)

**Output:**
- `output/data/unified_comparison/`: Individual simulation data
- `output/aggregated/unified_comparison/`: Aggregated results
  - `all_simulations_results.pkl`: DataFrame with all simulation results

### 3. Lambda Comparison (`lambda_comparison.py`)

Compares DivergenceTree with different lambda values using random space sampling. Compares 5 configurations:
- λ=0 (baseline, no region weighting)
- λ=1, regions_of_interest=[2]
- λ=2, regions_of_interest=[2]
- λ=3, regions_of_interest=[2]
- λ=4, regions_of_interest=[2]

**Usage:**
```bash
python lambda_comparison.py
```

**Configuration:**
- Number of simulations: 10000 (default, configurable in script)
- Batch size: 1000 (saves incrementally)
- Parallel jobs: All CPUs minus 1 (leaves 1 core free)

**Output:**
- `output/data/lambda_comparison/`: Individual simulation data
- `output/aggregated/lambda_comparison/`: Aggregated results
  - `all_simulations_results.pkl`: DataFrame with all simulation results

### 4. Method Comparison (`method_comparison.py`)

Compares DivergenceTree and TwoStepDivergenceTree variants using random space sampling. Compares 5 methods:
- DivTree with λ=0 (baseline)
- DivTree with λ=2, regions_of_interest=[2]
- TwoStep (unconstrained, no max leaves)
- TwoStep (constrained, max leaves = DivTree λ=0 leaves)
- TwoStep (constrained + FNR scoring, max leaves = DivTree λ=0 leaves, optimized for FNR of region 2)

**Usage:**
```bash
python method_comparison.py
```

**Configuration:**
- Number of simulations: 10000 (default, configurable in script)
- Batch size: 1000 (saves incrementally)
- Parallel jobs: All CPUs minus 1 (leaves 1 core free)

**Output:**
- `output/data/method_comparison/`: Individual simulation data
- `output/aggregated/method_comparison/`: Aggregated results
  - `all_simulations_results.pkl`: DataFrame with all simulation results

### 5. Analysis Script (`analyze_results.py`)

Unified analysis script that works with both lambda_comparison and method_comparison results. Generates:
- Aspect comparison plots (performance across all aspects for key metrics: accuracy, FNR region 2, F1 region 2)
- Statistical test matrices (pairwise significance tests for all method pairs)
- LaTeX tables with statistical test results
- Decision trees for method selection (one tree per metric)

**Usage:**
```bash
# Analyze lambda comparison results
python analyze_results.py
# Or specify paths and simulation type:
python -c "from analyze_results import analyze_results; analyze_results('output/aggregated/lambda_comparison/all_simulations_results.pkl', 'output/aggregated/lambda_comparison/analysis', simulation_type='lambda_comparison')"

# Analyze method comparison results
python -c "from analyze_results import analyze_results; analyze_results('output/aggregated/method_comparison/all_simulations_results.pkl', 'output/aggregated/method_comparison/analysis', simulation_type='method_comparison')"
```

**Output:**
- `output/aggregated/{simulation_type}/analysis/plots/`: Comparison plots and statistical test matrices
- `output/aggregated/{simulation_type}/analysis/tables/`: Statistical test results (CSV and LaTeX)
- `output/aggregated/{simulation_type}/analysis/trees/`: Decision tree visualizations

### 6. Legacy Scripts (Deprecated)

- **`random_space_comparison.py`**: [DEPRECATED] Legacy lambda comparison script (missing lambda=3). Use `lambda_comparison.py` instead.
- **`analyze_random_space_results.py`**: [DEPRECATED] Legacy analysis script. Use `analyze_results.py` instead.

## Output Structure

```
comprehensive_simulation/
├── config.py                          # Configuration
├── metrics.py                         # Evaluation metrics
├── utils.py                           # Helper functions
├── simulation_base.py                 # Base framework (MethodRunner classes, shared utilities)
├── run_simulation.py                  # One-at-a-time analysis
├── unified_comparison.py              # [RECOMMENDED] Unified comparison (all 8 methods)
├── lambda_comparison.py               # Lambda comparison simulation
├── method_comparison.py               # Method comparison simulation
├── analyze_results.py                 # Unified analysis script
├── random_space_comparison.py         # [DEPRECATED] Legacy lambda comparison
├── analyze_random_space_results.py    # [DEPRECATED] Legacy analysis script
├── output/
│   ├── data/
│   │   ├── one_at_a_time/             # One-at-a-time simulation data
│   │   ├── unified_comparison/        # [RECOMMENDED] Unified comparison simulation data
│   │   ├── lambda_comparison/         # Lambda comparison simulation data
│   │   ├── method_comparison/         # Method comparison simulation data
│   │   └── region2tuning_test/        # [LEGACY] Random space simulation data
│   └── aggregated/
│       ├── one_at_a_time/             # One-at-a-time aggregated results
│       ├── unified_comparison/        # [RECOMMENDED] Unified comparison aggregated results
│       │   └── analysis/              # Analysis outputs
│       │       ├── plots/             # Comparison plots and matrices
│       │       ├── tables/            # Statistical test results
│       │       └── trees/             # Decision tree visualizations
│       ├── lambda_comparison/         # Lambda comparison aggregated results
│       │   └── analysis/              # Analysis outputs
│       │       ├── plots/             # Comparison plots and matrices
│       │       ├── tables/            # Statistical test results
│       │       └── trees/             # Decision tree visualizations
│       ├── method_comparison/         # Method comparison aggregated results
│       │   └── analysis/              # Analysis outputs
│       │       ├── plots/             # Comparison plots and matrices
│       │       ├── tables/            # Statistical test results
│       │       └── trees/             # Decision tree visualizations
│       └── region2tuning_test/        # [LEGACY] Random space aggregated results
```

## DataFrame Structure

Each aggregated DataFrame contains:

**Aspect Columns:**
- `simulation_id`: Unique identifier
- `complexity`: m_firm = m_user value
- `noise`: effect_noise_std value
- `data_size`: n_users_train value
- `sparsity`: k value
- `rareness`: positive_ratio value
- `covariance`: similarity value

**Method Metrics (for each method):**
- `{method}_accuracy`
- `{method}_acc_region_1`, `{method}_acc_region_2`, `{method}_acc_region_3`, `{method}_acc_region_4`
- `{method}_fnr_region_2`
- `{method}_f1_region_1`, `{method}_f1_region_2`, `{method}_f1_region_3`, `{method}_f1_region_4`
- `{method}_balanced_accuracy`
- `{method}_mcc`
- `{method}_rig`
- `{method}_n_leaves`
- `{method}_runtime`

## Usage Workflow

### Step 1: Run Simulations

```bash
# One-at-a-time analysis
python run_simulation.py --mode one_at_a_time --n_replications 50

# Unified comparison (RECOMMENDED: compares all 8 methods on same datasets)
python unified_comparison.py

# Lambda comparison (compares lambda values 0, 1, 2, 3, 4)
python lambda_comparison.py

# Method comparison (compares DivTree and TwoStep variants)
python method_comparison.py
```

### Step 2: Analyze Results

```bash
# Analyze lambda comparison results (auto-detects simulation type)
python -c "from analyze_results import analyze_results; analyze_results('output/aggregated/lambda_comparison/all_simulations_results.pkl', 'output/aggregated/lambda_comparison/analysis')"

# Analyze method comparison results
python -c "from analyze_results import analyze_results; analyze_results('output/aggregated/method_comparison/all_simulations_results.pkl', 'output/aggregated/method_comparison/analysis')"

# Analyze unified comparison results (RECOMMENDED)
python -c "from analyze_results import analyze_results; analyze_results('output/aggregated/unified_comparison/all_simulations_results.pkl', 'output/aggregated/unified_comparison/analysis')"
```

### Step 3: Load and Analyze DataFrames

```python
import pandas as pd

# Load aggregated results
df = pd.read_pickle("output/aggregated/region2tuning_test/all_simulations_results.pkl")

# Analyze specific metric
print(df[["complexity", "divtree_lambda0_accuracy", "divtree_lambda2_region2_accuracy"]].groupby("complexity").mean())

# Compare methods
difference = df["divtree_lambda2_region2_accuracy"] - df["divtree_lambda0_accuracy"]
print(f"Mean difference: {difference.mean():.4f}")
```

## Dependencies

- **pandas**: DataFrame management
- **numpy**: Numerical operations
- **scikit-learn**: Metrics (accuracy, F1, MCC, balanced_accuracy), DecisionTreeClassifier
- **scipy**: Statistical tests (t-test)
- **matplotlib**: Visualizations
- **optuna**: Hyperparameter tuning
- **joblib**: Parallel processing
- **Existing codebase modules**: 
  - `divtree.tree.DivergenceTree`
  - `divtree.tune.tune_with_optuna`
  - `binary_data_generator.generate_binary_comparison_data`

## Notes

- **Random Seeds**: Each simulation uses a unique seed based on simulation_id for reproducibility
- **Error Handling**: If a single simulation fails, it continues and marks metrics as NaN
- **Incremental Saving**: Random space comparison saves results incrementally after each batch
- **Memory**: Large simulations may require significant memory for storing all results
- **Parallelization**: Uses joblib for parallel execution, leaving 1 CPU core free for system tasks

## Configuration

All configuration is in `config.py`:
- Search spaces for each aspect
- Default baseline values
- Hyperparameters for algorithms
- Number of replications

Modify `config.py` to change simulation settings.
