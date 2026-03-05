# Lambda Comparison Simulation

## Overview

This simulation compares **DivergenceTree** configurations with different lambda values across diverse data generation settings. The simulation systematically evaluates how the regularization parameter λ affects performance when focusing on specific regions of interest (Region 2: firm positive, consumer negative).

## Data Generation Process

### Categorical Variable Generation

The data generation process uses `binary_data_generator.py` to create synthetic data with categorical features. The process works as follows:

1. **Categorical Structure**: 
   - We have `k` categorical variables, where each variable `i` has `n_categories[i]` possible categories
   - Each observation belongs to exactly one combination of categories across all `k` variables
   - The total number of possible combinations is the product: `n_categories[0] × n_categories[1] × ... × n_categories[k-1]`

2. **One-Hot Encoding**:
   - Each categorical variable is one-hot encoded into binary features
   - The total number of binary features = `sum(n_categories)`
   - Each observation has exactly `k` features set to 1 (one per categorical variable), and all others set to 0
   - Example: If `k=3` with `n_categories=[3, 4, 5]`, we get:
     - Total combinations: 3×4×5 = 60 possible category combinations
     - Total binary features: 3+4+5 = 12 features
     - Each observation has exactly 3 features set to 1 (one from each variable)

3. **Feature Count**:
   - **Important**: The total number of features is fixed at **60** across all simulations
   - This is achieved by setting `n_categories = [60//k]*k` where `k` is the sparsity parameter
   - For example:
     - `k=1`: `n_categories=[60]` → 60 features, 60 combinations
     - `k=2`: `n_categories=[30, 30]` → 60 features, 900 combinations
     - `k=6`: `n_categories=[10, 10, 10, 10, 10, 10]` → 60 features, 1,000,000 combinations

### Treatment Effect Structure

The data generating process follows the standard structure: **Y = μ(X) + τ(X) × T + ε**

1. **Baseline Outcomes** (`μ(X)`):
   - Random coefficients are assigned to all binary features
   - Baseline outcomes are computed as linear combinations: `μ(X) = X @ baseline_coef`

2. **Treatment Effects** (`τ(X)`):
   - Specific category combinations are designated as "activating" for treatment effects
   - Observations in activating combinations receive treatment effect `+intensity`
   - Observations in non-activating combinations receive treatment effect `-intensity`
   - Separate sets of activating combinations exist for firm effects (`m_firm` combinations) and consumer effects (`m_user` combinations)

3. **Region Types**:
   - Region types are determined by the signs of treatment effects (before noise is added):
     - **Region 1**: `τF > 0` and `τC > 0` (both positive — win-win)
     - **Region 2**: `τF > 0` and `τC ≤ 0` (firm positive, consumer negative — trade-off favoring firm)
     - **Region 3**: `τF ≤ 0` and `τC > 0` (firm negative, consumer positive — trade-off favoring consumer)
     - **Region 4**: `τF ≤ 0` and `τC ≤ 0` (both negative — lose-lose)

4. **Outcome Generation**:
   - Treatment assignment `T` is random with probability 0.5
   - Outcomes are generated as: `Y = μ(X) + τ(X) × T + ε`
   - Noise `ε` is added to the outcomes (separate noise for firm and consumer outcomes)

### The 5 Factors

The simulation varies five key factors that control different aspects of data complexity and difficulty:

**Note**: The total number of features is fixed at 60 across all simulations. This is achieved by setting `n_categories = [60//k]*k` where `k` is the sparsity parameter.

#### 1. Complexity (`m_firm = m_user`)

- **Range**: Uniform [1, 30]
- **Maps to**: Number of activating combinations for both firm and consumer treatment effects
- **Effect**: Higher complexity means more complex treatment effect structure, requiring trees with more nodes to capture the heterogeneity. As the number of activating combinations increases, the algorithm needs to identify more distinct patterns in the data.

#### 2. Noise (`effect_noise_std`)

- **Range**: Log-uniform [0.001, 10.0]
- **Maps to**: Standard deviation of noise added to treatment effects (before outcomes are generated)
- **Effect**: Higher noise makes it harder to find treatment effects, as the signal-to-noise ratio decreases. The noise is added to the treatment effects themselves, which can flip signs and obscure the true underlying patterns.

#### 3. Sparsity (`k`)

- **Values**: [1, 2, 3, 4, 5, 6]
- **Maps to**: Number of categorical variables
- **Effect**: `n_categories = [60//k]*k` to keep total features = 60. Higher sparsity (higher k) means fewer categories per variable, which requires deeper trees to capture the treatment effects since the splitting structure needs to go through more levels. For example, with `k=6`, the tree must split on 6 different variables to fully capture a combination, whereas with `k=1`, a single split can identify a category.

#### 4. Rareness (`positive_ratio`)

- **Range**: Uniform [0.01, 0.99]
- **Maps to**: Proportion of observations forced into activating combinations
- **Effect**: Lower rareness makes it harder to find segments of interest. For example, if rareness = 0.01, only 1% of observations have positive treatment effects, making it very difficult to identify specific regions. The algorithm has fewer positive examples to learn from, and the signal is overwhelmed by the majority of negative examples.

#### 5. Covariance (`similarity`)

- **Range**: Uniform [0.0, 1.0]
- **Maps to**: Proportion of combinations shared between firm and user activating combinations
- **Effect**: Higher covariance means firm and consumer side signals are aligned in the same variables and same directions. This makes it easier for the algorithm to find specific regions because both outcomes are responding to the same underlying patterns. When similarity is high, splits that help identify firm effects also help identify consumer effects, creating a synergistic signal.

### Data Generation Flow

The complete data generation process (`generate_binary_comparison_data()`) follows these steps:

1. **Generate all possible category combinations**: Create the full set of all possible combinations across all categorical variables
2. **Select activating combinations**: 
   - Randomly select `m_firm` combinations for firm effects
   - Randomly select `m_user` combinations for consumer effects
   - Ensure `similarity × m_user` combinations are shared between firm and consumer
3. **Assign observations to combinations**:
   - Force `positive_ratio × n_users` observations into activating combinations (randomly distributed)
   - Force remaining observations into non-activating combinations
4. **Generate baseline outcomes**: Create random coefficients and compute `μ(X) = X @ baseline_coef` for both firm and consumer
5. **Assign treatment effects**: 
   - Set `τ = +intensity` for activated combinations
   - Set `τ = -intensity` for non-activated combinations
6. **Add noise to treatment effects**: Add Gaussian noise with standard deviation `effect_noise_std` to both firm and consumer effects
7. **Compute region types**: Determine region type (1-4) based on signs of `τF` and `τC` after noise
8. **Generate outcomes**: 
   - Randomly assign treatment `T ~ Bernoulli(0.5)`
   - Generate `YF = μF(X) + τF × T + εF`
   - Generate `YC = μC(X) + τC × T + εC`

## Lambda Comparison Simulation

### Overview

The lambda comparison simulation evaluates how different values of the regularization parameter λ affect DivergenceTree performance. All methods are evaluated on the same datasets to ensure fair comparison. The simulation uses random space sampling across the 5 factors described above.

### Methods Compared

The simulation compares **8 DivergenceTree configurations**:

1. **λ=0** (baseline, no region weighting)
   - `regions_of_interest=None`
   - No co-movement term in objective function

2. **λ=1, regions_of_interest=[2]**
   - Focuses on Region 2 (firm positive, consumer negative)
   - Co-movement term weighted by λ=1

3. **λ=2, regions_of_interest=[2]**
   - Same focus, higher regularization weight

4. **λ=3, regions_of_interest=[2]**
   - Same focus, even higher regularization weight

5. **λ=4, regions_of_interest=[2]**
   - Same focus, higher regularization weight

6. **λ=6, regions_of_interest=[2]**
   - Same focus, higher regularization weight

7. **λ=8, regions_of_interest=[2]**
   - Same focus, higher regularization weight

8. **λ=10, regions_of_interest=[2]**
   - Same focus, highest regularization weight

All methods with λ>0 focus on Region 2, which represents trade-off scenarios where firm benefits come at consumer expense.

### Simulation Process

The simulation runs **10,000 simulations** by default (configurable in the script). Each simulation follows these steps:

1. **Random Sampling**: Randomly sample one value from each factor's range:
   - Complexity: Uniform [1, 30]
   - Noise: Log-uniform [0.001, 10.0]
   - Data size: Log-uniform [1000, 200000] (training set size; test set is half)
   - Sparsity: Random choice from [1, 2, 3, 4, 5, 6]
   - Rareness: Uniform [0.01, 0.99]
   - Covariance: Uniform [0.0, 1.0]

2. **Data Generation**: Generate train and test datasets using the sampled parameters

3. **Data Persistence**: 
   - Check if data already exists for this simulation ID
   - If exists, load the saved data (ensures reproducibility and efficiency)
   - If not, generate new data and save it for future use

4. **Method Execution**: Run all 8 lambda methods on the same dataset:
   - Each method fits a DivergenceTree with the specified λ and `regions_of_interest`
   - All methods use the same hyperparameter tuning process (Optuna with cross-validation)

5. **Metric Computation**: Compute evaluation metrics for each method:
   - Classification metrics (accuracy, F1, precision, recall, etc.)
   - Per-region metrics
   - Complexity metrics (number of leaves, runtime)

6. **Incremental Saving**: Save results after each batch of 1000 simulations to prevent data loss

### Evaluation Metrics

The simulation computes comprehensive evaluation metrics for each method:

#### Classification Metrics
- **Overall Accuracy**: Proportion of correct region type predictions
- **Per-Region Accuracy**: Accuracy for each region type (1, 2, 3, 4)
- **False Negative Rate (FNR) for Region 2**: Proportion of true Region 2 observations incorrectly predicted as other regions
- **Precision and Recall for Region 2**: Precision and recall specifically for Region 2
- **F1 Score per Region**: F1 score for each region type (1, 2, 3, 4)
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Matthews Correlation Coefficient (MCC)**: Correlation between true and predicted classes

#### Complexity Metrics
- **Number of Leaves**: Tree complexity (number of terminal nodes)
- **Runtime**: Computation time in seconds

### Output Structure

```
comprehensive_simulation/
├── lambda_comparison.py              # Main simulation script (DivTree + TwoStep)
├── analyze_lambda_comparison.py      # Analysis script
├── output/
│   ├── data/
│   │   └── lambda_twostep_comparison/  # Individual simulation data
│   │       └── simulation_XXXXXX/     # Per-simulation directories
│   │           ├── train_data.pkl     # Training data (DataFrame)
│   │           ├── val_data.pkl       # Validation data (DataFrame)
│   │           ├── test_data.pkl      # Test data (DataFrame)
│   │           └── functional_form.pickle  # Functional form info
│   └── aggregated/
│       └── lambda_twostep_comparison/  # Aggregated results
│           ├── all_simulations_results.pkl  # Main results DataFrame
│           └── analysis/             # Analysis outputs (if run)
│               ├── plots/             # Comparison plots
│               └── tables/            # Summary tables
```

### Usage

#### Running the Simulation

```bash
cd simulations/binary_comparison/comprehensive_simulation
python lambda_comparison.py
```

**Configuration** (in `lambda_comparison.py`):
- Number of simulations: 10,000 (default, configurable)
- Batch size: 1,000 (saves incrementally)
- Parallel jobs: All CPUs minus 1 (leaves 1 core free for system tasks)

#### Analyzing Results

After running the simulation, use the analysis script to generate plots and statistical tests:

```bash
python analyze_lambda_comparison.py
```

Or programmatically:

```python
from analyze_lambda_comparison import create_lambda_comparison_plots

df = pd.read_pickle("output/aggregated/lambda_twostep_comparison/all_simulations_results.pkl")
create_lambda_comparison_plots(
    df, 
    output_dir="output/aggregated/lambda_twostep_comparison/analysis"
)
```

The analysis script:
- Creates plots showing each metric vs. lambda (x-axis: lambda, y-axis: metric value)
- Performs paired t-tests comparing:
  - Lambda 0 vs. all other lambdas
  - Lambda 2 vs. all other lambdas
- Displays significance markers (`*`, `**`, `***`) on plots
- Adjusts y-axis limits for 0-1 metrics to show only relevant ranges

### DataFrame Structure

The aggregated results DataFrame (`all_simulations_results.pkl`) contains:

**Aspect Columns**:
- `simulation_id`: Unique identifier
- `complexity`: m_firm = m_user value
- `noise`: effect_noise_std value
- `data_size`: n_users_train value
- `sparsity`: k value
- `rareness`: positive_ratio value
- `covariance`: similarity value

**Method Metrics** (for each of the 8 methods):
- `divtree_lambda0_{metric}` or `divtree_lambda{λ}_region2_{metric}`
- Where `{metric}` includes: `accuracy`, `acc_region_1`, `acc_region_2`, `acc_region_3`, `acc_region_4`, `fnr_region_2`, `precision_region_2`, `recall_region_2`, `f1_region_1`, `f1_region_2`, `f1_region_3`, `f1_region_4`, `balanced_accuracy`, `mcc`, `n_leaves`, `runtime`, `cpu_time`

## Dependencies

- **pandas**: DataFrame management
- **numpy**: Numerical operations
- **scikit-learn**: Metrics and DecisionTreeClassifier
- **scipy**: Statistical tests (paired t-test)
- **matplotlib**: Visualizations
- **optuna**: Hyperparameter tuning
- **joblib**: Parallel processing
- **Existing codebase modules**: 
  - `divtree.tree.DivergenceTree`
  - `divtree.tune.tune_with_optuna`
  - `binary_data_generator.generate_binary_comparison_data`

## Notes

- **Random Seeds**: Each simulation uses a unique seed based on `simulation_id` for reproducibility
- **Error Handling**: If a single simulation fails, it continues and marks metrics as NaN
- **Incremental Saving**: Results are saved after each batch of 1000 simulations
- **Memory**: Large simulations may require significant memory for storing all results
- **Parallelization**: Uses joblib for parallel execution, leaving 1 CPU core free for system tasks
- **Data Reuse**: Generated data is saved and reused if available, improving efficiency for repeated runs
