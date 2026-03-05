"""
Run DivergenceTree on the 10000 simulations and display the tree.
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, SCRIPT_DIR)

import config
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna
from divtree.viz import plot_divergence_tree
import matplotlib.pyplot as plt

# Region 4: both treatment effects negative (tauF<=0, tauC<=0)
REGIONS_OF_INTEREST = [4]
RANDOM_SEED = 42

# Load results (uses lambda_twostep_comparison output; divtree columns are compatible)
results_file = os.path.join(SCRIPT_DIR, "output", "aggregated", "lambda_twostep_comparison", "all_simulations_results.pkl")
df = pd.read_pickle(results_file)

# Prepare data: X = simulation characteristics, T = lambda2 vs lambda0, YF = accuracy, YC = recall
x_cols = ["complexity", "noise", "data_size", "sparsity", "rareness", "covariance"]
acc_0 = "divtree_lambda0_accuracy"
recall_0 = "divtree_lambda0_recall_region_2"
acc_2 = "divtree_lambda2_region2_accuracy"
recall_2 = "divtree_lambda2_region2_recall_region_2"

valid = df[acc_0].notna() & df[recall_0].notna() & df[acc_2].notna() & df[recall_2].notna()
df = df[valid]

X_base = df[x_cols].values.astype(np.float64)
X_base[:, x_cols.index("noise")] = np.log1p(X_base[:, x_cols.index("noise")])
X_base[:, x_cols.index("data_size")] = np.log1p(X_base[:, x_cols.index("data_size")])

X_list, T_list, YF_list, YC_list = [], [], [], []
for i in range(len(df)):
    x = X_base[i]
    X_list.extend([x, x])
    T_list.extend([0, 1])
    YF_list.extend([df[acc_0].iloc[i], df[acc_2].iloc[i]])
    YC_list.extend([df[recall_0].iloc[i], df[recall_2].iloc[i]])

X = np.array(X_list)
T = np.array(T_list)
YF = np.array(YF_list)
YC = np.array(YC_list)

# Hyperparameter optimization (max_partitions, min_improvement_ratio)
fixed_params = {
    "lambda_": 2.0,
    "regions_of_interest": REGIONS_OF_INTEREST,
    "n_quantiles": config.DIVTREE_FIXED_PARAMS.get("n_quantiles", 20),
    "eps_scale": config.DIVTREE_FIXED_PARAMS.get("eps_scale", 1e-8),
    "random_state": RANDOM_SEED,
}
best_params, best_loss = tune_with_optuna(
    X, T, YF, YC,
    fixed=fixed_params,
    search_space=config.DIVTREE_SEARCH_SPACE,
    n_trials=config.DIVTREE_N_TRIALS,
    n_splits=config.DIVTREE_N_SPLITS,
    random_state=RANDOM_SEED,
    verbose=True,
)
print(f"Best params: {best_params}, best loss: {best_loss:.6f}")

tree = DivergenceTree(**best_params)
tree.fit(X, T, YF, YC)

output_dir = os.path.join(SCRIPT_DIR, "output", "aggregated", "lambda_twostep_comparison", "analysis")
os.makedirs(output_dir, exist_ok=True)

fig, ax = plot_divergence_tree(tree, figsize=(14, 9))
out_path = os.path.join(output_dir, "meta_tree.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")
