# examples/basic.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path to import divtree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna
from divtree.viz import plot_divergence_tree


def make_data(n=4000, seed=0, aF=0.1, aC=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))  # we only really use X[:,0] and X[:,1]
    T = rng.integers(0, 2, size=n)

    # Your rules:
    # τF(x) > 0 iff X0 > 0; else < 0
    tauF_true = np.where(X[:, 0] > 0, +aF, -aF)

    # τC(x) > 0 iff X1 > 0; else < 0
    tauC_true = np.where(X[:, 1] > -1, +aC, -aC)

    # Base conversion probability
    p0 = np.clip(0.25 + 0.05 * np.tanh(X[:, 0]), 0.05, 0.95)
    p1 = np.clip(p0 + tauF_true, 0.01, 0.99)

    # Realized conversion YF (binary)
    YF0 = rng.binomial(1, p0)
    YF1 = rng.binomial(1, p1)
    YF = np.where(T == 1, YF1, YF0).astype(int)

    # Consumer outcome (observed only for converters)
    base_C = 0.5 + 0.2 * np.tanh(X[:, 1])
    noise = rng.normal(0, 0.35, size=n)
    YC_treat = base_C + tauC_true + noise
    YC_ctrl = base_C + noise

    YC = np.full(n, np.nan)
    mask_treat_conv = (T == 1) & (YF == 1)
    mask_ctrl_conv = (T == 0) & (YF == 1)
    YC[mask_treat_conv] = YC_treat[mask_treat_conv]
    YC[mask_ctrl_conv] = YC_ctrl[mask_ctrl_conv]

    return X, T, YF, YC


if __name__ == "__main__":
    # --- simulate data (or your real data) ---
    X, T, YF, YC = make_data(n=20000, seed=0, aF=0.20, aC=0.30)

    # --- freeze what you want fixed ---
    fixed = {
        "lambda_": 1,
        "n_quantiles": 50,
        "random_state": 0,  # for reproducibility
        "co_movement": "both",
        "eps_scale": 1e-8,
    }

    # --- tune only these two ---
    search_space = {
        "max_partitions": {"low": 4, "high": 15},
        "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
    }

    print("Running hyperparameter optimization...")
    best_params, best_loss = tune_with_optuna(
        X,
        T,
        YF,
        YC,
        fixed=fixed,
        search_space=search_space,
        n_trials=10,
        n_splits=2,
        random_state=0,
    )
    print("Best params:", best_params)
    print("Best CV loss:", best_loss)

    # --- fit final model on full data with best params ---
    print("\nTraining final tree with best parameters...")
    tree = DivergenceTree(**best_params)
    tree.fit(X, T, YF, YC)
    print(f"Root: tauF={tree.root_.tauF:.4f}, tauC={tree.root_.tauC:.4f}")

    # Print tree information
    leaf_effects = tree.leaf_effects()
    print(f"\nTree has {len(leaf_effects['leaves'])} leaves")
    print("\nLeaf effects summary:")
    for leaf in leaf_effects["leaves"][:10]:  # Show first 10 leaves
        print(
            f"  Leaf {leaf['leaf_id']}: tauF={leaf['tauF']:.4f}, "
            f"tauC={leaf['tauC']:.4f}, n={leaf['n']}"
        )

    # --- plot ---
    print("\nPlotting tree...")
    fig, ax = plot_divergence_tree(
        tree,
        figsize=(13, 7),
        h_spacing=3.2,
        v_spacing=2.4,
        node_width=2.8,
        node_height=1.1,
    )
    plt.show()
