"""
Example: How to use configurations to generate data and run analysis.

This shows the basic workflow:
1. Create a configuration
2. Generate data with that config
3. Run analysis with that config
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from synthetic_data_generator import generate_data
from helper import (
    plot_latent_overview,
    print_summary,
    make_group_table_from_arrays,
    plot_six_panels_outcomes_and_pi,
)
from config import DataGenerationConfig, AnalysisConfig, FreeTrialConfig
from divtree.tree import DivergenceTree
from divtree.viz import plot_divergence_tree

# ===================== GLOBAL PATHS =====================
DATA_DIR = "data"
RESULTS_DIR = "results"
DATA_SAVE_ADDR = f"{DATA_DIR}/example_data.pickle"
# ======================================================


def run_example():
    """Run the complete example with custom configuration."""

    # 1. Create a custom configuration - Check the full list of confits in config.py
    config = FreeTrialConfig(
        data_generation=DataGenerationConfig(
            N_USERS=100000,  # Smaller dataset for faster testing
            N_FEATURES=50,
            NOISE_SCALE=1.0,
            USEFULNESS_BIPOLARITY=1,
            NOVELTY_BIPOLARITY=2,
            SUNK_COST_BIPOLARITY=2,
        ),
        analysis=AnalysisConfig(
            N_TUNING_TRIALS=20,
            LAMBDA_=1,
        ),
    )

    # 2. Generate data with the config
    print("Generating data...")
    params = {
        "n": config.data_generation.N_USERS,
        "p": config.data_generation.N_FEATURES,
        "seed": config.data_generation.RANDOM_SEED,
        "gate_indices": config.data_generation.GATE_INDICES,
        "gate_thresholds": config.data_generation.GATE_THRESHOLDS,
        "v": config.data_generation.NOISE_SCALE,
        "m_U": config.data_generation.USEFULNESS_BIPOLARITY,
        "m_N": config.data_generation.NOVELTY_BIPOLARITY,
        "m_S": config.data_generation.SUNK_COST_BIPOLARITY,
        "a_u": config.data_generation.USEFULNESS_SHORT_WEIGHT,
        "a_s": config.data_generation.SUNK_COST_LONG_WEIGHT,
        "a0": config.data_generation.BASELINE_SUBSCRIPTION,
        "a_p": config.data_generation.VALUE_SUBSCRIPTION_WEIGHT,
    }

    X, L, T, YF, YC, P_short, P_long, pi = generate_data(**params)

    # Descriptive analysis of the generated data
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print_summary(YF, T, L)

    print("\n" + "=" * 60)
    print("GROUP TABLE: Average Outcomes by Latent Variables and Trial")
    print("=" * 60)
    table = make_group_table_from_arrays(T, YF, YC, L["U"], L["N"], L["S"])
    print(table)

    # Plot outcomes and subscription probability
    fig, axes = plot_six_panels_outcomes_and_pi(T, YF, YC, pi, L, nbins=20)
    # Create comprehensive DataFrame with all data
    df = pd.DataFrame(
        {
            # Features (first 10 columns for brevity)
            **{f"X_{i}": X[:, i] for i in range(min(10, X.shape[1]))},
            # Latent variables
            "U": L["U"],
            "N": L["N"],
            "S": L["S"],
            # Treatment and outcomes
            "Treat": T,
            "YF": YF,
            "YC": YC,
            # Perceived usefulness scores
            "P_short": P_short,
            "P_long": P_long,
            "P": np.where(T == 0, P_short, P_long),  # Combined P based on treatment
            # Subscription probability
            "pi": pi,
        }
    )

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save DataFrame as pickle
    df.to_pickle(DATA_SAVE_ADDR)
    print(f"Data saved to: {DATA_SAVE_ADDR}")

    # 3. Run analysis with fixed parameters
    print("\n" + "=" * 60)
    print("TRAINING DIVERGENCE TREE")
    print("=" * 60)
    analysis_config = config.analysis

    # Fixed parameters (for DivergenceTree)
    tree_params = {
        "max_partitions": 9,
        "min_improvement_ratio": 0.01,
        "lambda_": analysis_config.LAMBDA_,
        "n_quantiles": 50,
        "random_state": analysis_config.RANDOM_STATE,
        "co_movement": (
            analysis_config.CO_MOVEMENT
            if hasattr(analysis_config, "CO_MOVEMENT")
            else "both"
        ),
        "eps_scale": 1e-8,
    }

    print(f"Tree parameters: {tree_params}")

    # Build and train tree
    print("\nTraining tree...")
    tree = DivergenceTree(**tree_params)
    tree.fit(X, T, YF, YC)

    print("Tree training complete!")

    # Show the resulting tree
    print("\n" + "=" * 60)
    print("RESULTING DIVERGENCE TREE")
    print("=" * 60)

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot and save the tree
    fig, ax = plot_divergence_tree(
        tree,
        figsize=(15, 8),
        h_spacing=3.5,
        v_spacing=2.5,
        node_width=3.0,
        node_height=1.2,
    )
    plt.savefig(f"{RESULTS_DIR}/example_tree.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Tree visualization saved to: {RESULTS_DIR}/example_tree.png")


if __name__ == "__main__":
    run_example()
