"""
Example: How to use the new comparison data generator with divergence tree.

This shows the complete workflow with continuous outcomes:
1. Generate data with continuous firm and user outcomes (both always observed)
2. Optimize hyperparameters using cross-validation
3. Train divergence tree with best parameters
4. Train alternative two-step method
5. Compare both methods
6. Visualize the tree

The workflow is split into 4 independent steps that can be run separately:
Step 1: Generate and save data
Step 2: Load data, run DivergenceTree, save results
Step 3: Load data, run TwoStepDivergenceTree, save results
Step 4: Load results and compare
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from typing import Dict, Any, Tuple, Optional
from data_generator import generate_comparison_data, get_data_summary

# Add the src directory to the path to import divtree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna as tune_divtree
from divtree.viz import plot_divergence_tree
from twostepdivtree.tree import TwoStepDivergenceTree
from comparison_helper import compare_methods, print_comparison

# ===================== GLOBAL PATHS =====================
# Use data folder in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "comparison_data.pickle")
DIVTREE_RESULTS_FILE = os.path.join(DATA_DIR, "divtree_results.pickle")
TWOSTEP_RESULTS_FILE = os.path.join(DATA_DIR, "twostep_results.pickle")

# ===================== GLOBAL RANDOM SEED =====================
# Single random seed for all randomness in the simulation
RANDOM_SEED = 0
# ======================================================


def generate_data(
    n_users_train: int = 20000,
    n_users_test: int = 5000,
    n_features: int = 10,
    firm_outcome_base: float = 0.0,
    firm_outcome_noise_std: float = 1.0,
    user_outcome_base: float = 0.0,
    user_outcome_noise_std: float = 1.0,
    random_seed: int = RANDOM_SEED,
    firm_effect_strength: float = 1.0,
    user_effect_strength: float = 1.0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate synthetic training and test data with continuous outcomes.

    Parameters
    ----------
    n_users_train : int, default=20000
        Number of training observations to generate.
    n_users_test : int, default=5000
        Number of test observations to generate.
    n_features : int, default=10
        Number of features.
    firm_outcome_base : float, default=0.0
        Base value for firm outcome.
    firm_outcome_noise_std : float, default=1.0
        Standard deviation of noise for firm outcome.
    user_outcome_base : float, default=0.0
        Base value for user outcome.
    user_outcome_noise_std : float, default=1.0
        Standard deviation of noise for user outcome.
    random_seed : int, default=RANDOM_SEED
        Random seed for both training and test data.
    firm_effect_strength : float, default=1.0
        Direct firm effect strength.
    user_effect_strength : float, default=1.0
        Direct user effect strength.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix.
    T_train : np.ndarray
        Training treatment indicator.
    YF_train : np.ndarray
        Training firm outcome.
    YC_train : np.ndarray
        Training consumer outcome.
    region_type_train : np.ndarray
        Training true region type labels (1-4).
    X_test : np.ndarray
        Test feature matrix.
    T_test : np.ndarray
        Test treatment indicator.
    YF_test : np.ndarray
        Test firm outcome.
    YC_test : np.ndarray
        Test consumer outcome.
    region_type_test : np.ndarray
        Test true region type labels (1-4).
    """
    print("Generating training and test data with continuous outcomes...")

    # Generate training data
    X_train, T_train, YF_train, YC_train, tauF_train, tauC_train, region_type_train = (
        generate_comparison_data(
            n_users=n_users_train,
            n_features=n_features,
            firm_outcome_base=firm_outcome_base,
            firm_outcome_noise_std=firm_outcome_noise_std,
            user_outcome_base=user_outcome_base,
            user_outcome_noise_std=user_outcome_noise_std,
            random_seed=random_seed,
            firm_effect_strength=firm_effect_strength,
            user_effect_strength=user_effect_strength,
        )
    )

    # Generate test data (using the same random seed)
    X_test, T_test, YF_test, YC_test, tauF_test, tauC_test, region_type_test = (
        generate_comparison_data(
            n_users=n_users_test,
            n_features=n_features,
            firm_outcome_base=firm_outcome_base,
            firm_outcome_noise_std=firm_outcome_noise_std,
            user_outcome_base=user_outcome_base,
            user_outcome_noise_std=user_outcome_noise_std,
            random_seed=random_seed,
            firm_effect_strength=firm_effect_strength,
            user_effect_strength=user_effect_strength,
        )
    )

    # Print ground truth region type distribution for training
    print(f"\nTraining ground truth region type distribution:")
    for rt in [1, 2, 3, 4]:
        count = (region_type_train == rt).sum()
        print(
            f"  Region {rt}: {count} observations ({100*count/len(region_type_train):.2f}%)"
        )

    # Print ground truth region type distribution for test
    print(f"\nTest ground truth region type distribution:")
    for rt in [1, 2, 3, 4]:
        count = (region_type_test == rt).sum()
        print(
            f"  Region {rt}: {count} observations ({100*count/len(region_type_test):.2f}%)"
        )

    # Verify both outcomes are always observed
    print(f"\nTraining data verification:")
    print(f"  YF has NaN: {np.isnan(YF_train).any()}")
    print(f"  YC has NaN: {np.isnan(YC_train).any()}")

    print(f"\nTest data verification:")
    print(f"  YF has NaN: {np.isnan(YF_test).any()}")
    print(f"  YC has NaN: {np.isnan(YC_test).any()}")

    # Get training data summary
    print("\nTraining Data Summary:")
    summary = get_data_summary(X_train, T_train, YF_train, YC_train)
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )

    return (
        X_train,
        T_train,
        YF_train,
        YC_train,
        region_type_train,
        X_test,
        T_test,
        YF_test,
        YC_test,
        region_type_test,
    )


def save_data(
    X_train: np.ndarray,
    T_train: np.ndarray,
    YF_train: np.ndarray,
    YC_train: np.ndarray,
    region_type_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    YF_test: np.ndarray,
    YC_test: np.ndarray,
    region_type_test: np.ndarray,
    filepath: str = DATA_FILE,
):
    """Save training and test data to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "X_train": X_train,
        "T_train": T_train,
        "YF_train": YF_train,
        "YC_train": YC_train,
        "region_type_train": region_type_train,
        "X_test": X_test,
        "T_test": T_test,
        "YF_test": YF_test,
        "YC_test": YC_test,
        "region_type_test": region_type_test,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"\nData saved to: {filepath}")


def load_data(filepath: str = DATA_FILE) -> Dict[str, np.ndarray]:
    """Load training and test data from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from: {filepath}")
    return data


# ===================== STEP 1: Generate and Save Data =====================
def step1_generate_and_save_data():
    """Step 1: Generate training and test data and save to file."""
    print("=" * 60)
    print("STEP 1: Generate and Save Data")
    print("=" * 60)

    # Generate data
    (
        X_train,
        T_train,
        YF_train,
        YC_train,
        region_type_train,
        X_test,
        T_test,
        YF_test,
        YC_test,
        region_type_test,
    ) = generate_data(
        n_users_train=5000,
        n_users_test=2000,
        n_features=10,
        firm_outcome_base=0.0,
        firm_outcome_noise_std=1,
        user_outcome_base=0.0,
        user_outcome_noise_std=1,
        random_seed=RANDOM_SEED,
        firm_effect_strength=1,
        user_effect_strength=1,
    )

    # Save data
    save_data(
        X_train,
        T_train,
        YF_train,
        YC_train,
        region_type_train,
        X_test,
        T_test,
        YF_test,
        YC_test,
        region_type_test,
    )

    print("\nStep 1 complete!")


# ===================== STEP 2: Run DivergenceTree =====================
def step2_run_divergence_tree():
    """Step 2: Load data, run DivergenceTree, save results and tree."""
    print("=" * 60)
    print("STEP 2: Run DivergenceTree")
    print("=" * 60)

    # Load data
    data = load_data()
    X_train = data["X_train"]
    T_train = data["T_train"]
    YF_train = data["YF_train"]
    YC_train = data["YC_train"]
    X_test = data["X_test"]

    # Run DivergenceTree
    print("\nRunning DivergenceTree hyperparameter optimization...")
    fixed_params = {
        "lambda_": 1,
        "n_quantiles": 100,
        "random_state": RANDOM_SEED,
        "co_movement": "both",
        "eps_scale": 1e-8,
    }
    search_space = {
        "max_partitions": {"low": 4, "high": 15},
        "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
    }

    print(f"Fixed parameters: {fixed_params}")
    print(f"Search space: {search_space}")

    # Tune hyperparameters on training set
    best_params, best_loss = tune_divtree(
        X_train,
        T_train,
        YF_train,
        YC_train,
        fixed=fixed_params,
        search_space=search_space,
        n_trials=20,
        n_splits=2,
        random_state=RANDOM_SEED,
    )

    print(f"\nBest parameters found: {best_params}")
    print(f"Best CV loss: {best_loss:.6f}")

    # Train final tree with best parameters
    print("\nTraining DivergenceTree with best parameters...")
    tree = DivergenceTree(**best_params)
    tree.fit(X_train, T_train, YF_train, YC_train)

    # Print tree information
    leaf_effects = tree.leaf_effects()
    print(f"\nTree has {len(leaf_effects['leaves'])} leaves")

    # Predict on test set
    print("\nPredicting on test set...")
    region_type_pred_test = tree.predict_region_type(X_test)

    # Calculate performance metrics
    data_test = load_data()
    region_type_test = data_test["region_type_test"]
    accuracy_test = (region_type_pred_test == region_type_test).mean()

    print(f"\nTest set region type prediction accuracy: {accuracy_test:.4f}")

    # Save results
    results = {
        "tree": tree,
        "best_params": best_params,
        "best_loss": best_loss,
        "region_type_pred_test": region_type_pred_test,
        "accuracy_test": accuracy_test,
        "n_leaves": len(leaf_effects["leaves"]),
    }

    os.makedirs(os.path.dirname(DIVTREE_RESULTS_FILE), exist_ok=True)
    with open(DIVTREE_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"\nDivergenceTree results saved to: {DIVTREE_RESULTS_FILE}")
    print("Step 2 complete!")


# ===================== STEP 3: Run TwoStepDivergenceTree =====================
def step3_run_twostep_tree():
    """Step 3: Load data, run TwoStepDivergenceTree, save results and tree."""
    print("=" * 60)
    print("STEP 3: Run TwoStepDivergenceTree")
    print("=" * 60)

    # Load data
    data = load_data()
    X_train = data["X_train"]
    T_train = data["T_train"]
    YF_train = data["YF_train"]
    YC_train = data["YC_train"]
    X_test = data["X_test"]

    # Run TwoStepDivergenceTree
    print("\nRunning TwoStepDivergenceTree...")

    causal_forest_params = {
        "random_state": RANDOM_SEED,
        "n_jobs": -1,  # Use all available CPUs
    }
    causal_forest_tune_params = {
        "params": "auto",  # Use econml's default tuning grid
    }
    classification_tree_params = {"random_state": RANDOM_SEED}

    print("Causal forest params:", causal_forest_params)
    print("Causal forest tune params:", causal_forest_tune_params)

    # Train tree
    tree = TwoStepDivergenceTree(
        causal_forest_params=causal_forest_params,
        classification_tree_params=classification_tree_params,
        causal_forest_tune_params=causal_forest_tune_params,
    )
    tree.fit(
        X_train,
        T_train,
        YF_train,
        YC_train,
        auto_tune_classification_tree=True,
    )

    # Print tree information
    leaf_effects = tree.leaf_effects()
    print(f"\nTwoStepDivergenceTree has {len(leaf_effects['leaves'])} leaves")

    # Predict on test set
    print("\nPredicting on test set...")
    region_type_pred_test = tree.predict_region_type(X_test)

    # Calculate performance metrics
    data_test = load_data()
    region_type_test = data_test["region_type_test"]
    accuracy_test = (region_type_pred_test == region_type_test).mean()

    print(f"\nTest set region type prediction accuracy: {accuracy_test:.4f}")

    # Save results
    results = {
        "tree": tree,
        "region_type_pred_test": region_type_pred_test,
        "accuracy_test": accuracy_test,
        "n_leaves": len(leaf_effects["leaves"]),
    }

    os.makedirs(os.path.dirname(TWOSTEP_RESULTS_FILE), exist_ok=True)
    with open(TWOSTEP_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"\nTwoStepDivergenceTree results saved to: {TWOSTEP_RESULTS_FILE}")
    print("Step 3 complete!")


# ===================== STEP 4: Compare Results =====================
def step4_compare_results():
    """Step 4: Load results and trees, compare methods, visualize."""
    print("=" * 60)
    print("STEP 4: Compare Results")
    print("=" * 60)

    # Load data
    data = load_data()
    region_type_test = data["region_type_test"]

    # Load DivergenceTree results
    print("\nLoading DivergenceTree results...")
    with open(DIVTREE_RESULTS_FILE, "rb") as f:
        divtree_results = pickle.load(f)
    divtree_tree = divtree_results["tree"]
    divtree_region_pred_test = divtree_results["region_type_pred_test"]
    print(f"  DivergenceTree accuracy: {divtree_results['accuracy_test']:.4f}")

    # Load TwoStepDivergenceTree results
    print("\nLoading TwoStepDivergenceTree results...")
    with open(TWOSTEP_RESULTS_FILE, "rb") as f:
        twostep_results = pickle.load(f)
    twostep_tree = twostep_results["tree"]
    twostep_region_pred_test = twostep_results["region_type_pred_test"]
    print(f"  TwoStepDivergenceTree accuracy: {twostep_results['accuracy_test']:.4f}")

    # Compare methods
    print("\n" + "=" * 60)
    print("Method Comparison")
    print("=" * 60)

    comparison_results = compare_methods(
        divtree_region_pred=divtree_region_pred_test,
        twostep_region_pred=twostep_region_pred_test,
        region_type_true=region_type_test,
    )

    print_comparison(comparison_results)

    # Visualize trees side by side
    try:
        print("\nCreating tree visualizations...")
        os.makedirs(DATA_DIR, exist_ok=True)

        # Create subplot with both trees
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))

        # Plot DivergenceTree on left
        plot_divergence_tree(divtree_tree, ax=ax1, figsize=(15, 10), show_legend=False)
        ax1.set_title("DivergenceTree", fontsize=16, fontweight="bold", pad=20)

        # Plot TwoStepDivergenceTree on right
        from sklearn.tree import plot_tree

        plot_tree(
            twostep_tree.classification_tree_,
            ax=ax2,
            filled=True,
            fontsize=10,
            feature_names=[
                f"X{i}" for i in range(twostep_tree._fit_data["X"].shape[1])
            ],
            class_names=["Region 1", "Region 2", "Region 3", "Region 4"],
        )
        ax2.set_title("TwoStepDivergenceTree", fontsize=16, fontweight="bold", pad=20)

        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(DATA_DIR, "comparison_trees.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Tree comparison visualization saved as '{save_path}'")

        # Also save individual trees
        fig_div, ax_div = plot_divergence_tree(divtree_tree, figsize=(15, 10))
        fig_div.savefig(
            os.path.join(DATA_DIR, "divergence_tree.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_div)
        print(
            f"DivergenceTree visualization saved as '{os.path.join(DATA_DIR, 'divergence_tree.png')}'"
        )

        # Individual TwoStepDivergenceTree
        fig_twostep, ax_twostep = plt.subplots(figsize=(15, 10))
        plot_tree(
            twostep_tree.classification_tree_,
            ax=ax_twostep,
            filled=True,
            fontsize=10,
            feature_names=[
                f"X{i}" for i in range(twostep_tree._fit_data["X"].shape[1])
            ],
            class_names=["Region 1", "Region 2", "Region 3", "Region 4"],
        )
        ax_twostep.set_title("TwoStepDivergenceTree", fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig_twostep.savefig(
            os.path.join(DATA_DIR, "twostep_tree.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_twostep)
        print(
            f"TwoStepDivergenceTree visualization saved as '{os.path.join(DATA_DIR, 'twostep_tree.png')}'"
        )

    except Exception as e:
        print(f"Could not create visualization: {e}")
        import traceback

        traceback.print_exc()

    print("\nStep 4 complete!")


# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    # Comment/uncomment steps to run them independently
    # Each step can be run separately after the previous steps have been completed

    # Step 1: Generate and save data
    step1_generate_and_save_data()

    # Step 2: Run DivergenceTree and save results
    step2_run_divergence_tree()

    # Step 3: Run TwoStepDivergenceTree and save results
    step3_run_twostep_tree()

    # Step 4: Compare results and visualize
    step4_compare_results()
