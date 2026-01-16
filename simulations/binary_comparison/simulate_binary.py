"""
Simulation script using binary data generator for comparison.

This shows the complete workflow with binary categorical data:
1. Generate data using binary features (one-hot encoded categorical variables)
2. Optimize hyperparameters using cross-validation
3. Train divergence tree with best parameters
4. Train alternative two-step method
5. Compare both methods
6. Visualize the tree

The workflow is split into 4 independent steps that can be run separately:
Step 1: Generate and save data (with functional form)
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
from binary_data_generator import generate_binary_comparison_data, get_data_summary

# Add the src directory to the path to import divtree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna as tune_divtree
from divtree.viz import plot_divergence_tree
from twostepdivtree.tree import TwoStepDivergenceTree

# Import comparison helper from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "comparison"))
from comparison_helper import compare_methods, print_comparison

# ===================== GLOBAL PATHS =====================
# Use data folder in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "binary_comparison_data.pickle")
DIVTREE_RESULTS_FILE = os.path.join(DATA_DIR, "binary_divtree_results.pickle")
TWOSTEP_RESULTS_FILE = os.path.join(DATA_DIR, "binary_twostep_results.pickle")

# ===================== GLOBAL RANDOM SEED =====================
# Single random seed for all randomness in the simulation
RANDOM_SEED = 0
# ======================================================


def generate_data(
    n_users_train: int = 5000,
    n_users_test: int = 2000,
    k: int = 3,
    n_categories: list = [3, 4, 5],
    m_firm: int = 5,
    m_user: Optional[int] = None,
    similarity: float = 0.5,
    intensity: float = 2.0,
    effect_noise_std: float = 0.1,
    firm_outcome_noise_std: float = 1.0,
    user_outcome_noise_std: float = 1.0,
    positive_ratio: float = 0.5,
    random_seed: int = RANDOM_SEED,
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
    Dict[str, Any],
]:
    """
    Generate synthetic training and test data using binary data generator.

    Parameters
    ----------
    n_users_train : int, default=5000
        Number of training observations to generate.
    n_users_test : int, default=2000
        Number of test observations to generate.
    k : int, default=3
        Number of categorical variables.
    n_categories : list, default=[3, 4, 5]
        List of k integers, number of categories per variable.
    m_firm : int, default=5
        Number of combinations activating firm treatment effect.
    m_user : int, optional
        Number of combinations activating user treatment effect.
    similarity : float, default=0.5
        Proportion of combinations shared between firm and user.
    intensity : float, default=2.0
        Treatment effect intensity when activated.
    effect_noise_std : float, default=0.1
        Standard deviation of noise added to all treatment effects (mean=0).
    firm_outcome_noise_std : float, default=1.0
        Standard deviation of noise for firm outcome.
    user_outcome_noise_std : float, default=1.0
        Standard deviation of noise for user outcome.
    positive_ratio : float, default=0.5
        Minimum proportion of observations in activating combinations that should
        have positive treatment effects (after noise is added). Must be in [0, 1].
    random_seed : int, default=RANDOM_SEED
        Random seed for both training and test data.

    Returns
    -------
    X_train, T_train, YF_train, YC_train, region_type_train,
    X_test, T_test, YF_test, YC_test, region_type_test,
    functional_form
    """
    print("Generating training and test data using binary data generator...")

    # Generate all data at once to ensure same functional form (coefficients, combinations, etc.)
    n_users_total = n_users_train + n_users_test
    (
        X_all,
        T_all,
        YF_all,
        YC_all,
        tauF_all,
        tauC_all,
        region_type_all,
        functional_form,
    ) = generate_binary_comparison_data(
        n_users=n_users_total,
        k=k,
        n_categories=n_categories,
        m_firm=m_firm,
        m_user=m_user,
        similarity=similarity,
        intensity=intensity,
        effect_noise_std=effect_noise_std,
        firm_outcome_noise_std=firm_outcome_noise_std,
        user_outcome_noise_std=user_outcome_noise_std,
        positive_ratio=positive_ratio,
        random_seed=random_seed,
    )

    # Split into train and test sets using random permutation
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_users_total)
    train_indices = indices[:n_users_train]
    test_indices = indices[n_users_train:]

    # Split the data
    X_train = X_all[train_indices]
    T_train = T_all[train_indices]
    YF_train = YF_all[train_indices]
    YC_train = YC_all[train_indices]
    region_type_train = region_type_all[train_indices]

    X_test = X_all[test_indices]
    T_test = T_all[test_indices]
    YF_test = YF_all[test_indices]
    YC_test = YC_all[test_indices]
    region_type_test = region_type_all[test_indices]

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
        functional_form,
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
    functional_form: Dict[str, Any],
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
        "functional_form": functional_form,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"\nData saved to: {filepath}")


def load_data(filepath: str = DATA_FILE) -> Dict[str, Any]:
    """Load training and test data from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from: {filepath}")
    return data


# ===================== STEP 1: Generate and Save Data =====================
def step1_generate_and_save_data():
    """Step 1: Generate training and test data and save to file."""
    print("=" * 60)
    print("STEP 1: Generate and Save Data (Binary Generator)")
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
        functional_form,
    ) = generate_data(
        n_users_train=50000,
        n_users_test=25000,
        k=2,
        n_categories=[5]*2,
        m_firm=4,
        m_user=4,
        similarity=0.5,
        intensity=1,
        effect_noise_std=1,
        firm_outcome_noise_std=1,
        user_outcome_noise_std=1,
        positive_ratio=0.2,
        random_seed=RANDOM_SEED,
    )

    # Print functional form information with combinations
    print("\n" + "=" * 60)
    print("Functional Form Information")
    print("=" * 60)
    print(f"  Number of categorical variables (k): {functional_form['k']}")
    print(f"  Categories per variable: {functional_form['n_categories']}")
    print(f"  Firm activating combinations: {functional_form['m_firm']}")
    print(f"  User activating combinations: {functional_form['m_user']}")
    print(f"  Similarity: {functional_form['similarity']:.2f}")
    print(f"  Intensity: {functional_form['intensity']:.2f}")
    print(
        f"  Shared combinations: {len(set(functional_form['firm_combinations']) & set(functional_form['user_combinations']))}"
    )

    print(f"\n  Firm Treatment Effect Combinations:")
    for i, (combo, sign) in enumerate(
        zip(functional_form["firm_combinations"], functional_form["firm_signs"])
    ):
        sign_str = "+" if sign > 0 else "-"
        print(
            f"    {i+1}. {combo} -> sign={sign_str}, effect={sign * functional_form['intensity']:.2f}"
        )

    print(f"\n  User Treatment Effect Combinations:")
    for i, (combo, sign) in enumerate(
        zip(functional_form["user_combinations"], functional_form["user_signs"])
    ):
        sign_str = "+" if sign > 0 else "-"
        print(
            f"    {i+1}. {combo} -> sign={sign_str}, effect={sign * functional_form['intensity']:.2f}"
        )

    # Check shared combinations
    firm_set = set(functional_form["firm_combinations"])
    user_set = set(functional_form["user_combinations"])
    shared = firm_set & user_set
    if len(shared) > 0:
        print(f"\n  Shared Combinations:")
        for combo in shared:
            firm_idx = functional_form["firm_combinations"].index(combo)
            user_idx = functional_form["user_combinations"].index(combo)
            firm_sign = functional_form["firm_signs"][firm_idx]
            user_sign = functional_form["user_signs"][user_idx]
            print(f"    {combo}: Firm sign={firm_sign}, User sign={user_sign}")

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
        functional_form,
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
        "n_quantiles": 3,
        "random_state": RANDOM_SEED,
        "co_movement": "both",
        "eps_scale": 1e-8,
    }
    search_space = {
        "max_partitions": {"low": 4, "high": 50},
        "min_improvement_ratio": {"low": 0.001, "high": 0.1, "log": True},
    }

    print(f"Fixed parameters: {fixed_params}")
    print(f"Search space: {search_space}")
    print(f"Number of trials: 20")
    print(f"Number of CV folds: 2")
    print("\nStarting hyperparameter optimization (this may take a while)...")

    # Tune hyperparameters on training set
    best_params, best_loss = tune_divtree(
        X_train,
        T_train,
        YF_train,
        YC_train,
        fixed=fixed_params,
        search_space=search_space,
        n_trials=30,
        n_splits=2,
        random_state=RANDOM_SEED,
    )

    print(f"\nHyperparameter optimization complete!")
    print(f"Best parameters found: {best_params}")
    print(f"Best CV loss: {best_loss:.6f}")

    # Train final tree with best parameters
    print("\nTraining DivergenceTree with best parameters...")
    tree = DivergenceTree(**best_params)
    tree.fit(X_train, T_train, YF_train, YC_train)

    # Print tree information
    leaf_effects = tree.leaf_effects()
    print(f"\nTree has {len(leaf_effects['leaves'])} leaves")

    # Check training accuracy
    print("\nChecking training set accuracy...")
    region_type_pred_train = tree.predict_region_type(X_train)
    region_type_train = data["region_type_train"]
    accuracy_train = (region_type_pred_train == region_type_train).mean()
    print(f"Training set region type prediction accuracy: {accuracy_train:.4f}")

    # Plot and save the tree
    try:
        print("\nPlotting DivergenceTree...")
        os.makedirs(os.path.dirname(DIVTREE_RESULTS_FILE), exist_ok=True)
        fig, ax = plot_divergence_tree(tree, figsize=(15, 10))
        tree_plot_path = os.path.join(DATA_DIR, "binary_divergence_tree_step2.png")
        fig.savefig(tree_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Tree visualization saved as '{tree_plot_path}'")
    except Exception as e:
        print(f"Could not create tree visualization: {e}")
        import traceback

        traceback.print_exc()

    # Predict on test set
    print("\nPredicting on test set...")
    region_type_pred_test = tree.predict_region_type(X_test)

    # Calculate performance metrics
    data_test = load_data()
    region_type_test = data_test["region_type_test"]
    accuracy_test = (region_type_pred_test == region_type_test).mean()

    print(f"\nTest set region type prediction accuracy: {accuracy_test:.4f}")

    # Diagnostic: Check region type distributions
    print("\nRegion type distribution comparison:")
    print("  Training set (true):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_train == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_train):.2f}%)")
    print("  Test set (true):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_test == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_test):.2f}%)")
    print("  Test set (predicted):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_pred_test == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_pred_test):.2f}%)")

    # Diagnostic: Confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(region_type_test, region_type_pred_test, labels=[1, 2, 3, 4])
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("      ", " ".join([f"Pred {i}" for i in [1, 2, 3, 4]]))
    for i, row in enumerate(cm):
        print(f"True {i+1}:", " ".join([f"{val:6d}" for val in row]))

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

    # Check training accuracy
    print("\nChecking training set accuracy...")
    region_type_pred_train = tree.predict_region_type(X_train)
    region_type_train = data["region_type_train"]
    accuracy_train = (region_type_pred_train == region_type_train).mean()
    print(f"Training set region type prediction accuracy: {accuracy_train:.4f}")

    # Get estimated treatment effects for training set
    print("\nExtracting estimated treatment effects for training set...")
    tauF_train_est, tauC_train_est = tree.get_training_treatment_effects()

    # Predict on test set
    print("\nPredicting on test set...")
    region_type_pred_test = tree.predict_region_type(X_test)

    # Get estimated treatment effects for test set
    print("Extracting estimated treatment effects for test set...")
    tauF_test_est, tauC_test_est = tree.predict_treatment_effects(X_test)

    # Calculate performance metrics
    data_test = load_data()
    region_type_test = data_test["region_type_test"]
    accuracy_test = (region_type_pred_test == region_type_test).mean()

    print(f"\nTest set region type prediction accuracy: {accuracy_test:.4f}")

    # Diagnostic: Check region type distributions
    print("\nRegion type distribution comparison:")
    print("  Training set (true):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_train == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_train):.2f}%)")
    print("  Test set (true):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_test == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_test):.2f}%)")
    print("  Test set (predicted):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_pred_test == rt).sum()
        print(f"    Region {rt}: {count} ({100*count/len(region_type_pred_test):.2f}%)")

    # Diagnostic: Confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(region_type_test, region_type_pred_test, labels=[1, 2, 3, 4])
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("      ", " ".join([f"Pred {i}" for i in [1, 2, 3, 4]]))
    for i, row in enumerate(cm):
        print(f"True {i+1}:", " ".join([f"{val:6d}" for val in row]))

    # Diagnostic: Treatment effect statistics
    print("\nEstimated Treatment Effects Statistics:")
    print("  Training set:")
    print(
        f"    tauF: mean={tauF_train_est.mean():.4f}, std={tauF_train_est.std():.4f}, min={tauF_train_est.min():.4f}, max={tauF_train_est.max():.4f}"
    )
    print(
        f"    tauC: mean={tauC_train_est.mean():.4f}, std={tauC_train_est.std():.4f}, min={tauC_train_est.min():.4f}, max={tauC_train_est.max():.4f}"
    )
    print("  Test set:")
    print(
        f"    tauF: mean={tauF_test_est.mean():.4f}, std={tauF_test_est.std():.4f}, min={tauF_test_est.min():.4f}, max={tauF_test_est.max():.4f}"
    )
    print(
        f"    tauC: mean={tauC_test_est.mean():.4f}, std={tauC_test_est.std():.4f}, min={tauC_test_est.min():.4f}, max={tauC_test_est.max():.4f}"
    )

    # Save results
    results = {
        "tree": tree,
        "region_type_pred_test": region_type_pred_test,
        "accuracy_test": accuracy_test,
        "n_leaves": len(leaf_effects["leaves"]),
        "tauF_train_est": tauF_train_est,
        "tauC_train_est": tauC_train_est,
        "tauF_test_est": tauF_test_est,
        "tauC_test_est": tauC_test_est,
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
    functional_form = data.get("functional_form", None)

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
        save_path = os.path.join(DATA_DIR, "binary_comparison_trees.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Tree comparison visualization saved as '{save_path}'")

        # Also save individual trees
        fig_div, ax_div = plot_divergence_tree(divtree_tree, figsize=(15, 10))
        fig_div.savefig(
            os.path.join(DATA_DIR, "binary_divergence_tree.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_div)
        print(
            f"DivergenceTree visualization saved as '{os.path.join(DATA_DIR, 'binary_divergence_tree.png')}'"
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
            os.path.join(DATA_DIR, "binary_twostep_tree.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_twostep)
        print(
            f"TwoStepDivergenceTree visualization saved as '{os.path.join(DATA_DIR, 'binary_twostep_tree.png')}'"
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
