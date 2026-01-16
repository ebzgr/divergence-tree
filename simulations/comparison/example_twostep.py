"""
Example: How to use the two-step divergence tree method.

This shows the complete workflow for testing the two-step method:
1. Generate data with continuous outcomes
2. Get data summary
3. Optimize classification tree hyperparameters using Optuna
4. Train two-step divergence tree with best parameters
5. Print results and tree information
"""

import numpy as np
import os
import sys
from typing import Dict, Any, Optional

from data_generator import generate_comparison_data, get_data_summary

# Add the src directory to the path to import twostepdivtree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from twostepdivtree.tree import TwoStepDivergenceTree


def run_example():
    """Run the complete example with two-step divergence tree analysis."""

    # 1. Generate training and test data
    print("=" * 60)
    print("Step 1: Generating training and test data...")
    print("=" * 60)

    # Training data
    X_train, T_train, YF_train, YC_train, tauF_train, tauC_train, region_type_train = (
        generate_comparison_data(
            n_users=10000,
            n_features=10,
            firm_outcome_base=0.0,
            firm_outcome_noise_std=1.0,
            user_outcome_base=0.0,
            user_outcome_noise_std=1.0,
            random_seed=0,
            firm_effect_strength=1.0,
            user_effect_strength=1.0,
        )
    )

    # Test data (different seed)
    X_test, T_test, YF_test, YC_test, tauF_test, tauC_test, region_type_test = (
        generate_comparison_data(
            n_users=5000,
            n_features=10,
            firm_outcome_base=0.0,
            firm_outcome_noise_std=1.0,
            user_outcome_base=0.0,
            user_outcome_noise_std=1.0,
            random_seed=42,
            firm_effect_strength=1.0,
            user_effect_strength=1.0,
        )
    )

    print(f"Training data:")
    print(f"  X shape: {X_train.shape}")
    print(f"  T shape: {T_train.shape}")
    print(f"  YF shape: {YF_train.shape}, has NaN: {np.isnan(YF_train).any()}")
    print(f"  YC shape: {YC_train.shape}, has NaN: {np.isnan(YC_train).any()}")

    print(f"\nTest data:")
    print(f"  X shape: {X_test.shape}")
    print(f"  T shape: {T_test.shape}")
    print(f"  YF shape: {YF_test.shape}, has NaN: {np.isnan(YF_test).any()}")
    print(f"  YC shape: {YC_test.shape}, has NaN: {np.isnan(YC_test).any()}")

    # 2. Get training data summary
    print("\n" + "=" * 60)
    print("Step 2: Training Data Summary")
    print("=" * 60)
    summary = get_data_summary(X_train, T_train, YF_train, YC_train)
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )

    # Print training region type distribution
    print("\nTraining region type distribution (ground truth):")
    for rt in [1, 2, 3, 4]:
        count = (region_type_train == rt).sum()
        print(
            f"  Region {rt}: {count} observations ({100*count/len(region_type_train):.2f}%)"
        )

    # 3. Train two-step tree with automatic tuning
    print("\n" + "=" * 60)
    print("Step 3: Training two-step tree with automatic tuning...")
    print("=" * 60)
    print("Note: Causal forests are tuned automatically during tree building.")
    print("Classification tree will also be tuned automatically.")

    # Define fixed parameters for causal forest
    causal_forest_params = {
        "random_state": 42,
        "n_jobs": -1,  # Use all available CPUs for parallelization
    }

    causal_forest_tune_params = {
        "params": "auto",  # Use econml's default tuning grid
    }
    classification_tree_params = {
        "random_state": 42,
    }

    print(f"Causal forest params: {causal_forest_params}")
    print(f"Causal forest tune params: {causal_forest_tune_params}")

    # Create and fit tree with automatic tuning
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

    # 4. Print tree information
    print("\n" + "=" * 60)
    print("Step 4: Tree Information")
    print("=" * 60)
    leaf_effects = tree.leaf_effects()
    print(f"TwoStepDivergenceTree has {len(leaf_effects['leaves'])} leaves")
    print("\nLeaf effects summary:")
    for i, leaf in enumerate(leaf_effects["leaves"][:10]):  # Show first 10 leaves
        print(
            f"  Leaf {leaf['leaf_id']}: region_type={leaf['region_type']}, "
            f"tauF={leaf['tauF']:.4f}, tauC={leaf['tauC']:.4f}, n={leaf['n']}"
        )
    if len(leaf_effects["leaves"]) > 10:
        print(f"  ... and {len(leaf_effects['leaves']) - 10} more leaves")

    # 5. Evaluate on test set
    print("\n" + "=" * 60)
    print("Step 5: Test Set Evaluation - Region Type Prediction")
    print("=" * 60)
    region_type_pred_test = tree.predict_region_type(X_test)

    # Calculate accuracy on test set
    accuracy_test = (region_type_pred_test == region_type_test).mean()
    print(f"Test set region type prediction accuracy: {accuracy_test:.4f}")

    # Confusion matrix on test set
    print("\nTest set confusion matrix (rows=predicted, cols=true):")
    from sklearn.metrics import confusion_matrix

    cm_test = confusion_matrix(
        region_type_test, region_type_pred_test, labels=[1, 2, 3, 4]
    )
    print("      ", end="")
    for rt in [1, 2, 3, 4]:
        print(f"True {rt:2d}", end="  ")
    print()
    for i, rt in enumerate([1, 2, 3, 4]):
        print(f"Pred {rt:2d} ", end="")
        for j in range(4):
            print(f"{cm_test[i, j]:6d}", end="  ")
        print()

    # Per-region accuracy on test set
    print("\nTest set per-region accuracy:")
    for rt in [1, 2, 3, 4]:
        mask = region_type_test == rt
        if mask.sum() > 0:
            acc = (region_type_pred_test[mask] == region_type_test[mask]).mean()
            print(f"  Region {rt}: {acc:.4f} ({mask.sum()} observations)")

    # 6. Treatment effect predictions on test set
    print("\n" + "=" * 60)
    print("Step 6: Test Set Evaluation - Treatment Effect Predictions")
    print("=" * 60)
    tauF_pred_test, tauC_pred_test = tree.predict_treatment_effects(X_test)

    # Compare with ground truth on test set
    tauF_mae_test = np.nanmean(np.abs(tauF_pred_test - tauF_test))
    tauC_mae_test = np.nanmean(np.abs(tauC_pred_test - tauC_test))
    tauF_rmse_test = np.sqrt(np.nanmean((tauF_pred_test - tauF_test) ** 2))
    tauC_rmse_test = np.sqrt(np.nanmean((tauC_pred_test - tauC_test) ** 2))

    print(f"Test set firm outcome treatment effect:")
    print(f"  MAE: {tauF_mae_test:.4f}")
    print(f"  RMSE: {tauF_rmse_test:.4f}")
    print(f"Test set consumer outcome treatment effect:")
    print(f"  MAE: {tauC_mae_test:.4f}")
    print(f"  RMSE: {tauC_rmse_test:.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_example()
