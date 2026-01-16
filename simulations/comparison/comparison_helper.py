"""
Helper functions for comparing divergence tree methods.

This module provides functions to compare the performance of different
divergence tree algorithms, including the original DivergenceTree and the
TwoStepDivergenceTree alternative method.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix


def compare_methods(
    divtree_region_pred: np.ndarray,
    twostep_region_pred: np.ndarray,
    region_type_true: np.ndarray,
    divtree_name: str = "DivergenceTree",
    twostep_name: str = "TwoStepDivergenceTree",
) -> Dict[str, Any]:
    """
    Compare performance of two divergence tree methods.

    Parameters
    ----------
    divtree_region_pred : np.ndarray of shape (n_samples,)
        Predicted region types from DivergenceTree (1-4).
    twostep_region_pred : np.ndarray of shape (n_samples,)
        Predicted region types from TwoStepDivergenceTree (1-4).
    region_type_true : np.ndarray of shape (n_samples,)
        True region type labels (1-4).
    divtree_name : str, default="DivergenceTree"
        Name for the first method (for display).
    twostep_name : str, default="TwoStepDivergenceTree"
        Name for the second method (for display).

    Returns
    -------
    dict
        Dictionary containing comparison metrics:
        - 'divtree_accuracy': Overall accuracy for DivergenceTree
        - 'twostep_accuracy': Overall accuracy for TwoStepDivergenceTree
        - 'divtree_confusion_matrix': Confusion matrix for DivergenceTree
        - 'twostep_confusion_matrix': Confusion matrix for TwoStepDivergenceTree
        - 'divtree_per_region_accuracy': Per-region accuracy for DivergenceTree
        - 'twostep_per_region_accuracy': Per-region accuracy for TwoStepDivergenceTree
        - 'divtree_region_distribution': Predicted region distribution for DivergenceTree
        - 'twostep_region_distribution': Predicted region distribution for TwoStepDivergenceTree
    """
    # Overall accuracy
    divtree_accuracy = accuracy_score(region_type_true, divtree_region_pred)
    twostep_accuracy = accuracy_score(region_type_true, twostep_region_pred)

    # Confusion matrices
    divtree_cm = confusion_matrix(
        region_type_true, divtree_region_pred, labels=[1, 2, 3, 4]
    )
    twostep_cm = confusion_matrix(
        region_type_true, twostep_region_pred, labels=[1, 2, 3, 4]
    )

    # Per-region accuracy
    divtree_per_region = {}
    twostep_per_region = {}

    for rt in [1, 2, 3, 4]:
        mask = region_type_true == rt
        if mask.sum() > 0:
            divtree_correct = (divtree_region_pred[mask] == rt).sum()
            divtree_per_region[rt] = divtree_correct / mask.sum()

            twostep_correct = (twostep_region_pred[mask] == rt).sum()
            twostep_per_region[rt] = twostep_correct / mask.sum()
        else:
            divtree_per_region[rt] = np.nan
            twostep_per_region[rt] = np.nan

    # Region distributions
    divtree_dist = {
        rt: (divtree_region_pred == rt).sum() / len(divtree_region_pred)
        for rt in [1, 2, 3, 4]
    }
    twostep_dist = {
        rt: (twostep_region_pred == rt).sum() / len(twostep_region_pred)
        for rt in [1, 2, 3, 4]
    }

    return {
        "divtree_accuracy": divtree_accuracy,
        "twostep_accuracy": twostep_accuracy,
        "divtree_confusion_matrix": divtree_cm,
        "twostep_confusion_matrix": twostep_cm,
        "divtree_per_region_accuracy": divtree_per_region,
        "twostep_per_region_accuracy": twostep_per_region,
        "divtree_region_distribution": divtree_dist,
        "twostep_region_distribution": twostep_dist,
    }


def print_comparison(
    comparison_results: Dict[str, Any],
    divtree_name: str = "DivergenceTree",
    twostep_name: str = "TwoStepDivergenceTree",
):
    """
    Print formatted comparison results.

    Parameters
    ----------
    comparison_results : dict
        Results dictionary from compare_methods().
    divtree_name : str, default="DivergenceTree"
        Name for the first method (for display).
    twostep_name : str, default="TwoStepDivergenceTree"
        Name for the second method (for display).
    """
    print("\n" + "=" * 70)
    print("Method Comparison Results")
    print("=" * 70)

    # Overall accuracy
    print(f"\nOverall Accuracy:")
    print(f"  {divtree_name}: {comparison_results['divtree_accuracy']:.4f}")
    print(f"  {twostep_name}: {comparison_results['twostep_accuracy']:.4f}")

    # Confusion matrices
    print(f"\n{divtree_name} Confusion Matrix (Predicted vs True):")
    print("        ", end="")
    for rt_true in [1, 2, 3, 4]:
        print(f"True {rt_true:>6}", end="")
    print()

    for i, rt_pred in enumerate([1, 2, 3, 4]):
        print(f"Pred {rt_pred}:", end="")
        for rt_true in [1, 2, 3, 4]:
            count = comparison_results["divtree_confusion_matrix"][i, rt_true - 1]
            print(f"{count:>8}", end="")
        print()

    print(f"\n{twostep_name} Confusion Matrix (Predicted vs True):")
    print("        ", end="")
    for rt_true in [1, 2, 3, 4]:
        print(f"True {rt_true:>6}", end="")
    print()

    for i, rt_pred in enumerate([1, 2, 3, 4]):
        print(f"Pred {rt_pred}:", end="")
        for rt_true in [1, 2, 3, 4]:
            count = comparison_results["twostep_confusion_matrix"][i, rt_true - 1]
            print(f"{count:>8}", end="")
        print()

    # Per-region accuracy
    print(f"\nPer-Region Accuracy:")
    print(f"  {divtree_name}:")
    for rt in [1, 2, 3, 4]:
        acc = comparison_results["divtree_per_region_accuracy"][rt]
        if not np.isnan(acc):
            print(f"    Region {rt}: {acc:.4f}")
        else:
            print(f"    Region {rt}: N/A (no true observations)")

    print(f"  {twostep_name}:")
    for rt in [1, 2, 3, 4]:
        acc = comparison_results["twostep_per_region_accuracy"][rt]
        if not np.isnan(acc):
            print(f"    Region {rt}: {acc:.4f}")
        else:
            print(f"    Region {rt}: N/A (no true observations)")

    # Region distributions
    print(f"\nPredicted Region Type Distribution:")
    print(f"  {divtree_name}:")
    for rt in [1, 2, 3, 4]:
        prop = comparison_results["divtree_region_distribution"][rt]
        print(f"    Region {rt}: {prop*100:.2f}%")

    print(f"  {twostep_name}:")
    for rt in [1, 2, 3, 4]:
        prop = comparison_results["twostep_region_distribution"][rt]
        print(f"    Region {rt}: {prop*100:.2f}%")

    print("\n" + "=" * 70)
