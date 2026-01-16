"""
Evaluation metrics for divergence tree comparison.

Implements comprehensive evaluation metrics including accuracy, FNR, F1, MCC, RIG, etc.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, method_name: str = ""
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a given method.

    Parameters
    ----------
    y_true : np.ndarray
        True region type labels (1-4).
    y_pred : np.ndarray
        Predicted region type labels (1-4).
    method_name : str, default=""
        Prefix for metric names (e.g., "divtree" or "twostep").

    Returns
    -------
    dict
        Dictionary containing all computed metrics with method prefix.
    """
    prefix = f"{method_name}_" if method_name else ""

    metrics = {}

    # Overall accuracy
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)

    # Per-region accuracy
    for region in [1, 2, 3, 4]:
        mask = y_true == region
        if mask.sum() > 0:
            metrics[f"{prefix}acc_region_{region}"] = (
                (y_pred[mask] == region).sum() / mask.sum()
            )
        else:
            metrics[f"{prefix}acc_region_{region}"] = np.nan

    # False Negative Rate for region 2
    region_2_mask = y_true == 2
    if region_2_mask.sum() > 0:
        fnr_region_2 = (y_pred[region_2_mask] != 2).sum() / region_2_mask.sum()
        metrics[f"{prefix}fnr_region_2"] = fnr_region_2
    else:
        metrics[f"{prefix}fnr_region_2"] = np.nan

    # Precision and Recall for region 2
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    region_idx = 1  # Region 2 is at index 1 (regions are 1,2,3,4)
    tp = cm[region_idx, region_idx]
    fp = cm[:, region_idx].sum() - tp
    fn = cm[region_idx, :].sum() - tp
    
    if (tp + fp) > 0:
        metrics[f"{prefix}precision_region_2"] = tp / (tp + fp)
    else:
        metrics[f"{prefix}precision_region_2"] = np.nan
    
    if (tp + fn) > 0:
        metrics[f"{prefix}recall_region_2"] = tp / (tp + fn)
    else:
        metrics[f"{prefix}recall_region_2"] = np.nan

    # F1 score per region
    for region in [1, 2, 3, 4]:
        try:
            f1 = f1_score(
                y_true == region, y_pred == region, zero_division=0
            )
            metrics[f"{prefix}f1_region_{region}"] = f1
        except:
            metrics[f"{prefix}f1_region_{region}"] = np.nan

    # Balanced accuracy
    metrics[f"{prefix}balanced_accuracy"] = balanced_accuracy_score(
        y_true, y_pred
    )

    # Matthews Correlation Coefficient
    try:
        metrics[f"{prefix}mcc"] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics[f"{prefix}mcc"] = np.nan

    # RIG (Relative Information Gain)
    metrics[f"{prefix}rig"] = compute_rig(y_true, y_pred)

    return metrics


def compute_rig(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Relative Information Gain (RIG) for multiclass classification.

    RIG = (H_baseline - H_model) / H_baseline
    where H is entropy: H = -Σ p_i * log(p_i)

    Baseline: uniform distribution (1/4 for each class)
    Model: predicted class distribution

    Parameters
    ----------
    y_true : np.ndarray
        True region type labels (1-4).
    y_pred : np.ndarray
        Predicted region type labels (1-4).

    Returns
    -------
    float
        RIG value. Returns 0 if baseline entropy is 0.
    """
    n = len(y_pred)
    if n == 0:
        return np.nan

    # Baseline: uniform distribution (entropy = log(4))
    n_classes = 4
    h_baseline = np.log(n_classes)

    # Model: predicted class distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    probs_pred = counts_pred / n

    # Compute entropy of predicted distribution
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    h_model = -np.sum(probs_pred * np.log(probs_pred + eps))

    # RIG = (H_baseline - H_model) / H_baseline
    if h_baseline == 0:
        return 0.0

    rig = (h_baseline - h_model) / h_baseline
    return rig


def compute_confusion_matrix_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute confusion matrix and related metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True region type labels (1-4).
    y_pred : np.ndarray
        Predicted region type labels (1-4).

    Returns
    -------
    dict
        Dictionary containing confusion matrix and per-class metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

    # Per-class precision, recall, F1
    per_class_metrics = {}
    for i, region in enumerate([1, 2, 3, 4]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[f"region_{region}_precision"] = precision
        per_class_metrics[f"region_{region}_recall"] = recall
        per_class_metrics[f"region_{region}_f1"] = f1

    return {
        "confusion_matrix": cm,
        **per_class_metrics,
    }

