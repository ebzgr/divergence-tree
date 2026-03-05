"""
Evaluation metrics for divergence tree comparison.

Implements comprehensive evaluation metrics including accuracy, FNR, F1, MCC, etc.
"""

import numpy as np
from typing import Dict
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

    return metrics

