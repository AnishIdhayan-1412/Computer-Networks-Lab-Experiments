"""
fairness_metrics.py
-------------------
Fairness evaluation utilities for the Bias Mitigated Hiring Pipeline.

All functions accept NumPy arrays or pandas Series for flexibility.
"""

import numpy as np
import pandas as pd


def demographic_parity_difference(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """Difference in positive prediction rates between sensitive groups.

    A value of 0 is ideal.  Values close to ±0 indicate parity.

    Parameters
    ----------
    y_pred    : predicted binary labels (0/1)
    sensitive : binary sensitive attribute (0=unprivileged, 1=privileged)

    Returns
    -------
    float  –  P(Ŷ=1 | S=1) − P(Ŷ=1 | S=0)
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    rate_privileged = y_pred[sensitive == 1].mean()
    rate_unprivileged = y_pred[sensitive == 0].mean()
    return float(rate_privileged - rate_unprivileged)


def disparate_impact_ratio(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """Ratio of positive prediction rate for unprivileged vs privileged group.

    The "80 % rule" (four-fifths rule) considers a ratio ≥ 0.8 acceptable.

    Returns
    -------
    float  –  P(Ŷ=1 | S=0) / P(Ŷ=1 | S=1)
              Returns 1.0 when both rates are 0 to avoid division by zero.
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    rate_privileged = y_pred[sensitive == 1].mean()
    rate_unprivileged = y_pred[sensitive == 0].mean()

    # Both zero → no disparity; privileged zero only → undefined, treat as neutral
    if rate_privileged == 0:
        return 1.0
    return float(rate_unprivileged / rate_privileged)


def equalized_odds_difference(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray
) -> float:
    """Maximum of the TPR difference and FPR difference across groups.

    A value of 0 is ideal.

    Returns
    -------
    float  –  max(|ΔTPR|, |ΔFPR|)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    def _rates(group_mask):
        yt = y_true[group_mask]
        yp = y_pred[group_mask]
        tpr = yp[yt == 1].mean() if yt.sum() > 0 else 0.0
        fpr = yp[yt == 0].mean() if (yt == 0).sum() > 0 else 0.0
        return tpr, fpr

    tpr1, fpr1 = _rates(sensitive == 1)
    tpr0, fpr0 = _rates(sensitive == 0)
    return float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_cols: dict,
) -> pd.DataFrame:
    """Compute all fairness metrics for each sensitive attribute.

    Parameters
    ----------
    y_true         : ground-truth labels
    y_pred         : predicted labels
    sensitive_cols : mapping of {attribute_name: array_of_values}

    Returns
    -------
    pd.DataFrame with columns [attribute, metric, value]
    """
    rows = []
    accuracy = float((np.asarray(y_pred) == np.asarray(y_true)).mean())

    for attr_name, attr_values in sensitive_cols.items():
        s = np.asarray(attr_values)
        rows.append(
            {
                "attribute": attr_name,
                "metric": "demographic_parity_difference",
                "value": demographic_parity_difference(y_pred, s),
            }
        )
        rows.append(
            {
                "attribute": attr_name,
                "metric": "disparate_impact_ratio",
                "value": disparate_impact_ratio(y_pred, s),
            }
        )
        rows.append(
            {
                "attribute": attr_name,
                "metric": "equalized_odds_difference",
                "value": equalized_odds_difference(y_true, y_pred, s),
            }
        )

    rows.append({"attribute": "overall", "metric": "accuracy", "value": accuracy})
    return pd.DataFrame(rows)
