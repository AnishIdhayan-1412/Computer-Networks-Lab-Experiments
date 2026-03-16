"""
train.py
--------
Training and evaluation entry point for the Bias Mitigated Hiring Pipeline.

Usage
-----
    python train.py

This script:
1. Generates a synthetic hiring dataset with demographic bias.
2. Trains a *baseline* logistic regression (no bias mitigation).
3. Trains the *fair* pipeline with bias mitigation.
4. Prints side-by-side fairness metrics for comparison.
"""

import warnings

import pandas as pd

from data_utils import generate_synthetic_hiring_data, load_and_split
from pipeline import BiasMitigatedHiringPipeline
from fairness_metrics import compute_all_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


SENSITIVE_ATTRS = ["gender", "race"]
TARGET_COL = "hired"
RANDOM_STATE = 42


def train_baseline(X_train, y_train, X_test, y_test, sensitive_attrs):
    """Train a naïve logistic regression that ignores fairness.

    The classifier does not receive the sensitive columns as features, but
    we keep them in X_test so we can compute fairness metrics afterwards.
    """
    feature_cols = [c for c in X_train.columns if c not in sensitive_attrs]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[feature_cols])
    X_te = scaler.transform(X_test[feature_cols])

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    sensitive_cols = {
        attr: np.asarray(X_test[attr]) for attr in sensitive_attrs
    }
    return compute_all_metrics(np.asarray(y_test), y_pred, sensitive_cols)


def train_fair(X_train, y_train, X_test, y_test, sensitive_attrs):
    """Train the bias-mitigated pipeline."""
    pipeline = BiasMitigatedHiringPipeline(
        sensitive_attributes=sensitive_attrs,
        target_column=TARGET_COL,
        correlation_threshold=0.3,
        fairness_threshold=0.1,
        model_type="logistic",
        random_state=RANDOM_STATE,
    )
    pipeline.fit(X_train, y_train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = pipeline.evaluate(X_test, y_test)
        for w in caught:
            print(f"  ⚠  {w.message}")

    return results


def print_comparison(baseline_df: pd.DataFrame, fair_df: pd.DataFrame) -> None:
    baseline_df = baseline_df.rename(columns={"value": "baseline"})
    fair_df = fair_df.rename(columns={"value": "fair_pipeline"})
    merged = pd.merge(baseline_df, fair_df, on=["attribute", "metric"])
    merged["improvement"] = merged["baseline"] - merged["fair_pipeline"]

    print("\n" + "=" * 72)
    print("  FAIRNESS COMPARISON: Baseline vs Bias-Mitigated Pipeline")
    print("=" * 72)
    print(merged.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 72 + "\n")


def main():
    print("Generating synthetic hiring dataset …")
    df = generate_synthetic_hiring_data(n_samples=3000, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = load_and_split(
        df, target_column=TARGET_COL, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n--- Training baseline (no mitigation) ---")
    baseline_results = train_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        SENSITIVE_ATTRS,
    )

    print("\n--- Training bias-mitigated pipeline ---")
    fair_results = train_fair(X_train, y_train, X_test, y_test, SENSITIVE_ATTRS)

    print_comparison(baseline_results, fair_results)


if __name__ == "__main__":
    main()
