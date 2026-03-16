"""
pipeline.py
-----------
End-to-end Bias Mitigated Hiring Pipeline.

Design
------
1. Pre-processing  – drop sensitive attributes and correlated proxies;
                     compute per-sample fairness reweighing weights.
2. In-processing   – train a scikit-learn classifier using the weights.
3. Evaluation      – report standard accuracy + fairness metrics and emit
                     warnings when any metric exceeds the configured threshold.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fairness_metrics import compute_all_metrics


class BiasMitigatedHiringPipeline:
    """A scikit-learn–style pipeline that trains a fair binary classifier.

    Parameters
    ----------
    sensitive_attributes  : list of column names treated as protected characteristics
    target_column         : name of the binary outcome column (must NOT be in X)
    correlation_threshold : drop any non-sensitive feature whose absolute Pearson
                            correlation with any sensitive attribute exceeds this value
    fairness_threshold    : emit a warning when |fairness metric| exceeds this value
    model_type            : base classifier – "logistic" or "random_forest"
    random_state          : seed for reproducibility
    """

    SUPPORTED_MODELS = ("logistic", "random_forest")

    def __init__(
        self,
        sensitive_attributes: list = None,
        target_column: str = "hired",
        correlation_threshold: float = 0.5,
        fairness_threshold: float = 0.1,
        model_type: str = "logistic",
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_MODELS}, got '{model_type}'"
            )

        self.sensitive_attributes = sensitive_attributes or []
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.fairness_threshold = fairness_threshold
        self.model_type = model_type
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = self._build_model()
        self._feature_cols = []
        self._dropped_proxy_cols = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BiasMitigatedHiringPipeline":
        """Fit the pipeline on training data.

        Parameters
        ----------
        X : feature DataFrame (must include sensitive attribute columns)
        y : binary outcome Series
        """
        X = X.copy()
        y = np.asarray(y)

        # 1. Identify and drop proxy features
        proxy_cols = self._find_proxy_columns(X)
        self._dropped_proxy_cols = proxy_cols
        if proxy_cols:
            print(
                f"[BiasMitigatedHiringPipeline] Dropping proxy columns correlated with "
                f"sensitive attributes: {proxy_cols}"
            )

        # 2. Remove sensitive attributes and proxies from the feature set
        cols_to_drop = [
            c
            for c in self.sensitive_attributes + proxy_cols
            if c in X.columns
        ]
        X_fair = X.drop(columns=cols_to_drop)
        self._feature_cols = list(X_fair.columns)

        # 3. Compute reweighing sample weights
        sample_weights = self._compute_reweighing_weights(X, y)

        # 4. Scale features and train
        X_scaled = self._scaler.fit_transform(X_fair)
        self._model.fit(X_scaled, y, sample_weight=sample_weights)

        print("[BiasMitigatedHiringPipeline] Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions for *X*."""
        X_fair = X[self._feature_cols].copy()
        X_scaled = self._scaler.transform(X_fair)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates for *X* (shape: n_samples × 2)."""
        X_fair = X[self._feature_cols].copy()
        X_scaled = self._scaler.transform(X_fair)
        return self._model.predict_proba(X_scaled)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Predict on *X* and compute accuracy + fairness metrics.

        Returns a DataFrame with columns [attribute, metric, value].
        Emits warnings for metrics that exceed ``fairness_threshold``.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)

        sensitive_cols = {
            attr: np.asarray(X[attr])
            for attr in self.sensitive_attributes
            if attr in X.columns
        }

        results = compute_all_metrics(y_true, y_pred, sensitive_cols)
        self._warn_if_unfair(results)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
            )
        return RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
        )

    def _find_proxy_columns(self, X: pd.DataFrame) -> list:
        """Return columns (excluding sensitive ones) whose absolute Pearson
        correlation with any sensitive attribute exceeds ``correlation_threshold``."""
        proxy_cols = []
        non_sensitive = [
            c for c in X.columns if c not in self.sensitive_attributes
        ]
        for col in non_sensitive:
            for s_attr in self.sensitive_attributes:
                if s_attr not in X.columns:
                    continue
                try:
                    corr = X[col].corr(X[s_attr])
                    if abs(corr) > self.correlation_threshold:
                        proxy_cols.append(col)
                        break
                except Exception:
                    pass
        return proxy_cols

    def _compute_reweighing_weights(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> np.ndarray:
        """Compute per-sample fairness weights using the reweighing algorithm.

        Weights correct for the joint distribution of sensitive attributes and
        labels so that a classifier trained on them cannot exploit demographic
        imbalances in the training data.
        """
        n = len(y)
        weights = np.ones(n, dtype=float)

        if not self.sensitive_attributes:
            return weights

        # Build a group key from all sensitive attributes present in X
        group_keys = pd.Series(
            [
                "_".join(
                    str(X.iloc[i][s])
                    for s in self.sensitive_attributes
                    if s in X.columns
                )
                for i in range(n)
            ]
        )

        y_series = pd.Series(y)
        for group_val in group_keys.unique():
            group_mask = group_keys == group_val
            for label_val in np.unique(y):
                label_mask = y_series == label_val
                combo_mask = group_mask & label_mask

                p_group = group_mask.sum() / n
                p_label = label_mask.sum() / n
                p_combo = combo_mask.sum() / n

                if p_combo > 0:
                    ideal_weight = (p_group * p_label) / p_combo
                    weights[combo_mask.values] = ideal_weight

        return weights

    def _warn_if_unfair(self, results: pd.DataFrame) -> None:
        """Emit warnings for fairness metrics that exceed the threshold."""
        fairness_metrics = [
            "demographic_parity_difference",
            "equalized_odds_difference",
        ]
        for _, row in results.iterrows():
            if row["metric"] in fairness_metrics:
                if abs(row["value"]) > self.fairness_threshold:
                    warnings.warn(
                        f"[FairnessWarning] {row['metric']} for attribute "
                        f"'{row['attribute']}' = {row['value']:.4f} exceeds "
                        f"threshold {self.fairness_threshold}.",
                        stacklevel=2,
                    )
            if row["metric"] == "disparate_impact_ratio":
                if row["value"] < 0.8:
                    warnings.warn(
                        f"[FairnessWarning] disparate_impact_ratio for attribute "
                        f"'{row['attribute']}' = {row['value']:.4f} is below the "
                        f"80%% rule threshold (0.8).",
                        stacklevel=2,
                    )
