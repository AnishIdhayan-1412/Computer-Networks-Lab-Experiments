"""
data_utils.py
-------------
Dataset loading and preprocessing utilities for the Bias Mitigated Hiring Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_hiring_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a synthetic hiring dataset with demographic bias baked in.

    Columns
    -------
    years_experience  : continuous, 0–15
    education_level   : ordinal, 1 (high school) – 4 (PhD)
    skills_score      : continuous, 0–100
    interview_score   : continuous, 0–100
    gender            : binary sensitive attribute (0=female, 1=male)
    race              : categorical sensitive attribute (0=minority, 1=majority)
    hired             : binary outcome label (1=hired, 0=not hired)

    Bias
    ----
    The label generation intentionally inflates hire probability for the
    majority group so that a naïve model will learn biased representations.
    """
    rng = np.random.default_rng(random_state)

    years_experience = rng.uniform(0, 15, n_samples)
    education_level = rng.integers(1, 5, n_samples)
    skills_score = rng.uniform(0, 100, n_samples)
    interview_score = rng.uniform(0, 100, n_samples)
    gender = rng.integers(0, 2, n_samples)   # 0=female, 1=male
    race = rng.integers(0, 2, n_samples)     # 0=minority, 1=majority

    # Merit-based probability
    merit = (
        0.05 * years_experience
        + 0.1 * education_level
        + 0.004 * skills_score
        + 0.004 * interview_score
    )

    # Introduce historical bias: majority group gets a +0.15 boost
    bias_boost = 0.15 * race + 0.10 * gender

    probability = np.clip(merit + bias_boost, 0.0, 1.0)
    hired = rng.binomial(1, probability, n_samples)

    return pd.DataFrame(
        {
            "years_experience": years_experience,
            "education_level": education_level,
            "skills_score": skills_score,
            "interview_score": interview_score,
            "gender": gender,
            "race": race,
            "hired": hired,
        }
    )


def load_and_split(
    df: pd.DataFrame,
    target_column: str = "hired",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split *df* into train / test feature matrices and label vectors.

    Parameters
    ----------
    df            : full dataset as a DataFrame
    target_column : name of the binary outcome column
    test_size     : fraction of samples for the test split
    random_state  : random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Label-encode categorical string columns in place and return the DataFrame."""
    df = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = le.fit_transform(df[col].astype(str))
    return df
