"""Leakage-safe feature engineering and preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def engineer_features(frame: pd.DataFrame, *, include_interaction: bool = True) -> pd.DataFrame:
    """Create model features from the raw FCC input columns."""
    features = frame.loc[:, NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    if include_interaction:
        smoker_flag = features["smoker"].astype(str).str.lower().eq("yes").astype(float)
        features["smoker_bmi_interaction"] = features["bmi"].astype(float) * smoker_flag
    return features


def build_preprocessor(*, include_interaction: bool = True) -> ColumnTransformer:
    numeric = NUMERIC_COLUMNS + (["smoker_bmi_interaction"] if include_interaction else [])
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
