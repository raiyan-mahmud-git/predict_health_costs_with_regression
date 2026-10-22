"""Survey-weighted MEPS model training and inference helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, RANDOM_SEED
from .meps_data import (
    MEPS_FEATURES,
    MEPS_GROUP,
    MEPS_TARGET,
    MEPS_WEIGHT,
    load_meps_dataset,
)

MEPS_MODEL_DIR = DEFAULT_MODEL_DIR / "meps_2024"
MEPS_REPORT_DIR = DEFAULT_REPORT_DIR / "meps_2024"
NUMERIC_FEATURES = ["age", "bmi"]
CATEGORICAL_FEATURES = ["sex", "region", "smoking_frequency", "self_rated_health"]
VALID_CATEGORIES = {
    "sex": {"Female", "Male"},
    "region": {"Northeast", "Midwest", "South", "West"},
    "smoking_frequency": {"Every day", "Some days", "Not at all"},
    "self_rated_health": {"Excellent", "Very good", "Good", "Fair", "Poor"},
}


def build_meps_estimator() -> Pipeline:
    """Return a fixed, regularized model for annual expenditure prediction."""
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _weighted_metrics(actual, predicted, weights) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted, sample_weight=weights)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted, sample_weight=weights))),
        "r2": float(r2_score(actual, predicted, sample_weight=weights)),
    }


def evaluate_meps_cv(frame: pd.DataFrame, *, n_splits: int = 5) -> pd.DataFrame:
    """Grouped weighted CV; records from one household stay in the same fold."""
    groups = frame[MEPS_GROUP].astype(str).to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"At least {n_splits} distinct households are required for grouped CV.")
    x = frame[MEPS_FEATURES]
    y = frame[MEPS_TARGET].to_numpy(dtype=float)
    weights = frame[MEPS_WEIGHT].to_numpy(dtype=float)
    cv = GroupKFold(n_splits=n_splits)
    fold_metrics: dict[str, list[dict[str, float]]] = {"weighted_mean_baseline": [], "hist_gradient_boosting": []}
    for train_idx, valid_idx in cv.split(x, y, groups):
        baseline = np.average(y[train_idx], weights=weights[train_idx])
        fold_metrics["weighted_mean_baseline"].append(
            _weighted_metrics(y[valid_idx], np.full(len(valid_idx), baseline), weights[valid_idx])
        )
        estimator = build_meps_estimator()
        estimator.fit(x.iloc[train_idx], y[train_idx], regressor__sample_weight=weights[train_idx])
        prediction = np.maximum(estimator.predict(x.iloc[valid_idx]), 0.0)
        fold_metrics["hist_gradient_boosting"].append(
            _weighted_metrics(y[valid_idx], prediction, weights[valid_idx])
        )

    rows = []
    for name, scores in fold_metrics.items():
        rows.append(
            {
                "model": name,
                "folds": n_splits,
                **{f"{metric}_mean": float(np.mean([score[metric] for score in scores])) for metric in ("mae", "rmse", "r2")},
                "mae_std": float(np.std([score["mae"] for score in scores], ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)


def train_meps_project(
    frame: pd.DataFrame | None = None,
    *,
    model_dir: str | Path = MEPS_MODEL_DIR,
    report_dir: str | Path = MEPS_REPORT_DIR,
    timeout_seconds: float = 45.0,
) -> dict:
    """Train with a household-grouped holdout and CV, then save model/metadata."""
    data = (load_meps_dataset(timeout_seconds=timeout_seconds) if frame is None else frame.copy())
    if len(data) < 100:
        raise ValueError("At least 100 valid MEPS adults are needed for training and evaluation.")

    groups = data[MEPS_GROUP].astype(str).to_numpy()
    if len(np.unique(groups)) < 6:
        raise ValueError("MEPS training data needs at least six distinct households.")
    x = data[MEPS_FEATURES]
    y = data[MEPS_TARGET].to_numpy(dtype=float)
    weights = data[MEPS_WEIGHT].to_numpy(dtype=float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    development = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    cv_results = evaluate_meps_cv(development)
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(reports / "cross_validation.csv", index=False)

    model = build_meps_estimator()
    model.fit(
        development[MEPS_FEATURES],
        development[MEPS_TARGET].to_numpy(dtype=float),
        regressor__sample_weight=development[MEPS_WEIGHT].to_numpy(dtype=float),
    )
    predictions = np.maximum(model.predict(test[MEPS_FEATURES]), 0.0)
    holdout_metrics = _weighted_metrics(
        test[MEPS_TARGET].to_numpy(dtype=float), predictions, test[MEPS_WEIGHT].to_numpy(dtype=float)
    )

    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "annual_expenditure.joblib")
    metadata = {
        "dataset": "AHRQ MEPS HC-256, 2024 Full Year Consolidated Public Use File",
        "target": MEPS_TARGET,
        "target_description": "Annual total health-care expenditure in 2024 dollars; not an insurance premium or quote.",
        "features": MEPS_FEATURES,
        "feature_ranges": {
            "age": [float(data["age"].min()), float(data["age"].max())],
            "bmi": [float(data["bmi"].min()), float(data["bmi"].max())],
        },
        "valid_rows": int(len(data)),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(test)),
        "weight": "PESAQWT24F survey weight",
        "split": "GroupShuffleSplit by household; 20% holdout",
        "cv": "5-fold GroupKFold by household on development rows only",
        "holdout_weighted_metrics": holdout_metrics,
        "warning": "Educational estimate from survey expenditures; not a clinical or insurance-pricing decision.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": model, "metadata": metadata, "cv_results": cv_results}


def predict_meps_expenditure(features: pd.DataFrame, model, metadata: dict | None = None) -> np.ndarray:
    """Validate public-facing inputs, warn outside training range, and predict."""
    missing = sorted(set(MEPS_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Missing MEPS prediction features: {', '.join(missing)}")
    frame = features.loc[:, MEPS_FEATURES].copy()
    for column, allowed in VALID_CATEGORIES.items():
        unexpected = sorted(set(frame[column].dropna()) - allowed)
        if unexpected:
            raise ValueError(f"Invalid {column} value(s): {', '.join(map(str, unexpected))}")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} must contain numeric values.")
        if metadata and column in metadata.get("feature_ranges", {}):
            lower, upper = metadata["feature_ranges"][column]
            if ((frame[column] < lower) | (frame[column] > upper)).any():
                warnings.warn(
                    f"{column} is outside the observed adult MEPS training range [{lower:g}, {upper:g}].",
                    UserWarning,
                    stacklevel=2,
                )
    if frame.isna().any().any():
        raise ValueError("Prediction input contains missing feature values.")
    return np.maximum(np.asarray(model.predict(frame), dtype=float).reshape(-1), 0.0)


def load_meps_artifacts(model_dir: str | Path = MEPS_MODEL_DIR):
    directory = Path(model_dir)
    model = joblib.load(directory / "annual_expenditure.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata
