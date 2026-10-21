"""Keras model construction"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_MODEL_DIR,
    NUMERIC_COLUMNS,
    RANGE_EXPECTATIONS,
    RANDOM_SEED,
)
from .preprocessing import engineer_features


def build_keras_model(input_dim: int, *, learning_rate: float = 1e-3) -> keras.Model:
    """Build a modest network with grader-compatible metric ordering."""
    keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dense(1),
        ],
        name="health_cost_regressor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae", "mse"],
    )
    return model


def early_stopping() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, min_delta=5.0, restore_best_weights=True
    )


def save_artifacts(model, preprocessor, output_dir=DEFAULT_MODEL_DIR, *, metadata=None) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "health_cost_model.keras")
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    payload = {
        "model_file": "health_cost_model.keras",
        "preprocessor_file": "preprocessor.joblib",
        "framework": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "input_ranges": RANGE_EXPECTATIONS,
        "target": "expenses",
        "warning": "Educational prediction on a small public teaching dataset; not real pricing.",
    }
    payload.update(metadata or {})
    (output / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_artifacts(model_dir=DEFAULT_MODEL_DIR):
    directory = Path(model_dir)
    model = keras.models.load_model(directory / "health_cost_model.keras")
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return model, preprocessor, metadata


def predict_expenses(inputs, model, preprocessor, *, warn_out_of_range=True) -> np.ndarray:
    """Predict from raw feature values and warn outside observed data ranges."""
    frame = inputs.copy() if isinstance(inputs, pd.DataFrame) else pd.DataFrame(inputs)
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction input is missing required features: {', '.join(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Prediction input contains missing required feature values.")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip().str.lower()
    allowed = {
        "sex": {"female", "male"},
        "smoker": {"no", "yes"},
        "region": {"northeast", "northwest", "southeast", "southwest"},
    }
    for column, values_allowed in allowed.items():
        unexpected = sorted(set(frame[column]) - values_allowed)
        if unexpected:
            raise ValueError(f"Unexpected {column} values: {', '.join(unexpected)}")
    for column, (lower, upper) in RANGE_EXPECTATIONS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Prediction feature {column!r} must contain numeric values.")
        frame[column] = values
        if warn_out_of_range and ((values < lower) | (values > upper)).any():
            warnings.warn(
                f"{column} is outside the observed teaching-dataset range [{lower:g}, {upper:g}]. "
                "The model has not been validated for extrapolation.",
                UserWarning,
                stacklevel=2,
            )
    transformed = preprocessor.transform(engineer_features(frame)).astype(np.float32)
    return np.asarray(model.predict(transformed, verbose=0)).reshape(-1)
