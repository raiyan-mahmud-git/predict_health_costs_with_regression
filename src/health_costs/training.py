"""Cross-validation and final leakage-safe Keras training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .config import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, RANDOM_SEED, TARGET
from .preprocessing import build_preprocessor, engineer_features


@dataclass
class TrainingRun:
    model: Any
    preprocessor: Any
    train_dataset: np.ndarray
    train_labels: pd.Series
    validation_dataset: np.ndarray
    validation_labels: pd.Series
    test_dataset: np.ndarray
    test_labels: pd.Series
    history: Any
    cv_results: pd.DataFrame
    heldout_indices: np.ndarray


def duplicate_groups(frame: pd.DataFrame) -> np.ndarray:
    """Give exact duplicate records the same group id for splitting."""
    return pd.util.hash_pandas_object(frame, index=False).astype(str).to_numpy()


def _classical_candidates():
    def pipeline(estimator, *, interaction=True):
        return Pipeline(
            [
                ("preprocess", build_preprocessor(include_interaction=interaction)),
                ("regressor", estimator),
            ]
        )

    params = {
        "n_estimators": 180,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 8,
        "random_state": RANDOM_SEED,
    }
    log_boosting = TransformedTargetRegressor(
        regressor=pipeline(GradientBoostingRegressor(loss="squared_error", **params)),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return {
        "mean_baseline": pipeline(DummyRegressor(strategy="mean")),
        "linear_with_interaction": pipeline(LinearRegression()),
        "gradient_boosting_with_interaction": pipeline(
            GradientBoostingRegressor(loss="absolute_error", **params)
        ),
        "gradient_boosting_log_target": log_boosting,
        "gradient_boosting_without_interaction": pipeline(
            GradientBoostingRegressor(loss="absolute_error", **params), interaction=False
        ),
    }


def evaluate_classical_cv(development_data: pd.DataFrame, *, n_splits: int = 5) -> pd.DataFrame:
    """Compare mean/linear/boosted references with grouped CV.

    Exact duplicates remain in the data, but share a group and therefore cannot
    appear on opposite sides of any fold. The final test set is excluded by the
    caller before this function runs.
    """
    raw_features = development_data.drop(columns=TARGET)
    labels = development_data[TARGET].to_numpy(dtype=float)
    groups = duplicate_groups(development_data)
    splitter = GroupKFold(n_splits=n_splits)
    records: list[dict[str, Any]] = []
    for name, estimator in _classical_candidates().items():
        fold_scores = []
        interaction = "without_interaction" not in name
        for train_idx, valid_idx in splitter.split(raw_features, labels, groups):
            train_features = engineer_features(raw_features.iloc[train_idx], include_interaction=interaction)
            valid_features = engineer_features(raw_features.iloc[valid_idx], include_interaction=interaction)
            estimator.fit(train_features, labels[train_idx])
            predicted = np.maximum(estimator.predict(valid_features), 0.0)
            truth = labels[valid_idx]
            fold_scores.append(
                (
                    mean_absolute_error(truth, predicted),
                    mean_squared_error(truth, predicted) ** 0.5,
                    r2_score(truth, predicted),
                )
            )
        score_array = np.asarray(fold_scores)
        records.append(
            {
                "model": name,
                "folds": len(fold_scores),
                "mae_mean": score_array[:, 0].mean(),
                "mae_std": score_array[:, 0].std(ddof=1),
                "rmse_mean": score_array[:, 1].mean(),
                "r2_mean": score_array[:, 2].mean(),
            }
        )
    return pd.DataFrame(records).sort_values("mae_mean").reset_index(drop=True)


def _grouped_split(frame: pd.DataFrame, *, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = duplicate_groups(frame)
    train_idx, held_idx = next(splitter.split(frame, frame[TARGET], groups))
    return train_idx, held_idx


def evaluate_keras_cv(
    development_data: pd.DataFrame, *, n_splits: int = 5, max_epochs: int = 300
) -> pd.DataFrame:
    """Estimate Keras model performance with grouped CV on development rows."""
    from tensorflow import keras

    from .model import build_keras_model, early_stopping

    labels = development_data[TARGET].to_numpy(dtype=np.float32)
    groups = duplicate_groups(development_data)
    splitter = GroupKFold(n_splits=n_splits)
    scores = []
    for fold_number, (fit_and_validation_idx, holdout_idx) in enumerate(
        splitter.split(development_data, labels, groups), start=1
    ):
        fold_development = development_data.iloc[fit_and_validation_idx].reset_index(drop=True)
        fold_holdout = development_data.iloc[holdout_idx].reset_index(drop=True)
        fit_idx, validation_idx = _grouped_split(
            fold_development, test_size=0.125, seed=RANDOM_SEED + fold_number
        )
        raw_fit = fold_development.drop(columns=TARGET).iloc[fit_idx]
        raw_validation = fold_development.drop(columns=TARGET).iloc[validation_idx]
        raw_holdout = fold_holdout.drop(columns=TARGET)
        preprocessor = build_preprocessor()
        fit_matrix = preprocessor.fit_transform(engineer_features(raw_fit)).astype(np.float32)
        validation_matrix = preprocessor.transform(engineer_features(raw_validation)).astype(np.float32)
        holdout_matrix = preprocessor.transform(engineer_features(raw_holdout)).astype(np.float32)
        keras.backend.clear_session()
        keras.utils.set_random_seed(RANDOM_SEED + fold_number)
        model = build_keras_model(fit_matrix.shape[1], learning_rate=1e-3)
        model.fit(
            fit_matrix,
            fold_development[TARGET].iloc[fit_idx].to_numpy(dtype=np.float32),
            validation_data=(
                validation_matrix,
                fold_development[TARGET].iloc[validation_idx].to_numpy(dtype=np.float32),
            ),
            epochs=max_epochs,
            batch_size=32,
            callbacks=[early_stopping()],
            verbose=0,
        )
        predicted = np.maximum(model.predict(holdout_matrix, verbose=0).reshape(-1), 0.0)
        truth = fold_holdout[TARGET].to_numpy(dtype=float)
        scores.append(
            (
                mean_absolute_error(truth, predicted),
                mean_squared_error(truth, predicted) ** 0.5,
                r2_score(truth, predicted),
            )
        )
    score_array = np.asarray(scores)
    return pd.DataFrame(
        [
            {
                "model": "keras_neural_network",
                "folds": len(scores),
                "mae_mean": score_array[:, 0].mean(),
                "mae_std": score_array[:, 0].std(ddof=1),
                "rmse_mean": score_array[:, 1].mean(),
                "r2_mean": score_array[:, 2].mean(),
            }
        ]
    )


def train_project(
    data: pd.DataFrame,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    run_cv: bool = True,
    max_epochs: int = 300,
    verbose: int = 0,
) -> TrainingRun:
    """Train and save the final model while keeping the test set untouched."""
    data = data.reset_index(drop=True).copy()
    if len(data) < 50:
        raise ValueError("At least 50 validated rows are needed for grouped splits and training.")

    development_idx, test_idx = _grouped_split(data, test_size=0.20, seed=RANDOM_SEED)
    development = data.iloc[development_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    # Reference-model CV uses development rows only, never the final test rows.
    if run_cv:
        cv_results = pd.concat(
            [
                evaluate_classical_cv(development),
                evaluate_keras_cv(development, max_epochs=max_epochs),
            ],
            ignore_index=True,
        ).sort_values("mae_mean").reset_index(drop=True)
    else:
        cv_results = pd.DataFrame()
    if run_cv:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(report_path / "cross_validation.csv", index=False)

    train_idx, validation_idx = _grouped_split(development, test_size=0.125, seed=RANDOM_SEED + 1)
    raw_train = development.drop(columns=TARGET).iloc[train_idx]
    raw_validation = development.drop(columns=TARGET).iloc[validation_idx]
    raw_test = test.drop(columns=TARGET)
    train_labels = development[TARGET].iloc[train_idx].reset_index(drop=True)
    validation_labels = development[TARGET].iloc[validation_idx].reset_index(drop=True)
    test_labels = test[TARGET].reset_index(drop=True)

    train_features = engineer_features(raw_train)
    validation_features = engineer_features(raw_validation)
    test_features = engineer_features(raw_test)
    preprocessor = build_preprocessor()
    train_dataset = preprocessor.fit_transform(train_features).astype(np.float32)
    validation_dataset = preprocessor.transform(validation_features).astype(np.float32)
    test_dataset = preprocessor.transform(test_features).astype(np.float32)

    # Import TensorFlow only for the Keras training step so CV utilities and
    # preprocessing can be used independently.
    from tensorflow import keras

    from .model import build_keras_model, early_stopping, save_artifacts

    keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED)
    model = build_keras_model(train_dataset.shape[1], learning_rate=1e-3)
    history = model.fit(
        train_dataset,
        train_labels.to_numpy(dtype=np.float32),
        validation_data=(validation_dataset, validation_labels.to_numpy(dtype=np.float32)),
        epochs=max_epochs,
        batch_size=32,
        callbacks=[early_stopping()],
        verbose=verbose,
    )
    save_artifacts(
        model,
        preprocessor,
        model_dir,
        metadata={
            "seed": RANDOM_SEED,
            "learning_rate": 1e-3,
            "epochs_trained": len(history.history["loss"]),
            "development_rows": int(len(development)),
            "test_rows": int(len(test)),
            "interaction_feature": "bmi * 1(smoker == yes)",
            "cv_method": "5-fold GroupKFold on development data; exact duplicate rows stay together",
        },
    )
    return TrainingRun(
        model=model,
        preprocessor=preprocessor,
        train_dataset=train_dataset,
        train_labels=train_labels,
        validation_dataset=validation_dataset,
        validation_labels=validation_labels,
        test_dataset=test_dataset,
        test_labels=test_labels,
        history=history,
        cv_results=cv_results,
        heldout_indices=np.asarray(test_idx),
    )
