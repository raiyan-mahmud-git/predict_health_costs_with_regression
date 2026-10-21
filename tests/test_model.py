import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow is required for Keras model tests")

from health_costs.model import build_keras_model, load_artifacts, predict_expenses, save_artifacts
from health_costs.preprocessing import build_preprocessor, engineer_features


class ConstantModel:
    def predict(self, values, verbose=0):
        return np.full((len(values), 1), 1234.0)


def _raw_features():
    return pd.DataFrame(
        {
            "age": [30, 45, 50, 27], "bmi": [25.0, 33.2, 29.0, 22.0],
            "children": [0, 2, 1, 0], "sex": ["female", "male", "female", "male"],
            "smoker": ["no", "yes", "no", "no"],
            "region": ["southwest", "southeast", "northwest", "northeast"],
        }
    )


def test_grader_evaluate_returns_loss_mae_mse_in_order():
    model = build_keras_model(2)
    model.fit(np.ones((8, 2)), np.arange(8, dtype=float), epochs=1, verbose=0)
    metrics = model.evaluate(np.ones((3, 2)), np.array([1.0, 2.0, 3.0]), verbose=0)
    assert len(metrics) == 3
    assert model.loss == "mae"
    assert model.metrics[0].name == "loss"


def test_out_of_range_inputs_warn():
    raw = _raw_features()
    preprocessor = build_preprocessor().fit(engineer_features(raw))
    raw.loc[0, "age"] = 70
    with pytest.warns(UserWarning, match="outside the observed"):
        result = predict_expenses(raw.iloc[[0]], ConstantModel(), preprocessor)
    assert result.tolist() == [1234.0]


def test_saved_keras_and_preprocessor_reload_with_prediction_parity(tmp_path):
    raw = _raw_features()
    transformed = build_preprocessor().fit(engineer_features(raw))
    matrix = transformed.transform(engineer_features(raw)).astype(np.float32)
    model = build_keras_model(matrix.shape[1])
    model.fit(matrix, np.array([1000.0, 22000.0, 9000.0, 1800.0]), epochs=1, verbose=0)
    expected = model.predict(matrix, verbose=0)
    save_artifacts(model, transformed, tmp_path)
    reloaded, loaded_preprocessor, _ = load_artifacts(tmp_path)
    actual = reloaded.predict(
        loaded_preprocessor.transform(engineer_features(raw)).astype(np.float32), verbose=0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
