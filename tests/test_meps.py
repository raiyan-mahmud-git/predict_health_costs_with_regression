import numpy as np
import pandas as pd
import pytest

from health_costs.meps_data import (
    MEPS_FEATURES,
    MEPS_GROUP,
    MEPS_TARGET,
    MEPS_WEIGHT,
    prepare_meps_frame,
)
from health_costs.meps_model import build_meps_estimator, predict_meps_expenditure


def test_meps_cleanup_decodes_codes_and_drops_invalid_adult_records():
    raw = pd.DataFrame(
        {
            "AGE24X": [30, 45, 17, 50, 41, 38],
            "SEX": ["1 MALE", "2 FEMALE", "1 MALE", "2 FEMALE", "1 MALE", "2 FEMALE"],
            "REGION24": ["1 NORTHEAST", "4 WEST", "3 SOUTH", "2 MIDWEST", "-1 INAPPLICABLE", "1 NORTHEAST"],
            "ADBMI42": [25.0, 30.1, 28.0, "-15 CANNOT BE COMPUTED", 22.0, 31.0],
            "OFTSMK53": ["3 NOT AT ALL", "2 SOME DAYS", "3 NOT AT ALL", "1 EVERY DAY", "3 NOT AT ALL", "-7 REFUSED"],
            "RTHLTH53": ["2 VERY GOOD", "3 GOOD", "1 EXCELLENT", "4 FAIR", "5 POOR", "1 EXCELLENT"],
            "TOTEXP24": [400, 2000, 10, 100, 0, 900],
            "PESAQWT24F": [10, 8, 9, 8, 10, 0],
            "DUID": [101, 102, 103, 104, 105, 106],
        }
    )

    cleaned = prepare_meps_frame(raw)

    assert len(cleaned) == 2
    assert cleaned.loc[0, "sex"] == "Male"
    assert cleaned.loc[0, "region"] == "Northeast"
    assert cleaned.loc[0, "smoking_frequency"] == "Not at all"
    assert cleaned.loc[1, MEPS_TARGET] == 2000
    assert cleaned.loc[1, MEPS_WEIGHT] == 8


def test_meps_estimator_fits_and_accepts_public_input_features():
    rows = 80
    frame = pd.DataFrame(
        {
            "age": np.linspace(18, 80, rows),
            "bmi": np.linspace(19, 42, rows),
            "sex": np.where(np.arange(rows) % 2, "Female", "Male"),
            "region": np.array(["Northeast", "Midwest", "South", "West"])[np.arange(rows) % 4],
            "smoking_frequency": np.array(["Not at all", "Some days", "Every day"])[np.arange(rows) % 3],
            "self_rated_health": np.array(["Excellent", "Very good", "Good", "Fair", "Poor"])[np.arange(rows) % 5],
        }
    )
    target = 800 + frame["age"] * 75 + frame["bmi"] * 60 + (frame["smoking_frequency"] == "Every day") * 2500
    estimator = build_meps_estimator()
    estimator.fit(frame[MEPS_FEATURES], target, regressor__sample_weight=np.linspace(1, 5, rows))

    estimate = predict_meps_expenditure(frame.iloc[[0]], estimator)

    assert estimate.shape == (1,)
    assert np.isfinite(estimate[0])
    assert estimate[0] >= 0


def test_meps_prediction_rejects_invalid_categories_and_warns_outside_range():
    class ConstantEstimator:
        def predict(self, frame):
            return np.full(len(frame), 1200.0)

    row = pd.DataFrame(
        [{"age": 90, "bmi": 28.0, "sex": "Female", "region": "South", "smoking_frequency": "Not at all", "self_rated_health": "Good"}]
    )
    metadata = {"feature_ranges": {"age": [18.0, 85.0], "bmi": [10.0, 50.0]}}
    with pytest.warns(UserWarning, match="outside the observed adult MEPS training range"):
        result = predict_meps_expenditure(row, ConstantEstimator(), metadata)
    assert result.tolist() == [1200.0]

    row.loc[0, "region"] = "Unknown"
    with pytest.raises(ValueError, match="Invalid region"):
        predict_meps_expenditure(row, ConstantEstimator(), metadata)
