import numpy as np
import pandas as pd

from health_costs.preprocessing import build_preprocessor, engineer_features
from health_costs.training import duplicate_groups


def test_smoker_bmi_interaction_is_only_nonzero_for_smokers():
    raw = pd.DataFrame(
        {
            "age": [30, 30], "bmi": [25.0, 25.0], "children": [0, 0],
            "sex": ["female", "male"], "smoker": ["no", "yes"],
            "region": ["southwest", "southeast"],
        }
    )
    features = engineer_features(raw)
    assert features["smoker_bmi_interaction"].tolist() == [0.0, 25.0]


def test_preprocessor_handles_unseen_categories():
    train = pd.DataFrame(
        {
            "age": [20, 40], "bmi": [22.0, 31.0], "children": [0, 2],
            "sex": ["female", "male"], "smoker": ["no", "yes"],
            "region": ["northwest", "southeast"],
        }
    )
    preprocessor = build_preprocessor().fit(engineer_features(train))
    unseen = train.iloc[[0]].copy()
    unseen["region"] = "northeast"
    output = preprocessor.transform(engineer_features(unseen))
    assert output.shape[0] == 1
    assert np.isfinite(output).all()


def test_exact_duplicate_rows_share_a_group(sample_data):
    duplicate = sample_data.iloc[[0]]
    frame = pd.concat([sample_data, duplicate], ignore_index=True)
    groups = duplicate_groups(frame)
    assert groups[0] == groups[-1]
