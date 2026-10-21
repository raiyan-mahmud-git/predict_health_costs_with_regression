import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rows = []
    for i in range(60):
        rows.append(
            {
                "age": 18 + (i % 47),
                "sex": "female" if i % 2 else "male",
                "bmi": 20.0 + (i % 20),
                "children": i % 4,
                "smoker": "yes" if i % 5 == 0 else "no",
                "region": ["southwest", "southeast", "northwest", "northeast"][i % 4],
                "expenses": 1000.0 + 120.0 * (i % 47) + (18000.0 if i % 5 == 0 else 0.0),
            }
        )
    return pd.DataFrame(rows)
