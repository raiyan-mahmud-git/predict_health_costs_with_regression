"""Shared project settings and schema."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "expenses"
NUMERIC_COLUMNS = ["age", "bmi", "children"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET]
RANGE_EXPECTATIONS = {
    "age": (18.0, 64.0),
    "bmi": (16.0, 53.1),
    "children": (0.0, 5.0),
}
RANDOM_SEED = 42
