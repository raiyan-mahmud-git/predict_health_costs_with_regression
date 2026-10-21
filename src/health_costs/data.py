"""Data loading, caching, and validation."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_URL, DEFAULT_CACHE_DIR, REQUIRED_COLUMNS, TARGET


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the source schema and return a cleanly typed copy.

    Exact duplicate rows are retained for challenge compatibility; evaluation
    groups identical rows so copies cannot straddle CV folds.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ["age", "bmi", "children", TARGET]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[REQUIRED_COLUMNS].isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing or non-numeric values in: {', '.join(bad)}")
    if (result[TARGET] <= 0).any():
        raise ValueError("All expenses must be positive for the log-target comparison.")
    allowed = {
        "sex": {"female", "male"},
        "smoker": {"no", "yes"},
        "region": {"northeast", "northwest", "southeast", "southwest"},
    }
    for column, values in allowed.items():
        result[column] = result[column].astype(str).str.strip().str.lower()
        unknown = sorted(set(result[column]) - values)
        if unknown:
            raise ValueError(f"Unexpected {column} values: {', '.join(unknown)}")
    if result.empty:
        raise ValueError("Dataset is empty.")
    return result.reset_index(drop=True)


def load_dataset(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    timeout_seconds: float = 20.0,
) -> pd.DataFrame:
    """Use an explicit CSV, the project cache, then a timed FCC download."""
    if path is not None:
        csv_path = Path(path).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return validate_dataset(pd.read_csv(csv_path))
    cache_path = Path(cache_dir).expanduser() / "insurance.csv"
    if cache_path.is_file():
        return validate_dataset(pd.read_csv(cache_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".csv.part")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "health-costs-regression/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
        if not payload:
            raise ValueError("The FCC dataset download returned an empty response.")
        temporary_path.write_bytes(payload)
        validated = validate_dataset(pd.read_csv(temporary_path))
        temporary_path.replace(cache_path)
        return validated
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, pd.errors.ParserError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download or validate the FCC dataset from {DATA_URL} "
            f"(timeout {timeout_seconds:g}s). No cached copy is available. "
            "Retry with network access or provide a local CSV using --data PATH."
        ) from exc
