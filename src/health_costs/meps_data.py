"""Download and prepare AHRQ MEPS 2024 data for annual expenditure modeling."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

from .config import PROJECT_ROOT

MEPS_2024_URL = "https://meps.ahrq.gov/mepsweb/data_files/pufs/h256/h256dta.zip"
DEFAULT_MEPS_CACHE = PROJECT_ROOT / ".cache" / "meps" / "h256dta.zip"
MEPS_FEATURES = ["age", "bmi", "sex", "region", "smoking_frequency", "self_rated_health"]
MEPS_TARGET = "annual_healthcare_expenditure"
MEPS_WEIGHT = "survey_weight"
MEPS_GROUP = "household_group"


def _download_meps(path: Path, timeout_seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MEPS_2024_URL, headers={"User-Agent": "health-costs-educational-project/1.0"})
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            data_file = next(name for name in archive.namelist() if name.lower().endswith(".dta"))
            archive.read(data_file)
        temporary.write_bytes(payload)
        temporary.replace(path)
    except (OSError, URLError, BadZipFile, StopIteration) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download or validate the MEPS 2024 data within {timeout_seconds:g} seconds. "
            f"Check your connection, or provide a valid local MEPS Stata ZIP file. Details: {exc}"
        ) from exc
    return path


def _read_meps_archive(path: Path) -> pd.DataFrame:
    try:
        with ZipFile(path) as archive:
            data_file = next(name for name in archive.namelist() if name.lower().endswith(".dta"))
            return pd.read_stata(BytesIO(archive.read(data_file)), convert_categoricals=True)
    except (OSError, BadZipFile, StopIteration) as exc:
        raise ValueError(f"{path} is not a valid MEPS Stata ZIP archive: {exc}") from exc


def _numeric_code(values: pd.Series) -> pd.Series:
    """Decode numeric Stata labels such as '1 MALE' and sentinel labels."""
    text = values.astype("string")
    code = text.str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(code, errors="coerce")


def prepare_meps_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Map MEPS 2024 columns to adult-facing model inputs and clean sentinels.

    The resulting target is annual total health-care expenditure, not an
    insurance premium or a price quote. Rows are limited to adults with valid
    BMI and cigarette-frequency responses and a positive PSAQ expenditure
    survey weight.
    """
    required = {
        "AGE24X", "SEX", "REGION24", "ADBMI42", "OFTSMK53", "RTHLTH53", "TOTEXP24",
        "PESAQWT24F", "DUID",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"MEPS input is missing required columns: {', '.join(missing)}")

    age = pd.to_numeric(raw["AGE24X"], errors="coerce")
    sex_code = _numeric_code(raw["SEX"])
    region_code = _numeric_code(raw["REGION24"])
    bmi = _numeric_code(raw["ADBMI42"])
    smoke_code = _numeric_code(raw["OFTSMK53"])
    health_code = _numeric_code(raw["RTHLTH53"])
    expenditure = pd.to_numeric(raw["TOTEXP24"], errors="coerce")
    weight = pd.to_numeric(raw["PESAQWT24F"], errors="coerce")

    valid = (
        age.between(18, 120)
        & bmi.between(10, 50)
        & sex_code.isin([1, 2])
        & region_code.isin([1, 2, 3, 4])
        & smoke_code.isin([1, 2, 3])
        & health_code.isin([1, 2, 3, 4, 5])
        & expenditure.ge(0)
        & weight.gt(0)
        & raw["DUID"].notna()
    )
    cleaned = pd.DataFrame(
        {
            "age": age[valid].astype(float),
            "bmi": bmi[valid].astype(float),
            "sex": sex_code[valid].map({1: "Male", 2: "Female"}),
            "region": region_code[valid].map(
                {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
            ),
            "smoking_frequency": smoke_code[valid].map(
                {1: "Every day", 2: "Some days", 3: "Not at all"}
            ),
            "self_rated_health": health_code[valid].map(
                {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
            ),
            MEPS_TARGET: expenditure[valid].astype(float),
            MEPS_WEIGHT: weight[valid].astype(float),
            MEPS_GROUP: raw.loc[valid, "DUID"].astype(str),
        }
    ).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No MEPS rows remain after validating age, BMI, smoking, target, and weight.")
    return cleaned


def load_meps_dataset(
    path: str | Path | None = None, *, timeout_seconds: float = 45.0
) -> pd.DataFrame:
    """Load a local MEPS ZIP or the cached/downloaded AHRQ 2024 ZIP."""
    archive = Path(path) if path is not None else DEFAULT_MEPS_CACHE
    if not archive.is_file():
        if path is not None:
            raise FileNotFoundError(f"MEPS data archive not found: {archive}")
        _download_meps(archive, timeout_seconds)
    return prepare_meps_frame(_read_meps_archive(archive))
