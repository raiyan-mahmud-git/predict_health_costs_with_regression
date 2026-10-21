# Health Costs Regression

A reproducible, educational upgrade of the FreeCodeCamp health-costs regression challenge. It predicts the `expenses` column from age, BMI, children, sex, smoking status, and region. This small public teaching dataset is not suitable for real insurance pricing or underwriting.

## Project layout

```text
.
├── DATASET_ATTRIBUTION.md
├── LICENSE
├── notebooks/Copy_of_fcc_predict_health_costs_with_regression.ipynb
├── src/health_costs/       # data, preprocessing, training, inference, CLI
├── tests/
├── models/                 # generated .keras model + joblib preprocessor (ignored)
└── reports/                # generated CV results (ignored)
```

## Runtime versions

Use **Python 3.12**, **TensorFlow 2.16.2**, and **Keras 3.4.1**. All direct dependencies and notebook/test tools are pinned in the requirements files. The `.keras` model and the scikit-learn `ColumnTransformer` are saved as a bundle: `health_cost_model.keras`, `preprocessor.joblib`, and `metadata.json`.

## Setup

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

## Data and offline use

The first run without `--data` downloads the CSV from the FreeCodeCamp URL in `DATASET_ATTRIBUTION.md`, with a 20-second timeout, and caches it at `.cache/insurance.csv`. Later runs use that cache without network access. If the download fails and there is no cache, the command exits with a clear error; supply a local copy with `--data PATH` to work offline. The project does not silently substitute data.

The loader validates required columns and values, reports duplicate rows, and retains them to preserve the challenge dataset. Identical records are kept in the same validation/test fold so a duplicate cannot leak across a split. The expected published data has 1,338 rows, ages 18–64 and children 0–5; BMI spans approximately 16.0–53.1. These observed ranges are used for inference warnings, not hard input rejection.

## Train and evaluate

```bash
health-costs train
# or use an existing CSV without downloading:
health-costs train --data path/to/insurance.csv
```

Training first holds out a grouped 20% test set. Five-fold `GroupKFold` then evaluates the Keras network and compares a mean baseline, linear regression, and gradient boosting on the remaining development rows. It compares gradient boosting with and without the smoker×BMI interaction and compares raw expenses with a log-transformed target. Each Keras fold uses its own validation subset for early stopping; preprocessing is fitted on that fold's training rows only. CV scores and fold-to-fold MAE spread are written to `reports/cross_validation.csv`. The final Keras network uses the interaction feature, a 0.001 learning rate, and early stopping on a separate validation subset. The holdout test set is not used for CV or model selection.

The Keras model is compiled with `loss="mae"` and metrics `['mae', 'mse']` in that order. Its test input is the already-transformed dense output of the saved `ColumnTransformer`; therefore the original challenge grader can call `model.evaluate(test_dataset, test_labels)` and unpack `loss, mae, mse`. The challenge cell is preserved verbatim in the notebook. A test MAE below $3,500 is reported only if that run actually achieves it.

The notebook can be run from the project root after setup:

```bash
python -m nbconvert --to notebook --execute \
  notebooks/Copy_of_fcc_predict_health_costs_with_regression.ipynb \
  --output executed_health_costs.ipynb --output-dir reports \
  --ExecutePreprocessor.timeout=900
```

## Predict with saved artifacts

Create a CSV with `age,bmi,children,sex,smoker,region` columns, then run:

```bash
health-costs predict --input examples/prediction_input.csv --output predictions.csv
```

The input features remain in their original units and categories. Missing or invalid categories are rejected; a valid category absent from a particular training fold is handled by the encoder's unknown-category rule. Values outside the observed age, BMI, or children ranges generate warnings because extrapolation has not been validated.

## Checks

```bash
pytest
```

Tests cover schema/data loading, caching and offline errors, interaction encoding, duplicate grouping, grader metric ordering, out-of-range warnings, and model/preprocessor save-and-reload prediction parity.

## Interpretation and limitations

Expenses are strongly right-skewed, and smoking status has a large association in this dataset. The BMI×smoker feature and log-target boosted-tree comparison test two ways to account for that structure; they do not establish causality. A single held-out MAE is only one sample estimate, so inspect the grouped CV mean and spread alongside it. The data is small, public, and intended for teaching. It may contain historical collection or sampling biases and must not be treated as a current quote, a medical judgment, or a real pricing tool.


