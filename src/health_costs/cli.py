"""Command-line entry points for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .config import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR
from .data import load_dataset
from .model import load_artifacts, predict_expenses
from .training import train_project


def _train(args) -> int:
    data = load_dataset(args.data, timeout_seconds=args.timeout)
    duplicates = int(data.duplicated().sum())
    print(f"Loaded {len(data):,} rows; exact duplicate rows retained: {duplicates}.")
    run = train_project(
        data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        run_cv=not args.skip_cv,
        max_epochs=args.epochs,
        verbose=1 if args.verbose else 0,
    )
    if not run.cv_results.empty:
        print("\nGrouped 5-fold reference-model CV (development data only):")
        print(
            run.cv_results.to_string(
                index=False,
                formatters={
                    "mae_mean": "{:,.2f}".format,
                    "mae_std": "{:,.2f}".format,
                    "rmse_mean": "{:,.2f}".format,
                    "r2_mean": "{:.3f}".format,
                },
            )
        )
    _, mae, mse = run.model.evaluate(run.test_dataset, run.test_labels, verbose=0)
    predictions = run.model.predict(run.test_dataset, verbose=0).reshape(-1)
    rmse = mean_squared_error(run.test_labels, predictions) ** 0.5
    r2 = r2_score(run.test_labels, predictions)
    print(f"\nFinal untouched grouped holdout ({len(run.test_labels)} rows):")
    print(f"MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.3f} | Keras MSE: {mse:,.2f}")
    print(f"Artifacts saved to {Path(args.model_dir).resolve()}")
    return 0


def _predict(args) -> int:
    frame = pd.read_csv(args.input)
    model, preprocessor, _ = load_artifacts(args.model_dir)
    frame["predicted_expenses"] = predict_expenses(frame, model, preprocessor)
    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Predictions written to {Path(args.output).resolve()}")
    else:
        print(frame.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="health-costs")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run CV, train Keras, and report holdout metrics")
    train.add_argument("--data", help="local CSV; otherwise use cache or download the FCC CSV")
    train.add_argument("--timeout", type=float, default=20.0, help="download timeout in seconds (default: 20)")
    train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    train.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--skip-cv", action="store_true", help="skip reference-model cross-validation")
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(handler=_train)

    predict = commands.add_parser("predict", help="predict expenses from raw input-feature CSV")
    predict.add_argument("--input", required=True, help="CSV with age,bmi,children,sex,smoker,region")
    predict.add_argument("--output", help="optional output CSV path")
    predict.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    predict.set_defaults(handler=_predict)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
