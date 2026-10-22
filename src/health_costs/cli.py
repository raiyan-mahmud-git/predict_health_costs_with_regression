"""Command-line entry points for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR
from .data import load_dataset
from .meps_data import load_meps_dataset
from .meps_model import MEPS_MODEL_DIR, MEPS_REPORT_DIR, train_meps_project
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
    
    test_dataset = np.asarray(run.test_dataset, dtype=np.float32)
    truth = np.asarray(run.test_labels, dtype=np.float32).reshape(-1)
    predictions = run.model.predict(test_dataset, verbose=0).reshape(-1)
    mae = mean_absolute_error(truth, predictions)
    mse = mean_squared_error(truth, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(truth, predictions)
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


def _meps_train(args) -> int:
    data = load_meps_dataset(args.data, timeout_seconds=args.timeout)
    print(
        f"Loaded {len(data):,} valid adult MEPS records from 2024 "
        f"across {data['household_group'].nunique():,} households."
    )
    result = train_meps_project(
        data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
    )
    print("\nGrouped, survey-weighted 5-fold CV:")
    print(result["cv_results"].to_string(index=False, formatters={"mae_mean": "${:,.2f}".format}))
    metrics = result["metadata"]["holdout_weighted_metrics"]
    print("\nUntouched, household-grouped 20% holdout (survey-weighted):")
    print(
        f"MAE: ${metrics['mae']:,.2f} | RMSE: ${metrics['rmse']:,.2f} | "
        f"R²: {metrics['r2']:.3f}"
    )
    print(f"MEPS model and metadata saved to {Path(args.model_dir).resolve()}")
    print(f"CV results saved to {Path(args.report_dir).resolve()}")
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

    meps_train = commands.add_parser(
        "meps-train", help="download/load MEPS 2024, run grouped CV, and train the expenditure model"
    )
    meps_train.add_argument("--data", help="local AHRQ MEPS Stata ZIP; otherwise use cache or download")
    meps_train.add_argument("--timeout", type=float, default=45.0, help="MEPS download timeout in seconds")
    meps_train.add_argument("--model-dir", default=str(MEPS_MODEL_DIR))
    meps_train.add_argument("--report-dir", default=str(MEPS_REPORT_DIR))
    meps_train.set_defaults(handler=_meps_train)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
