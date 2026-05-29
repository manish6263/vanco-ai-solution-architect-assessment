"""Train and evaluate forecasting models for Use Case 1."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, ID_COLUMN, MODELS_DIR, REPORTS_DIR, TARGET
from .data_loader import load_all
from .dataset import build_modeling_frame
from .features import feature_columns, make_features
from .metrics import rmsle
from .models import (
    BASELINE_MODELS,
    feature_importance_frame,
    predict_lightgbm_sales,
    train_lightgbm_log_target,
)
from .validation import describe_window, make_holdout_window, split_by_window


def log_step(message: str) -> None:
    """Print a timestamped progress message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [train] {message}", flush=True)


def evaluate_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate baseline forecasts on the final time-aware validation window."""
    log_step("evaluating baseline models")
    train_rows = frame.loc[frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    train_split, valid_split = split_by_window(train_rows, window)
    log_step(
        "baseline split: "
        f"train={len(train_split):,} rows, valid={len(valid_split):,} rows, "
        f"window={window.valid_start_date.date()} to {window.valid_end_date.date()}"
    )

    rows = []
    for model_name, predict_fn in BASELINE_MODELS.items():
        log_step(f"running baseline: {model_name}")
        predictions = predict_fn(train_split, valid_split)
        rows.append(
            {
                **describe_window(window),
                "model": model_name,
                "valid_rows": len(valid_split),
                "rmsle": rmsle(valid_split[TARGET], predictions),
            }
        )
    return pd.DataFrame(rows).sort_values("rmsle")


def train_validation_lightgbm(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train LightGBM on the final holdout split and return report tables."""
    log_step("preparing LightGBM validation split")
    train_rows = frame.loc[frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    train_split, valid_split = split_by_window(train_rows, window)
    features = feature_columns(frame)
    log_step(
        "LightGBM split: "
        f"train={len(train_split):,} rows, valid={len(valid_split):,} rows, "
        f"features={len(features):,}"
    )

    log_step("training LightGBM model")
    start = perf_counter()
    model = train_lightgbm_log_target(train_split, valid_split, features)
    log_step(f"LightGBM training finished in {perf_counter() - start:.1f}s")

    log_step("scoring validation predictions")
    predictions = predict_lightgbm_sales(model, valid_split, features)
    score = rmsle(valid_split[TARGET], predictions)

    metrics = pd.DataFrame(
        [
            {
                **describe_window(window),
                "model": "lightgbm_log_target",
                "valid_rows": len(valid_split),
                "feature_count": len(features),
                "rmsle": score,
            }
        ]
    )

    validation_predictions = valid_split[
        ["id", DATE_COLUMN, "store_nbr", "family", TARGET, "onpromotion"]
    ].copy()
    validation_predictions["prediction"] = predictions
    validation_predictions["absolute_error"] = (
        validation_predictions[TARGET] - validation_predictions["prediction"]
    ).abs()
    validation_predictions["squared_log_error"] = (
        np.log1p(validation_predictions["prediction"].clip(lower=0))
        - np.log1p(validation_predictions[TARGET])
    ) ** 2

    feature_importance = feature_importance_frame(model, features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_step("saving validation LightGBM model")
    model.booster_.save_model(str(MODELS_DIR / "lightgbm_validation_model.txt"))

    return {
        "metrics": metrics,
        "validation_predictions": validation_predictions,
        "feature_importance": feature_importance,
        "model": model,
        "features": features,
    }


def recursive_validation_predictions(
    base_frame: pd.DataFrame,
    model,
    features: list[str],
) -> pd.DataFrame:
    """Score validation sequentially so target lags use prior predictions."""
    train_rows = base_frame.loc[base_frame["is_train"]].copy()
    window = make_holdout_window(train_rows)

    working = base_frame.copy()
    validation_mask = working[DATE_COLUMN].between(
        window.valid_start_date, window.valid_end_date
    ) & working["is_train"]
    actuals = working.loc[
        validation_mask,
        [ID_COLUMN, DATE_COLUMN, "store_nbr", "family", TARGET, "onpromotion"],
    ].copy()

    # Hide validation targets before feature recomputation. This simulates the test
    # horizon instead of letting lag features peek at actual validation sales.
    working.loc[validation_mask, TARGET] = float("nan")
    prediction_parts = []
    validation_dates = sorted(working.loc[validation_mask, DATE_COLUMN].unique())

    for index, valid_date in enumerate(validation_dates, start=1):
        log_step(
            f"recursive validation date {index}/{len(validation_dates)}: "
            f"{pd.Timestamp(valid_date).date()}"
        )
        featured = make_features(working, verbose=False)
        day_mask = (
            working[DATE_COLUMN].eq(valid_date)
            & working["is_train"]
            & working[ID_COLUMN].isin(actuals[ID_COLUMN])
        )
        day_ids = working.loc[day_mask, ID_COLUMN]
        day_rows = featured.loc[featured[ID_COLUMN].isin(day_ids)].copy()
        predictions = predict_lightgbm_sales(model, day_rows, features)

        day_output = day_rows[
            [ID_COLUMN, DATE_COLUMN, "store_nbr", "family", "onpromotion"]
        ].copy()
        day_output["prediction"] = predictions
        prediction_parts.append(day_output)

        working.loc[day_mask, TARGET] = predictions

    predictions = pd.concat(prediction_parts, ignore_index=True)
    result = actuals.merge(
        predictions,
        on=["id", DATE_COLUMN, "store_nbr", "family", "onpromotion"],
        how="left",
        validate="one_to_one",
    )
    return result


def recursive_validation_metrics(
    base_frame: pd.DataFrame,
    model,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute an honest recursive validation score for the holdout horizon."""
    train_rows = base_frame.loc[base_frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    predictions = recursive_validation_predictions(base_frame, model, features)
    score = rmsle(predictions[TARGET], predictions["prediction"])
    predictions["absolute_error"] = (predictions[TARGET] - predictions["prediction"]).abs()
    predictions["squared_log_error"] = (
        np.log1p(predictions["prediction"].clip(lower=0))
        - np.log1p(predictions[TARGET])
    ) ** 2
    metrics = pd.DataFrame(
        [
            {
                **describe_window(window),
                "model": "lightgbm_log_target_recursive",
                "valid_rows": len(predictions),
                "feature_count": len(features),
                "rmsle": score,
            }
        ]
    )
    return metrics, predictions


def main() -> None:
    total_start = perf_counter()
    log_step("loading raw Kaggle datasets")
    datasets = load_all()
    for name, frame in datasets.items():
        log_step(f"loaded {name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    log_step("building joined modeling frame")
    base_frame = build_modeling_frame(datasets)
    log_step(
        f"joined frame ready: {len(base_frame):,} rows, "
        f"{len(base_frame.columns):,} columns"
    )

    log_step("building engineered features")
    frame = make_features(base_frame)
    log_step(f"feature frame ready: {len(frame):,} rows, {len(frame.columns):,} columns")

    baseline_results = evaluate_baselines(frame)
    lightgbm_results = train_validation_lightgbm(frame)
    log_step("running recursive validation check")
    recursive_metrics, recursive_predictions = recursive_validation_metrics(
        base_frame,
        lightgbm_results["model"],
        lightgbm_results["features"],
    )

    log_step("writing report artifacts")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = REPORTS_DIR / "baseline_validation_results.csv"
    lightgbm_metrics_path = REPORTS_DIR / "lightgbm_validation_metrics.csv"
    recursive_metrics_path = REPORTS_DIR / "lightgbm_recursive_validation_metrics.csv"
    validation_predictions_path = REPORTS_DIR / "lightgbm_validation_predictions.csv"
    recursive_predictions_path = REPORTS_DIR / "lightgbm_recursive_validation_predictions.csv"
    feature_importance_path = REPORTS_DIR / "lightgbm_feature_importance.csv"

    baseline_results.to_csv(baseline_path, index=False)
    lightgbm_results["metrics"].to_csv(lightgbm_metrics_path, index=False)
    recursive_metrics.to_csv(recursive_metrics_path, index=False)
    lightgbm_results["validation_predictions"].to_csv(
        validation_predictions_path, index=False
    )
    recursive_predictions.to_csv(recursive_predictions_path, index=False)
    lightgbm_results["feature_importance"].to_csv(feature_importance_path, index=False)

    print("Baseline validation")
    print(baseline_results.to_string(index=False))
    print("\nLightGBM validation")
    print(lightgbm_results["metrics"].to_string(index=False))
    print("\nLightGBM recursive validation")
    print(recursive_metrics.to_string(index=False))
    print(f"\nWrote baseline validation results to {baseline_path}")
    print(f"Wrote LightGBM validation metrics to {lightgbm_metrics_path}")
    print(f"Wrote recursive validation metrics to {recursive_metrics_path}")
    print(f"Wrote validation predictions to {validation_predictions_path}")
    print(f"Wrote recursive validation predictions to {recursive_predictions_path}")
    print(f"Wrote feature importance to {feature_importance_path}")
    log_step(f"training pipeline completed in {perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
