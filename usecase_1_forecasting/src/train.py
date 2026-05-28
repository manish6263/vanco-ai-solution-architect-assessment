"""Train and evaluate forecasting models for Use Case 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, MODELS_DIR, REPORTS_DIR, TARGET
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


def evaluate_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate baseline forecasts on the final time-aware validation window."""
    train_rows = frame.loc[frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    train_split, valid_split = split_by_window(train_rows, window)

    rows = []
    for model_name, predict_fn in BASELINE_MODELS.items():
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
    train_rows = frame.loc[frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    train_split, valid_split = split_by_window(train_rows, window)
    features = feature_columns(frame)

    model = train_lightgbm_log_target(train_split, valid_split, features)
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
    model.booster_.save_model(str(MODELS_DIR / "lightgbm_validation_model.txt"))

    return {
        "metrics": metrics,
        "validation_predictions": validation_predictions,
        "feature_importance": feature_importance,
    }


def main() -> None:
    datasets = load_all()
    base_frame = build_modeling_frame(datasets)
    frame = make_features(base_frame)
    baseline_results = evaluate_baselines(frame)
    lightgbm_results = train_validation_lightgbm(frame)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = REPORTS_DIR / "baseline_validation_results.csv"
    lightgbm_metrics_path = REPORTS_DIR / "lightgbm_validation_metrics.csv"
    validation_predictions_path = REPORTS_DIR / "lightgbm_validation_predictions.csv"
    feature_importance_path = REPORTS_DIR / "lightgbm_feature_importance.csv"

    baseline_results.to_csv(baseline_path, index=False)
    lightgbm_results["metrics"].to_csv(lightgbm_metrics_path, index=False)
    lightgbm_results["validation_predictions"].to_csv(
        validation_predictions_path, index=False
    )
    lightgbm_results["feature_importance"].to_csv(feature_importance_path, index=False)

    print("Baseline validation")
    print(baseline_results.to_string(index=False))
    print("\nLightGBM validation")
    print(lightgbm_results["metrics"].to_string(index=False))
    print(f"\nWrote baseline validation results to {baseline_path}")
    print(f"Wrote LightGBM validation metrics to {lightgbm_metrics_path}")
    print(f"Wrote validation predictions to {validation_predictions_path}")
    print(f"Wrote feature importance to {feature_importance_path}")


if __name__ == "__main__":
    main()
