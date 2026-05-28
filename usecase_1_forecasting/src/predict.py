"""Generate the Kaggle submission file for Use Case 1."""

from __future__ import annotations

import pandas as pd

from .config import DATE_COLUMN, ID_COLUMN, MODELS_DIR, SUBMISSIONS_DIR, TARGET
from .data_loader import load_all
from .dataset import build_modeling_frame
from .features import feature_columns, make_features
from .models import (
    feature_importance_frame,
    predict_lightgbm_sales,
    train_final_lightgbm_log_target,
)


def train_final_model(featured_frame: pd.DataFrame):
    """Train the final model on every labeled training row."""
    train_rows = featured_frame.loc[featured_frame["is_train"]].copy()
    features = feature_columns(featured_frame)
    model = train_final_lightgbm_log_target(train_rows, features)
    return model, features


def recursive_test_predictions(
    base_frame: pd.DataFrame,
    model,
    features: list[str],
) -> pd.DataFrame:
    """Predict test dates sequentially so lag features can use prior predictions."""
    working = base_frame.copy()
    test_dates = sorted(working.loc[~working["is_train"], DATE_COLUMN].unique())
    prediction_parts = []

    for test_date in test_dates:
        featured = make_features(working)
        day_mask = (~featured["is_train"]) & (featured[DATE_COLUMN] == test_date)
        day_rows = featured.loc[day_mask].copy()

        if day_rows.empty:
            continue

        day_predictions = predict_lightgbm_sales(model, day_rows, features)
        day_output = day_rows[[ID_COLUMN, DATE_COLUMN, "store_nbr", "family"]].copy()
        day_output[TARGET] = day_predictions
        prediction_parts.append(day_output)

        update_mask = (~working["is_train"]) & (working[DATE_COLUMN] == test_date)
        working.loc[update_mask, TARGET] = day_predictions

    if not prediction_parts:
        raise ValueError("No test rows were found for prediction.")

    return pd.concat(prediction_parts, ignore_index=True)


def build_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Create a Kaggle-compatible submission aligned to sample_submission ids."""
    submission = sample_submission[[ID_COLUMN]].merge(
        predictions[[ID_COLUMN, TARGET]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if submission[TARGET].isna().any():
        missing_count = int(submission[TARGET].isna().sum())
        raise ValueError(f"Submission has {missing_count} missing predictions.")

    submission[TARGET] = submission[TARGET].clip(lower=0)
    return submission


def main() -> None:
    datasets = load_all()
    base_frame = build_modeling_frame(datasets)
    initial_features = make_features(base_frame)
    model, features = train_final_model(initial_features)

    predictions = recursive_test_predictions(base_frame, model, features)
    submission = build_submission(predictions, datasets["sample_submission"])

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    submission_path = SUBMISSIONS_DIR / "submission_lightgbm.csv"
    predictions_path = SUBMISSIONS_DIR / "test_predictions_lightgbm_detailed.csv"
    importance_path = SUBMISSIONS_DIR / "final_model_feature_importance.csv"
    model_path = MODELS_DIR / "lightgbm_final_model.txt"

    submission.to_csv(submission_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    feature_importance_frame(model, features).to_csv(importance_path, index=False)
    model.booster_.save_model(str(model_path))

    print(f"Wrote Kaggle submission to {submission_path}")
    print(f"Wrote detailed test predictions to {predictions_path}")
    print(f"Wrote final model feature importance to {importance_path}")
    print(f"Wrote final model to {model_path}")


if __name__ == "__main__":
    main()
