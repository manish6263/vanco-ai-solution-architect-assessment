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

GROUP_COLUMNS = ["store_nbr", "family"]


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


def build_postprocess_stats(train: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """Build recent sales guardrails for each store-family pair."""
    max_date = train[DATE_COLUMN].max()
    recent_start = max_date - pd.Timedelta(days=lookback_days - 1)
    recent = train.loc[train[DATE_COLUMN] >= recent_start].copy()

    stats = (
        recent.groupby(GROUP_COLUMNS, observed=True)[TARGET]
        .agg(
            recent_sum="sum",
            recent_mean="mean",
            recent_median="median",
            recent_max="max",
            recent_nonzero_days=lambda values: int((values > 0).sum()),
        )
        .reset_index()
    )

    long_stats = (
        train.groupby(GROUP_COLUMNS, observed=True)[TARGET]
        .agg(all_time_mean="mean", all_time_max="max")
        .reset_index()
    )

    stats = stats.merge(long_stats, on=GROUP_COLUMNS, how="outer")
    stats["recent_sum"] = stats["recent_sum"].fillna(0)
    stats["recent_nonzero_days"] = stats["recent_nonzero_days"].fillna(0)
    stats["recent_max"] = stats["recent_max"].fillna(0)
    stats["recent_mean"] = stats["recent_mean"].fillna(0)
    stats["recent_median"] = stats["recent_median"].fillna(0)
    stats["all_time_mean"] = stats["all_time_mean"].fillna(0)
    stats["all_time_max"] = stats["all_time_max"].fillna(0)

    # Conservative cap: high enough to allow promotions/spikes, low enough to stop
    # pathological predictions on low/zero-volume series.
    stats["prediction_cap"] = (
        pd.concat(
            [
                stats["recent_max"] * 1.75,
                stats["recent_mean"] * 6.0,
                stats["all_time_mean"] * 4.0,
            ],
            axis=1,
        )
        .max(axis=1)
        .clip(lower=0)
    )
    stats["force_zero"] = (
        (stats["recent_sum"] == 0)
        & (stats["recent_nonzero_days"] == 0)
        & (stats["all_time_max"] == 0)
    )
    stats["inactive_recently"] = (
        (stats["recent_sum"] == 0) & (stats["recent_nonzero_days"] == 0)
    )
    return stats


def postprocess_predictions(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply zero-sales and cap guardrails to raw model predictions."""
    stats = build_postprocess_stats(train)
    result = predictions.merge(stats, on=GROUP_COLUMNS, how="left", validate="many_to_one")
    result["raw_sales"] = result[TARGET]

    zero_mask = result["force_zero"].fillna(False)
    inactive_mask = result["inactive_recently"].fillna(False) & (result["onpromotion"] <= 0)
    result.loc[zero_mask | inactive_mask, TARGET] = 0

    cap = result["prediction_cap"].fillna(result[TARGET])
    positive_cap = cap > 0
    result.loc[positive_cap, TARGET] = result.loc[positive_cap, TARGET].clip(
        upper=cap.loc[positive_cap]
    )
    result[TARGET] = result[TARGET].clip(lower=0)

    diagnostics = pd.DataFrame(
        [
            {
                "rows": len(result),
                "raw_min": predictions[TARGET].min(),
                "raw_p50": predictions[TARGET].median(),
                "raw_p95": predictions[TARGET].quantile(0.95),
                "raw_p99": predictions[TARGET].quantile(0.99),
                "raw_max": predictions[TARGET].max(),
                "post_min": result[TARGET].min(),
                "post_p50": result[TARGET].median(),
                "post_p95": result[TARGET].quantile(0.95),
                "post_p99": result[TARGET].quantile(0.99),
                "post_max": result[TARGET].max(),
                "zeroed_rows": int((result[TARGET] == 0).sum()),
                "changed_rows": int((result[TARGET] != result["raw_sales"]).sum()),
            }
        ]
    )

    keep_columns = [ID_COLUMN, DATE_COLUMN, "store_nbr", "family", TARGET, "raw_sales"]
    return result[keep_columns], diagnostics


def main() -> None:
    datasets = load_all()
    base_frame = build_modeling_frame(datasets)
    initial_features = make_features(base_frame)
    model, features = train_final_model(initial_features)

    predictions = recursive_test_predictions(base_frame, model, features)
    submission = build_submission(predictions, datasets["sample_submission"])
    postprocessed_predictions, postprocess_diagnostics = postprocess_predictions(
        predictions, datasets["train"]
    )
    postprocessed_submission = build_submission(
        postprocessed_predictions, datasets["sample_submission"]
    )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    submission_path = SUBMISSIONS_DIR / "submission_lightgbm.csv"
    postprocessed_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_postprocessed.csv"
    predictions_path = SUBMISSIONS_DIR / "test_predictions_lightgbm_detailed.csv"
    postprocessed_predictions_path = (
        SUBMISSIONS_DIR / "test_predictions_lightgbm_postprocessed_detailed.csv"
    )
    diagnostics_path = SUBMISSIONS_DIR / "submission_diagnostics.csv"
    importance_path = SUBMISSIONS_DIR / "final_model_feature_importance.csv"
    model_path = MODELS_DIR / "lightgbm_final_model.txt"

    submission.to_csv(submission_path, index=False)
    postprocessed_submission.to_csv(postprocessed_submission_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    postprocessed_predictions.to_csv(postprocessed_predictions_path, index=False)
    postprocess_diagnostics.to_csv(diagnostics_path, index=False)
    feature_importance_frame(model, features).to_csv(importance_path, index=False)
    model.booster_.save_model(str(model_path))

    print(f"Wrote Kaggle submission to {submission_path}")
    print(f"Wrote postprocessed Kaggle submission to {postprocessed_submission_path}")
    print(f"Wrote detailed test predictions to {predictions_path}")
    print(f"Wrote postprocessed detailed test predictions to {postprocessed_predictions_path}")
    print(f"Wrote submission diagnostics to {diagnostics_path}")
    print(f"Wrote final model feature importance to {importance_path}")
    print(f"Wrote final model to {model_path}")


if __name__ == "__main__":
    main()
