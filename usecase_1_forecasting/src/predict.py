"""Generate the Kaggle submission file for Use Case 1."""

from __future__ import annotations

from datetime import datetime

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


def log_step(message: str) -> None:
    """Print a timestamped prediction progress message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [predict] {message}", flush=True)


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

    for index, test_date in enumerate(test_dates, start=1):
        log_step(f"predicting test date {index}/{len(test_dates)}: {pd.Timestamp(test_date).date()}")
        featured = make_features(working, verbose=False)
        day_mask = (~featured["is_train"]) & (featured[DATE_COLUMN] == test_date)
        day_rows = featured.loc[day_mask].copy()

        if day_rows.empty:
            continue

        day_predictions = predict_lightgbm_sales(model, day_rows, features)
        day_output = day_rows[
            [ID_COLUMN, DATE_COLUMN, "store_nbr", "family", "onpromotion"]
        ].copy()
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
            recent_q90=lambda values: values.quantile(0.90),
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
    stats["recent_q90"] = stats["recent_q90"].fillna(0)
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


def apply_prediction_guardrails(
    predictions: pd.DataFrame,
    stats: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """Apply one guardrail policy to raw model predictions."""
    result = predictions.merge(stats, on=GROUP_COLUMNS, how="left", validate="many_to_one")
    result["raw_sales"] = result[TARGET]

    if variant == "mild":
        cap = result["prediction_cap"].fillna(result[TARGET])
        inactive_mask = result["inactive_recently"].fillna(False) & (
            result["onpromotion"] <= 0
        )
    elif variant == "guarded":
        cap = (
            pd.concat(
                [
                    result["recent_q90"] * 2.0,
                    result["recent_mean"] * 4.0,
                    result["all_time_mean"] * 2.5,
                    result["recent_max"] * 1.15,
                ],
                axis=1,
            )
            .max(axis=1)
            .fillna(result[TARGET])
        )
        inactive_mask = (result["recent_nonzero_days"].fillna(0) <= 2) & (
            result["onpromotion"] <= 0
        )
    elif variant in {"blend_recent", "blend_recent_60", "blend_recent_50", "blend_recent_40"}:
        cap = (
            pd.concat(
                [
                    result["recent_q90"] * 2.5,
                    result["recent_mean"] * 5.0,
                    result["recent_max"] * 1.35,
                ],
                axis=1,
            )
            .max(axis=1)
            .fillna(result[TARGET])
        )
        recent_anchor = result["recent_mean"].fillna(result["all_time_mean"]).fillna(0)
        blend_weights = {
            "blend_recent": 0.70,
            "blend_recent_60": 0.60,
            "blend_recent_50": 0.50,
            "blend_recent_40": 0.40,
        }
        blend_weight = blend_weights[variant]
        promo_mask = result["onpromotion"] > 0
        result.loc[~promo_mask, TARGET] = (
            blend_weight * result.loc[~promo_mask, TARGET]
            + (1 - blend_weight) * recent_anchor.loc[~promo_mask]
        )
        inactive_mask = (result["recent_nonzero_days"].fillna(0) <= 1) & (
            result["onpromotion"] <= 0
        )
    else:
        raise ValueError(f"Unknown postprocess variant: {variant}")

    zero_mask = result["force_zero"].fillna(False)
    result.loc[zero_mask | inactive_mask, TARGET] = 0

    positive_cap = cap > 0
    result.loc[positive_cap, TARGET] = result.loc[positive_cap, TARGET].clip(
        upper=cap.loc[positive_cap]
    )
    result[TARGET] = result[TARGET].clip(lower=0)
    result["variant"] = variant
    return result


def prediction_diagnostics(result: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Summarize one prediction variant."""
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "rows": len(result),
                "raw_min": result["raw_sales"].min(),
                "raw_p50": result["raw_sales"].median(),
                "raw_p95": result["raw_sales"].quantile(0.95),
                "raw_p99": result["raw_sales"].quantile(0.99),
                "raw_max": result["raw_sales"].max(),
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


def postprocess_prediction_variants(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate several Kaggle candidate prediction variants."""
    stats = build_postprocess_stats(train)
    variants = {}
    diagnostics = []

    for variant in [
        "mild",
        "guarded",
        "blend_recent",
        "blend_recent_60",
        "blend_recent_50",
        "blend_recent_40",
    ]:
        result = apply_prediction_guardrails(predictions, stats, variant)
        keep_columns = [
            ID_COLUMN,
            DATE_COLUMN,
            "store_nbr",
            "family",
            "onpromotion",
            TARGET,
            "raw_sales",
            "variant",
        ]
        variants[variant] = result[keep_columns]
        diagnostics.append(prediction_diagnostics(result, variant))

    return variants, pd.concat(diagnostics, ignore_index=True)


def postprocess_predictions(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible mild postprocessing helper."""
    variants, diagnostics = postprocess_prediction_variants(predictions, train)
    return variants["mild"], diagnostics.loc[diagnostics["variant"] == "mild"].copy()


def main() -> None:
    datasets = load_all()
    base_frame = build_modeling_frame(datasets)
    initial_features = make_features(base_frame)
    model, features = train_final_model(initial_features)

    predictions = recursive_test_predictions(base_frame, model, features)
    submission = build_submission(predictions, datasets["sample_submission"])
    variant_predictions, postprocess_diagnostics = postprocess_prediction_variants(
        predictions, datasets["train"]
    )
    postprocessed_predictions = variant_predictions["mild"]
    postprocessed_submission = build_submission(
        postprocessed_predictions, datasets["sample_submission"]
    )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    submission_path = SUBMISSIONS_DIR / "submission_lightgbm.csv"
    postprocessed_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_postprocessed.csv"
    guarded_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_guarded.csv"
    blend_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_blend_recent.csv"
    blend_60_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_blend_recent_60.csv"
    blend_50_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_blend_recent_50.csv"
    blend_40_submission_path = SUBMISSIONS_DIR / "submission_lightgbm_blend_recent_40.csv"
    predictions_path = SUBMISSIONS_DIR / "test_predictions_lightgbm_detailed.csv"
    postprocessed_predictions_path = (
        SUBMISSIONS_DIR / "test_predictions_lightgbm_postprocessed_detailed.csv"
    )
    diagnostics_path = SUBMISSIONS_DIR / "submission_diagnostics.csv"
    importance_path = SUBMISSIONS_DIR / "final_model_feature_importance.csv"
    model_path = MODELS_DIR / "lightgbm_final_model.txt"

    submission.to_csv(submission_path, index=False)
    postprocessed_submission.to_csv(postprocessed_submission_path, index=False)
    build_submission(
        variant_predictions["guarded"], datasets["sample_submission"]
    ).to_csv(guarded_submission_path, index=False)
    build_submission(
        variant_predictions["blend_recent"], datasets["sample_submission"]
    ).to_csv(blend_submission_path, index=False)
    build_submission(
        variant_predictions["blend_recent_60"], datasets["sample_submission"]
    ).to_csv(blend_60_submission_path, index=False)
    build_submission(
        variant_predictions["blend_recent_50"], datasets["sample_submission"]
    ).to_csv(blend_50_submission_path, index=False)
    build_submission(
        variant_predictions["blend_recent_40"], datasets["sample_submission"]
    ).to_csv(blend_40_submission_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    postprocessed_predictions.to_csv(postprocessed_predictions_path, index=False)
    postprocess_diagnostics.to_csv(diagnostics_path, index=False)
    feature_importance_frame(model, features).to_csv(importance_path, index=False)
    model.booster_.save_model(str(model_path))

    print(f"Wrote Kaggle submission to {submission_path}")
    print(f"Wrote postprocessed Kaggle submission to {postprocessed_submission_path}")
    print(f"Wrote guarded Kaggle submission to {guarded_submission_path}")
    print(f"Wrote blend-recent Kaggle submission to {blend_submission_path}")
    print(f"Wrote blend-recent-60 Kaggle submission to {blend_60_submission_path}")
    print(f"Wrote blend-recent-50 Kaggle submission to {blend_50_submission_path}")
    print(f"Wrote blend-recent-40 Kaggle submission to {blend_40_submission_path}")
    print(f"Wrote detailed test predictions to {predictions_path}")
    print(f"Wrote postprocessed detailed test predictions to {postprocessed_predictions_path}")
    print(f"Wrote submission diagnostics to {diagnostics_path}")
    print(f"Wrote final model feature importance to {importance_path}")
    print(f"Wrote final model to {model_path}")


if __name__ == "__main__":
    main()
