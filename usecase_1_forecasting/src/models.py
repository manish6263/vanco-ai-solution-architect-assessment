"""Baseline and machine-learning model definitions."""

from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, RANDOM_SEED, TARGET

GROUP_COLUMNS = ["store_nbr", "family"]
DEFAULT_LIGHTGBM_DEVICE = os.getenv("LGBM_DEVICE", "auto").lower()


def log_step(message: str) -> None:
    """Print a timestamped model-training progress message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [models] {message}", flush=True)


def _fallback_means(train: pd.DataFrame) -> tuple[pd.Series, float]:
    group_mean = train.groupby(GROUP_COLUMNS, observed=True)[TARGET].mean()
    global_mean = float(train[TARGET].mean())
    return group_mean, global_mean


def predict_group_mean_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Predict each store-family validation row using historical group mean."""
    group_mean, global_mean = _fallback_means(train)
    indexed = valid.set_index(GROUP_COLUMNS).index
    predictions = pd.Series(indexed.map(group_mean).to_numpy(), index=valid.index)
    return predictions.fillna(global_mean).clip(lower=0)


def predict_last_value_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Predict with the latest known sales value for each store-family pair."""
    latest = (
        train.sort_values(DATE_COLUMN)
        .groupby(GROUP_COLUMNS, observed=True)[TARGET]
        .last()
    )
    _, global_mean = _fallback_means(train)
    indexed = valid.set_index(GROUP_COLUMNS).index
    predictions = pd.Series(indexed.map(latest).to_numpy(), index=valid.index)
    return predictions.fillna(global_mean).clip(lower=0)


def predict_seasonal_naive_baseline(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seasonal_lag_days: int = 7,
) -> pd.Series:
    """Predict validation rows from same store-family sales one season ago."""
    history = train[GROUP_COLUMNS + [DATE_COLUMN, TARGET]].copy()
    lookup = history.rename(
        columns={DATE_COLUMN: "lookup_date", TARGET: "seasonal_sales"}
    )

    valid_lookup = valid[GROUP_COLUMNS + [DATE_COLUMN]].copy()
    valid_lookup["lookup_date"] = valid_lookup[DATE_COLUMN] - pd.Timedelta(
        days=seasonal_lag_days
    )
    valid_lookup["_row_id"] = valid.index

    merged = valid_lookup.merge(
        lookup,
        on=GROUP_COLUMNS + ["lookup_date"],
        how="left",
        validate="many_to_one",
    )
    predictions = merged.set_index("_row_id")["seasonal_sales"].reindex(valid.index)

    fallback = predict_group_mean_baseline(train, valid)
    return predictions.fillna(fallback).clip(lower=0)


def predict_rolling_mean_baseline(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    window_days: int = 28,
) -> pd.Series:
    """Predict with trailing store-family mean over the recent history window."""
    cutoff = valid[DATE_COLUMN].min()
    start = cutoff - pd.Timedelta(days=window_days)
    recent = train.loc[
        (train[DATE_COLUMN] >= start) & (train[DATE_COLUMN] < cutoff)
    ].copy()

    if recent.empty:
        return predict_group_mean_baseline(train, valid)

    recent_mean = recent.groupby(GROUP_COLUMNS, observed=True)[TARGET].mean()
    group_mean, global_mean = _fallback_means(train)

    indexed = valid.set_index(GROUP_COLUMNS).index
    predictions = pd.Series(indexed.map(recent_mean).to_numpy(), index=valid.index)
    fallback = pd.Series(indexed.map(group_mean).to_numpy(), index=valid.index)
    return predictions.fillna(fallback).fillna(global_mean).clip(lower=0)


BASELINE_MODELS = {
    "group_mean": predict_group_mean_baseline,
    "last_value": predict_last_value_baseline,
    "seasonal_naive_7": predict_seasonal_naive_baseline,
    "rolling_mean_28": predict_rolling_mean_baseline,
}


def clip_predictions(values) -> np.ndarray:
    """Force predictions into the valid non-negative sales range."""
    return np.maximum(np.asarray(values, dtype=float), 0)


def make_lightgbm_regressor(device_type: str = "cpu"):
    """Create the primary LightGBM model used for tabular forecasting."""
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM is required for model training. Install dependencies with "
            "`pip install -r requirements.txt` from usecase_1_forecasting/."
        ) from exc

    params = {
        "objective": "regression",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    if device_type == "gpu":
        params.update(
            {
                "device_type": "gpu",
                "max_bin": 63,
                "gpu_use_dp": False,
            }
        )

    return LGBMRegressor(**params)


def lightgbm_device_candidates() -> list[str]:
    """Return device candidates based on LGBM_DEVICE."""
    if DEFAULT_LIGHTGBM_DEVICE in {"cpu", "gpu"}:
        return [DEFAULT_LIGHTGBM_DEVICE]
    if DEFAULT_LIGHTGBM_DEVICE != "auto":
        raise ValueError("LGBM_DEVICE must be one of: auto, cpu, gpu")
    return ["gpu", "cpu"]


def fit_with_device_fallback(fit_fn):
    """Try requested LightGBM devices and fall back from GPU to CPU in auto mode."""
    candidates = lightgbm_device_candidates()
    last_error = None

    for device_type in candidates:
        try:
            log_step(f"trying LightGBM device_type={device_type}")
            model = make_lightgbm_regressor(device_type=device_type)
            fit_fn(model)
            log_step(f"LightGBM fitted with device_type={device_type}")
            return model
        except Exception as exc:
            last_error = exc
            if device_type == "gpu" and candidates[-1] == "cpu":
                log_step(
                    "GPU LightGBM failed; falling back to CPU. "
                    f"Reason: {type(exc).__name__}: {exc}"
                )
                continue
            raise

    raise RuntimeError("Unable to fit LightGBM model") from last_error


def train_lightgbm_log_target(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
):
    """Train LightGBM on log1p(sales) and evaluate against validation rows."""
    categorical_features = [
        column
        for column in feature_names
        if str(train[column].dtype) == "category"
    ]

    try:
        from lightgbm import early_stopping, log_evaluation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM is required for model training. Install dependencies with "
            "`pip install -r requirements.txt` from usecase_1_forecasting/."
        ) from exc

    def fit_fn(model) -> None:
        model.fit(
            train[feature_names],
            np.log1p(train[TARGET]),
            eval_set=[(valid[feature_names], np.log1p(valid[TARGET]))],
            eval_metric="rmse",
            categorical_feature=categorical_features or "auto",
            callbacks=[early_stopping(100), log_evaluation(100)],
        )

    return fit_with_device_fallback(fit_fn)


def train_final_lightgbm_log_target(
    train: pd.DataFrame,
    feature_names: list[str],
):
    """Train the final LightGBM model on all labeled training rows."""
    categorical_features = [
        column
        for column in feature_names
        if str(train[column].dtype) == "category"
    ]

    def fit_fn(model) -> None:
        model.fit(
            train[feature_names],
            np.log1p(train[TARGET]),
            categorical_feature=categorical_features or "auto",
        )

    return fit_with_device_fallback(fit_fn)


def predict_lightgbm_sales(model, frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Predict non-negative sales from a LightGBM log-target model."""
    predictions = np.expm1(model.predict(frame[feature_names]))
    return clip_predictions(predictions)


def feature_importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    """Return split/gain feature importance from the trained LightGBM model."""
    booster = model.booster_
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_split": booster.feature_importance(importance_type="split"),
                "importance_gain": booster.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )
