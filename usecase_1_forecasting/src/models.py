"""Baseline and machine-learning model definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, TARGET

GROUP_COLUMNS = ["store_nbr", "family"]


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
