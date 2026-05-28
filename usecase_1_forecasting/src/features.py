"""Feature engineering for leakage-safe grocery sales forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, PROCESSED_DATA_DIR, TARGET
from .data_loader import load_all
from .dataset import build_modeling_frame

GROUP_COLUMNS = ["store_nbr", "family"]
STORE_COLUMNS = ["store_nbr"]

SALES_LAGS = [7, 14, 28]
SALES_ROLLING_WINDOWS = [7, 14, 28]
PROMOTION_LAGS = [1, 7, 14]
PROMOTION_ROLLING_WINDOWS = [7, 14]
TRANSACTION_LAGS = [1, 7, 14]
TRANSACTION_ROLLING_WINDOWS = [7, 14, 28]
OIL_LAGS = [1, 7, 14]
OIL_ROLLING_WINDOWS = [7, 14, 28]

CATEGORICAL_FEATURES = [
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "national_holiday_type",
    "regional_holiday_type",
    "local_holiday_type",
]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic date-derived features known for train and test."""
    result = frame.copy()
    date = result[DATE_COLUMN]
    iso_calendar = date.dt.isocalendar()

    result["day_of_week"] = date.dt.dayofweek
    result["day_of_month"] = date.dt.day
    result["day_of_year"] = date.dt.dayofyear
    result["week_of_year"] = iso_calendar.week.astype("int16")
    result["month"] = date.dt.month
    result["quarter"] = date.dt.quarter
    result["year"] = date.dt.year
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("int8")
    result["is_month_start"] = date.dt.is_month_start.astype("int8")
    result["is_month_end"] = date.dt.is_month_end.astype("int8")
    result["is_payday_window"] = (
        result["day_of_month"].isin([15, 16, 30, 31, 1])
    ).astype("int8")

    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    return result


def add_oil_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add date-level oil price lag, rolling, and change features."""
    result = frame.copy()
    oil_by_date = (
        result[[DATE_COLUMN, "dcoilwtico"]]
        .drop_duplicates(DATE_COLUMN)
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    for lag in OIL_LAGS:
        oil_by_date[f"oil_lag_{lag}"] = oil_by_date["dcoilwtico"].shift(lag)

    shifted_oil = oil_by_date["dcoilwtico"].shift(1)
    for window in OIL_ROLLING_WINDOWS:
        oil_by_date[f"oil_roll_mean_{window}"] = shifted_oil.rolling(
            window, min_periods=1
        ).mean()
        oil_by_date[f"oil_roll_std_{window}"] = shifted_oil.rolling(
            window, min_periods=2
        ).std()

    oil_by_date["oil_pct_change_1"] = oil_by_date["dcoilwtico"].pct_change(
        fill_method=None
    )
    oil_feature_columns = [
        col for col in oil_by_date.columns if col != "dcoilwtico"
    ]
    result = result.merge(
        oil_by_date[oil_feature_columns],
        on=DATE_COLUMN,
        how="left",
        validate="many_to_one",
    )
    return result


def add_promotion_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add promotion intensity features; promotion is known in train and test."""
    result = frame.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).copy()
    result["onpromotion"] = result["onpromotion"].fillna(0)

    group = result.groupby(GROUP_COLUMNS, observed=True)["onpromotion"]
    for lag in PROMOTION_LAGS:
        result[f"promo_lag_{lag}"] = group.shift(lag)

    shifted = group.shift(1)
    for window in PROMOTION_ROLLING_WINDOWS:
        result[f"promo_roll_mean_{window}"] = shifted.groupby(
            [result["store_nbr"], result["family"]], observed=True
        ).rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    result["promo_store_family_mean_history"] = group.transform(
        lambda values: values.expanding().mean().shift(1)
    )
    result["has_promotion"] = (result["onpromotion"] > 0).astype("int8")
    return result


def add_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add store-level transaction features from prior known transaction history."""
    result = frame.copy()
    store_dates = (
        result[[DATE_COLUMN, "store_nbr", "transactions"]]
        .drop_duplicates([DATE_COLUMN, "store_nbr"])
        .sort_values(["store_nbr", DATE_COLUMN])
        .reset_index(drop=True)
    )

    group = store_dates.groupby(STORE_COLUMNS, observed=True)["transactions"]
    for lag in TRANSACTION_LAGS:
        store_dates[f"transactions_lag_{lag}"] = group.shift(lag)

    shifted = group.shift(1)
    for window in TRANSACTION_ROLLING_WINDOWS:
        store_dates[f"transactions_roll_mean_{window}"] = shifted.groupby(
            store_dates["store_nbr"], observed=True
        ).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

    feature_columns = [
        col
        for col in store_dates.columns
        if col not in {"transactions"}
    ]
    result = result.merge(
        store_dates[feature_columns],
        on=[DATE_COLUMN, "store_nbr"],
        how="left",
        validate="many_to_one",
    )
    return result


def add_sales_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe store-family sales lag and rolling target features."""
    result = frame.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).copy()
    group = result.groupby(GROUP_COLUMNS, observed=True)[TARGET]

    for lag in SALES_LAGS:
        result[f"sales_lag_{lag}"] = group.shift(lag)

    shifted_sales = group.shift(1)
    for window in SALES_ROLLING_WINDOWS:
        rolled = shifted_sales.groupby(
            [result["store_nbr"], result["family"]], observed=True
        ).rolling(window, min_periods=1)
        result[f"sales_roll_mean_{window}"] = rolled.mean().reset_index(
            level=[0, 1], drop=True
        )
        result[f"sales_roll_std_{window}"] = rolled.std().reset_index(
            level=[0, 1], drop=True
        )

    result["sales_group_mean_history"] = group.transform(
        lambda values: values.expanding().mean().shift(1)
    )
    return result


def encode_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas category dtype for tree models."""
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        if column in result.columns:
            result[column] = result[column].fillna("unknown").astype("category")
    return result


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature table from the joined modeling frame."""
    result = add_calendar_features(frame)
    result = add_oil_features(result)
    result = add_promotion_features(result)
    result = add_transaction_features(result)
    result = add_sales_history_features(result)
    result = encode_categoricals(result)
    return result.sort_values(["store_nbr", "family", DATE_COLUMN]).reset_index(
        drop=True
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model input columns, excluding identifiers and target leakage fields."""
    excluded = {
        "id",
        DATE_COLUMN,
        TARGET,
        "is_train",
        "transactions",
        "national_holiday_description",
        "regional_holiday_description",
        "local_holiday_description",
    }
    return [col for col in frame.columns if col not in excluded]


def write_feature_frame(frame: pd.DataFrame) -> None:
    """Persist the engineered feature table for training experiments."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)


def main() -> None:
    datasets = load_all()
    base_frame = build_modeling_frame(datasets)
    featured = make_features(base_frame)
    write_feature_frame(featured)
    print(f"Wrote {len(featured):,} rows to {PROCESSED_DATA_DIR / 'features.parquet'}")
    print(f"Feature columns: {len(feature_columns(featured))}")


if __name__ == "__main__":
    main()
