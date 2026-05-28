"""Time-aware validation and backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DATE_COLUMN, FORECAST_HORIZON_DAYS


@dataclass(frozen=True)
class ValidationWindow:
    """One leakage-safe validation split defined by calendar dates."""

    name: str
    train_end_date: pd.Timestamp
    valid_start_date: pd.Timestamp
    valid_end_date: pd.Timestamp


def make_holdout_window(
    train_frame: pd.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> ValidationWindow:
    """Create a final holdout window matching the Kaggle forecast horizon."""
    max_date = pd.Timestamp(train_frame[DATE_COLUMN].max())
    valid_end = max_date
    valid_start = valid_end - pd.Timedelta(days=horizon_days - 1)
    train_end = valid_start - pd.Timedelta(days=1)
    return ValidationWindow(
        name=f"holdout_last_{horizon_days}_days",
        train_end_date=train_end,
        valid_start_date=valid_start,
        valid_end_date=valid_end,
    )


def make_rolling_windows(
    train_frame: pd.DataFrame,
    n_windows: int = 3,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> list[ValidationWindow]:
    """Create rolling backtest windows, each with the same forecast horizon."""
    max_date = pd.Timestamp(train_frame[DATE_COLUMN].max())
    windows = []
    for index in range(n_windows):
        valid_end = max_date - pd.Timedelta(days=index * horizon_days)
        valid_start = valid_end - pd.Timedelta(days=horizon_days - 1)
        train_end = valid_start - pd.Timedelta(days=1)
        windows.append(
            ValidationWindow(
                name=f"backtest_{index + 1}_{horizon_days}_days",
                train_end_date=train_end,
                valid_start_date=valid_start,
                valid_end_date=valid_end,
            )
        )
    return windows


def split_by_window(
    frame: pd.DataFrame,
    window: ValidationWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into past training data and future validation data."""
    train_mask = frame[DATE_COLUMN] <= window.train_end_date
    valid_mask = frame[DATE_COLUMN].between(window.valid_start_date, window.valid_end_date)
    return frame.loc[train_mask].copy(), frame.loc[valid_mask].copy()


def describe_window(window: ValidationWindow) -> dict[str, str]:
    """Return a serializable description for reports and logs."""
    return {
        "name": window.name,
        "train_end_date": window.train_end_date.date().isoformat(),
        "valid_start_date": window.valid_start_date.date().isoformat(),
        "valid_end_date": window.valid_end_date.date().isoformat(),
    }
