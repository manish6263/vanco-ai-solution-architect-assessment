"""Small no-data smoke tests for pipeline utilities."""

from __future__ import annotations

import pandas as pd

from .dataset import build_modeling_frame
from .features import feature_columns, make_features
from .models import BASELINE_MODELS
from .validation import make_holdout_window, split_by_window


def make_synthetic_datasets() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2017-01-01", periods=24, freq="D")
    train = pd.DataFrame(
        {
            "id": range(len(dates)),
            "date": dates,
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": [float(10 + index) for index in range(len(dates))],
            "onpromotion": 0,
        }
    )
    test = pd.DataFrame(
        {
            "id": [100, 101],
            "date": pd.to_datetime(["2017-01-25", "2017-01-26"]),
            "store_nbr": [1, 1],
            "family": ["GROCERY I", "GROCERY I"],
            "onpromotion": [0, 1],
        }
    )
    return {
        "train": train,
        "test": test,
        "stores": pd.DataFrame(
            {
                "store_nbr": [1],
                "city": ["Quito"],
                "state": ["Pichincha"],
                "type": ["D"],
                "cluster": [13],
            }
        ),
        "oil": pd.DataFrame({"date": dates, "dcoilwtico": 50.0}),
        "holidays": pd.DataFrame(
            {
                "date": [pd.Timestamp("2017-01-05")],
                "type": ["Holiday"],
                "locale": ["National"],
                "locale_name": ["Ecuador"],
                "description": ["Synthetic holiday"],
                "transferred": [False],
            }
        ),
        "transactions": pd.DataFrame(
            {"date": dates, "store_nbr": 1, "transactions": 100}
        ),
    }


def main() -> None:
    frame = build_modeling_frame(make_synthetic_datasets())
    featured = make_features(frame)
    columns = feature_columns(featured)

    assert "sales_lag_7" in featured.columns
    assert "transactions_lag_1" in featured.columns
    assert "oil_lag_1" in featured.columns
    assert "day_of_week" in columns
    assert "sales" not in columns

    train_rows = frame.loc[frame["is_train"]].copy()
    window = make_holdout_window(train_rows)
    train_split, valid_split = split_by_window(train_rows, window)

    assert len(train_split) == 8
    assert len(valid_split) == 16

    for name, predict_fn in BASELINE_MODELS.items():
        predictions = predict_fn(train_split, valid_split)
        assert len(predictions) == len(valid_split), name
        assert predictions.notna().all(), name
        assert (predictions >= 0).all(), name

    print("smoke test ok")


if __name__ == "__main__":
    main()
