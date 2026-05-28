"""Train and evaluate forecasting models for Use Case 1."""

from __future__ import annotations

import pandas as pd

from .config import REPORTS_DIR, TARGET
from .data_loader import load_all
from .dataset import build_modeling_frame
from .metrics import rmsle
from .models import BASELINE_MODELS
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


def main() -> None:
    datasets = load_all()
    frame = build_modeling_frame(datasets)
    results = evaluate_baselines(frame)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "baseline_validation_results.csv"
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nWrote baseline validation results to {output_path}")


if __name__ == "__main__":
    main()
