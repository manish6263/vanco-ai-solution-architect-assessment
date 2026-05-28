"""Data loading helpers for the Kaggle Store Sales dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DATA_DIR, RAW_FILES


def read_csv(name: str, raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read one expected Kaggle CSV by logical name."""
    if name not in RAW_FILES:
        expected = ", ".join(sorted(RAW_FILES))
        raise KeyError(f"Unknown dataset '{name}'. Expected one of: {expected}")

    path = raw_data_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Kaggle competition files into data/raw/."
        )

    return pd.read_csv(path)


def load_all(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all expected Store Sales CSV files."""
    return {name: read_csv(name, raw_data_dir) for name in RAW_FILES}

