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


def expected_file_paths(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Return the expected CSV paths keyed by logical dataset name."""
    return {name: raw_data_dir / filename for name, filename in RAW_FILES.items()}


def missing_files(raw_data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return expected Kaggle CSV paths that are not available locally."""
    return [path for path in expected_file_paths(raw_data_dir).values() if not path.exists()]


def summarize_raw_files(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Build a small availability summary for the required raw CSV files."""
    rows = []
    for name, path in expected_file_paths(raw_data_dir).items():
        rows.append(
            {
                "dataset": name,
                "file": path.name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
                if path.exists()
                else None,
            }
        )
    return pd.DataFrame(rows)

