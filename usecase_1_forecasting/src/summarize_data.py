"""Print initial raw-data and joined-frame summaries for Use Case 1."""

from __future__ import annotations

from .data_loader import load_all, summarize_loaded_frames
from .dataset import build_modeling_frame


def main() -> None:
    datasets = load_all()
    print("Raw datasets")
    print(summarize_loaded_frames(datasets).to_string(index=False))

    frame = build_modeling_frame(datasets)
    print("\nJoined modeling frame")
    print(f"rows: {len(frame):,}")
    print(f"columns: {len(frame.columns):,}")
    print(f"date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"train rows: {frame['is_train'].sum():,}")
    print(f"test rows: {(~frame['is_train']).sum():,}")
    print("\nColumns")
    print("\n".join(frame.columns))


if __name__ == "__main__":
    main()
