"""Validate that the required Kaggle Store Sales files exist locally."""

from __future__ import annotations

from .data_loader import missing_files, summarize_raw_files


def main() -> None:
    summary = summarize_raw_files()
    print(summary.to_string(index=False))

    missing = missing_files()
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Missing required Kaggle files. Download/extract them into data/raw/:\n"
            f"{missing_list}"
        )

    print("\nAll required Kaggle files are present.")


if __name__ == "__main__":
    main()

