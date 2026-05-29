"""Validate that the source PDF is available before ingestion."""

from __future__ import annotations

from config import PDF_PATH, PDF_URL, ensure_directories


def main() -> None:
    ensure_directories()
    if not PDF_PATH.exists():
        raise SystemExit(
            "Source PDF is missing.\n"
            f"Download: {PDF_URL}\n"
            f"Save as:  {PDF_PATH}\n"
            "Or run:   python -m ingestion.download_pdf"
        )

    size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    with PDF_PATH.open("rb") as file:
        signature = file.read(4)
    if signature != b"%PDF":
        raise SystemExit(f"File exists but is not a valid PDF: {PDF_PATH}")

    print(f"Source PDF found: {PDF_PATH}")
    print(f"Size: {size_mb:.1f} MB")
    print("PDF setup is ready for parsing.")


if __name__ == "__main__":
    main()
