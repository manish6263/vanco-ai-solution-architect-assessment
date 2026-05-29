"""Download the source NCERT Physics PDF for local ingestion."""

from __future__ import annotations

import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import PDF_PATH, PDF_URL, ensure_directories


def download_pdf(force: bool = False) -> None:
    ensure_directories()
    if PDF_PATH.exists() and not force:
        print(f"PDF already exists: {PDF_PATH}")
        print("Use --force to re-download it.")
        return

    print(f"Downloading PDF from: {PDF_URL}")
    request = Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(
            "Could not download the PDF automatically. "
            f"Download it manually from {PDF_URL} and save it as {PDF_PATH}.\n"
            f"Original error: {exc}"
        ) from exc

    if not payload.startswith(b"%PDF"):
        raise SystemExit("Downloaded file does not look like a PDF. Please download it manually.")

    PDF_PATH.write_bytes(payload)
    print(f"Wrote PDF: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size / (1024 * 1024):.1f} MB")


def main() -> None:
    download_pdf(force="--force" in sys.argv)


if __name__ == "__main__":
    main()
