"""Run the local ingestion pipeline for Use Case 3."""

from ingestion.chunking import build_chunks, load_jsonl, print_summary, write_chunks
from ingestion.parse_pdf import parse_pdf, write_jsonl
from config import PAGES_JSONL_PATH, ensure_directories


def main() -> None:
    ensure_directories()
    page_records = parse_pdf()
    write_jsonl(page_records)
    pages = load_jsonl(PAGES_JSONL_PATH)
    chunks = build_chunks(pages)
    write_chunks(chunks)
    print_summary(chunks)


if __name__ == "__main__":
    main()
