"""Parse the NCERT Physics PDF into page-level JSONL records."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from config import PAGES_JSONL_PATH, PDF_PATH, ensure_directories


CHAPTER_PATTERN = re.compile(r"^\s*(chapter\s+\d+|chapter[-\s]*\d+)\b", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)+\s+[A-Z][A-Za-z0-9 ,;:'()/-]+$")
NUMBERED_HEADING_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,;:'()/-]+$")
FORMULA_PATTERN = re.compile(
    r"[=\u2248\u221d\u00b1\u00d7\u00f7\u221a\u03a3\u222b]"
    r"|(?:\b[A-Za-z]\s*=\s*)"
    r"|(?:\b\d+\s*[A-Za-z]?\s*/\s*[A-Za-z])"
)


@dataclass
class PageRecord:
    document: str
    page_number: int
    text: str
    headings: list[str]
    chapter: str | None
    section: str | None
    formula_candidates: list[str]
    char_count: int
    word_count: int


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def iter_clean_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        line = normalize_line(line)
        if line:
            yield line


def looks_like_heading(line: str) -> bool:
    if len(line) < 4 or len(line) > 120:
        return False
    if CHAPTER_PATTERN.match(line):
        return True
    if SECTION_PATTERN.match(line):
        return True
    if NUMBERED_HEADING_PATTERN.match(line) and line.upper() == line:
        return True
    if line.isupper() and any(character.isalpha() for character in line):
        return True
    return False


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in iter_clean_lines(text):
        if looks_like_heading(line) and line not in headings:
            headings.append(line)
    return headings


def update_context(
    headings: list[str],
    current_chapter: str | None,
    current_section: str | None,
) -> tuple[str | None, str | None]:
    for heading in headings:
        if CHAPTER_PATTERN.match(heading):
            current_chapter = heading
            current_section = None
        elif SECTION_PATTERN.match(heading) or NUMBERED_HEADING_PATTERN.match(heading):
            current_section = heading
    return current_chapter, current_section


def extract_formula_candidates(text: str) -> list[str]:
    formulas: list[str] = []
    for line in iter_clean_lines(text):
        if 6 <= len(line) <= 180 and FORMULA_PATTERN.search(line):
            formulas.append(line)
    return formulas[:20]


def parse_pdf() -> list[PageRecord]:
    if not PDF_PATH.exists():
        raise SystemExit(
            "Source PDF is missing. Run `python -m ingestion.download_pdf` "
            "or place it at data/raw/ncert_physics_part1.pdf."
        )

    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF parsing. "
            "Run `pip install -r requirements.txt` from usecase_3_hybrid_rag."
        ) from exc

    ensure_directories()
    records: list[PageRecord] = []
    current_chapter: str | None = None
    current_section: str | None = None

    with fitz.open(PDF_PATH) as document:
        print(f"Parsing PDF: {PDF_PATH}")
        print(f"Detected pages: {document.page_count}")
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text", sort=True)
            text = clean_text(raw_text)
            headings = extract_headings(text)
            current_chapter, current_section = update_context(
                headings,
                current_chapter,
                current_section,
            )
            records.append(
                PageRecord(
                    document=PDF_PATH.name,
                    page_number=index,
                    text=text,
                    headings=headings,
                    chapter=current_chapter,
                    section=current_section,
                    formula_candidates=extract_formula_candidates(text),
                    char_count=len(text),
                    word_count=len(text.split()),
                )
            )
            if index == 1 or index % 25 == 0 or index == document.page_count:
                print(f"Parsed page {index}/{document.page_count}")

    return records


def write_jsonl(records: list[PageRecord]) -> None:
    PAGES_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAGES_JSONL_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    print(f"Wrote page records: {PAGES_JSONL_PATH}")
    print(f"Records: {len(records)}")


def main() -> None:
    records = parse_pdf()
    write_jsonl(records)


if __name__ == "__main__":
    main()
