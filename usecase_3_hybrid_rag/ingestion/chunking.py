"""Page-aware and heading-aware chunking utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from config import CHUNKS_JSONL_PATH, PAGES_JSONL_PATH, ensure_directories


DEFAULT_MAX_CHARS = 1600
DEFAULT_OVERLAP_CHARS = 250
MIN_CHUNK_CHARS = 180
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)+\s+\S+")


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    page_number: int
    chunk_index: int
    text: str
    search_text: str
    citation: str
    chapter: str | None
    section: str | None
    headings: list[str]
    formula_candidates: list[str]
    content_type: str
    char_count: int
    word_count: int


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing input file: {path}\n"
            "Run `python -m ingestion.parse_pdf` before chunking."
        )

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_paragraphs(text: str) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_spacing(paragraph)
        if paragraph:
            yield paragraph


def chunk_paragraphs(
    paragraphs: list[str],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_len = 0
            chunks.extend(split_long_text(paragraph, max_chars, overlap_chars))
            continue

        projected_len = current_len + len(paragraph) + (2 if current_parts else 0)
        if current_parts and projected_len > max_chars:
            chunks.append("\n\n".join(current_parts).strip())
            overlap = build_overlap(current_parts, overlap_chars)
            current_parts = [overlap] if overlap else []
            current_len = len(overlap)

        current_parts.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_CHARS]


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if sentence_boundary > start + max_chars // 2:
                end = sentence_boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def build_overlap(parts: list[str], overlap_chars: int) -> str:
    overlap = ""
    for part in reversed(parts):
        candidate = f"{part}\n\n{overlap}".strip() if overlap else part
        if len(candidate) > overlap_chars:
            break
        overlap = candidate
    if overlap:
        return overlap
    tail = parts[-1][-overlap_chars:].strip() if parts else ""
    return tail


def infer_content_type(page: dict, chunk_text: str) -> str:
    page_number = int(page.get("page_number", 0))
    text = str(chunk_text)
    headings = [str(heading).upper() for heading in page.get("headings", [])]
    if page_number <= 5 or "CONTENTS" in headings or text.upper().startswith("CONTENTS"):
        return "front_matter"
    if re.search(r"[=\u2248\u221d\u00b1\u00d7\u00f7\u221a\u03a3\u222b]", text):
        return "formula_or_derivation"
    if re.search(r"\b(example|exercise|summary|points to ponder)\b", text, re.IGNORECASE):
        return "pedagogical"
    return "expository"


def make_search_text(page: dict, chunk_text: str) -> str:
    if infer_content_type(page, chunk_text) == "front_matter":
        return chunk_text

    section = infer_section(page)
    context_parts = [
        str(page.get("chapter") or ""),
        section or "",
        " | ".join(page.get("headings") or []),
    ]
    context = " ".join(part for part in context_parts if part).strip()
    if context:
        return f"{context}\n\n{chunk_text}"
    return chunk_text


def infer_section(page: dict) -> str | None:
    headings = [str(heading).strip() for heading in page.get("headings") or []]
    numbered_headings = [heading for heading in headings if NUMBERED_HEADING_PATTERN.match(heading)]
    if numbered_headings:
        return numbered_headings[-1]
    return page.get("section")


def build_chunks(pages: list[dict]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for page in pages:
        text = normalize_spacing(str(page.get("text", "")))
        if len(text) < MIN_CHUNK_CHARS:
            continue

        page_number = int(page["page_number"])
        paragraphs = list(iter_paragraphs(text))
        page_chunks = chunk_paragraphs(paragraphs)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            content_type = infer_content_type(page, chunk_text)
            section = infer_section(page)
            citation = f"p. {page_number}"
            if section and content_type != "front_matter":
                citation = f"{citation}, {section}"
            chunks.append(
                ChunkRecord(
                    chunk_id=f"page_{page_number:03d}_chunk_{chunk_index:02d}",
                    document=str(page.get("document", "")),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    search_text=make_search_text(page, chunk_text),
                    citation=citation,
                    chapter=page.get("chapter"),
                    section=section,
                    headings=list(page.get("headings") or []),
                    formula_candidates=list(page.get("formula_candidates") or []),
                    content_type=content_type,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                )
            )
    return chunks


def write_chunks(chunks: list[ChunkRecord], path: Path = CHUNKS_JSONL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def print_summary(chunks: list[ChunkRecord]) -> None:
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.content_type] = by_type.get(chunk.content_type, 0) + 1

    print(f"Wrote chunk records: {CHUNKS_JSONL_PATH}")
    print(f"Chunks: {len(chunks)}")
    print("Chunk types:")
    for content_type, count in sorted(by_type.items()):
        print(f"  {content_type}: {count}")


def main() -> None:
    ensure_directories()
    pages = load_jsonl(PAGES_JSONL_PATH)
    chunks = build_chunks(pages)
    write_chunks(chunks)
    print_summary(chunks)


if __name__ == "__main__":
    main()
