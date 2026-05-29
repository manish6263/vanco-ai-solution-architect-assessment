"""Shared paths and constants for the hybrid RAG use case."""

from pathlib import Path


USECASE_DIR = Path(__file__).resolve().parent
DATA_DIR = USECASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = USECASE_DIR / "indexes"
GRAPH_DIR = USECASE_DIR / "graph"
REPORTS_DIR = USECASE_DIR / "reports"
SCREENSHOTS_DIR = USECASE_DIR / "screenshots"

PDF_URL = "https://www.drishtiias.com/images/pdf/NCERT-Class-12-Physics-Part-1.pdf"
PDF_FILENAME = "ncert_physics_part1.pdf"
PDF_PATH = RAW_DATA_DIR / PDF_FILENAME

PAGES_JSONL_PATH = PROCESSED_DATA_DIR / "pages.jsonl"
CHUNKS_JSONL_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
VECTOR_INDEX_PATH = INDEXES_DIR / "faiss.index"
VECTOR_METADATA_PATH = INDEXES_DIR / "vector_metadata.jsonl"
KEYWORD_INDEX_PATH = INDEXES_DIR / "bm25.pkl"
GRAPH_JSON_PATH = GRAPH_DIR / "knowledge_graph.json"
GRAPH_GRAPHML_PATH = GRAPH_DIR / "knowledge_graph.graphml"


def ensure_directories() -> None:
    """Create local data/artifact directories used by the pipeline."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INDEXES_DIR,
        GRAPH_DIR,
        REPORTS_DIR,
        SCREENSHOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
