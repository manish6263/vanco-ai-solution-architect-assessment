"""Build vector and keyword indexes from parsed chunks."""

from __future__ import annotations

from collections import Counter

from config import CHUNKS_JSONL_PATH, KEYWORD_INDEX_PATH, VECTOR_INDEX_PATH, ensure_directories
from ingestion.chunking import load_jsonl
from retrieval.keyword_store import BM25KeywordStore
from retrieval.vector_store import FaissVectorStore


def main() -> None:
    ensure_directories()
    chunks = load_jsonl(CHUNKS_JSONL_PATH)
    if not chunks:
        raise SystemExit("No chunks found. Run `python -m ingestion.chunking` first.")

    print(f"Loaded chunks: {len(chunks)}")
    print("Chunk types:")
    for content_type, count in sorted(Counter(chunk["content_type"] for chunk in chunks).items()):
        print(f"  {content_type}: {count}")

    print("Building BM25 keyword index")
    keyword_store = BM25KeywordStore.build(chunks)
    keyword_store.save(KEYWORD_INDEX_PATH)
    print(f"Wrote BM25 index: {KEYWORD_INDEX_PATH}")

    print("Building FAISS vector index with lightweight TF-IDF embeddings")
    vector_store = FaissVectorStore.build(chunks)
    vector_store.save()
    print(f"Wrote FAISS index: {VECTOR_INDEX_PATH}")
    print("Index build complete.")


if __name__ == "__main__":
    main()
