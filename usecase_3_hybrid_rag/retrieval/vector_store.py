"""FAISS-backed vector retrieval over embedded chunks."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import VECTOR_ENCODER_PATH, VECTOR_INDEX_PATH, VECTOR_METADATA_PATH


@dataclass
class VectorSearchResult:
    chunk: dict
    score: float
    source: str = "vector_faiss"


def import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "faiss-cpu is required for vector indexing. "
            "Run `pip install -r requirements.txt` from usecase_3_hybrid_rag."
        ) from exc
    return faiss


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class FaissVectorStore:
    def __init__(self, index, encoder: TfidfVectorizer, chunks: list[dict]):
        self.index = index
        self.encoder = encoder
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[dict]) -> "FaissVectorStore":
        faiss = import_faiss()
        texts = [chunk.get("search_text") or chunk["text"] for chunk in chunks]
        encoder = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        matrix = encoder.fit_transform(texts)
        vectors = normalize(matrix, norm="l2", axis=1).astype(np.float32).toarray()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, encoder=encoder, chunks=chunks)

    def save(
        self,
        index_path: Path = VECTOR_INDEX_PATH,
        encoder_path: Path = VECTOR_ENCODER_PATH,
        metadata_path: Path = VECTOR_METADATA_PATH,
    ) -> None:
        faiss = import_faiss()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with encoder_path.open("wb") as file:
            pickle.dump(self.encoder, file)
        write_jsonl(metadata_path, self.chunks)

    @classmethod
    def load(
        cls,
        index_path: Path = VECTOR_INDEX_PATH,
        encoder_path: Path = VECTOR_ENCODER_PATH,
        metadata_path: Path = VECTOR_METADATA_PATH,
    ) -> "FaissVectorStore":
        if not index_path.exists() or not encoder_path.exists() or not metadata_path.exists():
            raise SystemExit("Missing vector index files. Run `python -m ingestion.build_indexes`.")
        faiss = import_faiss()
        index = faiss.read_index(str(index_path))
        with encoder_path.open("rb") as file:
            encoder = pickle.load(file)
        chunks = read_jsonl(metadata_path)
        return cls(index=index, encoder=encoder, chunks=chunks)

    def search(self, query: str, top_k: int = 8) -> list[VectorSearchResult]:
        query_vector = self.encoder.transform([query])
        query_vector = normalize(query_vector, norm="l2", axis=1).astype(np.float32).toarray()
        scores, indices = self.index.search(query_vector, top_k)

        results: list[VectorSearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(VectorSearchResult(chunk=self.chunks[int(index)], score=float(score)))
        return results
