"""FAISS-backed vector retrieval over embedded chunks."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from config import VECTOR_ENCODER_PATH, VECTOR_INDEX_PATH, VECTOR_METADATA_PATH


@dataclass
class VectorSearchResult:
    chunk: dict
    score: float
    source: str = "vector_faiss"


def import_faiss(required: bool = True):
    try:
        import faiss
    except ImportError as exc:
        if required:
            raise SystemExit(
                "faiss-cpu is required for FAISS vector indexing. "
                "Run `pip install -r requirements.txt` from usecase_3_hybrid_rag."
            ) from exc
        return None
    return faiss


def build_sklearn_index(chunks: list[dict], encoder: TfidfVectorizer | None = None) -> "FaissVectorStore":
    texts = [chunk.get("search_text") or chunk["text"] for chunk in chunks]
    if encoder is None:
        encoder = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        matrix = encoder.fit_transform(texts)
    else:
        matrix = encoder.transform(texts)
    vectors = normalize(matrix, norm="l2", axis=1).astype(np.float32).toarray()
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(vectors)
    return FaissVectorStore(
        index=index,
        encoder=encoder,
        chunks=chunks,
        backend="sklearn_nearest_neighbors",
        vectors=vectors,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class FaissVectorStore:
    def __init__(
        self,
        index,
        encoder: TfidfVectorizer,
        chunks: list[dict],
        backend: str,
        vectors: np.ndarray | None = None,
    ):
        self.index = index
        self.encoder = encoder
        self.chunks = chunks
        self.backend = backend
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[dict]) -> "FaissVectorStore":
        faiss = import_faiss(required=False)
        texts = [chunk.get("search_text") or chunk["text"] for chunk in chunks]
        encoder = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        matrix = encoder.fit_transform(texts)
        vectors = normalize(matrix, norm="l2", axis=1).astype(np.float32).toarray()
        if faiss is not None:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return cls(index=index, encoder=encoder, chunks=chunks, backend="faiss")

        print("faiss-cpu not installed; using sklearn NearestNeighbors fallback.")
        return build_sklearn_index(chunks, encoder=encoder)

    def save(
        self,
        index_path: Path = VECTOR_INDEX_PATH,
        encoder_path: Path = VECTOR_ENCODER_PATH,
        metadata_path: Path = VECTOR_METADATA_PATH,
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            faiss = import_faiss(required=True)
            faiss.write_index(self.index, str(index_path))
        else:
            with index_path.open("wb") as file:
                pickle.dump(
                    {
                        "backend": self.backend,
                        "index": self.index,
                        "vectors": self.vectors,
                    },
                    file,
                )
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
        backend = "faiss"
        vectors = None
        try:
            with index_path.open("rb") as file:
                payload = pickle.load(file)
            if isinstance(payload, dict) and payload.get("backend"):
                backend = payload["backend"]
                index = payload["index"]
                vectors = payload.get("vectors")
            else:
                raise ValueError("Not a sklearn vector index payload.")
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with encoder_path.open("rb") as file:
                encoder = pickle.load(file)
        chunks = read_jsonl(metadata_path)
        try:
            index
        except UnboundLocalError:
            faiss = import_faiss(required=False)
            if faiss is not None:
                index = faiss.read_index(str(index_path))
            else:
                print("FAISS index exists but faiss-cpu is unavailable; rebuilding sklearn vector index.")
                return build_sklearn_index(chunks, encoder=encoder)
        return cls(index=index, encoder=encoder, chunks=chunks, backend=backend, vectors=vectors)

    def search(self, query: str, top_k: int = 8) -> list[VectorSearchResult]:
        query_vector = self.encoder.transform([query])
        query_vector = normalize(query_vector, norm="l2", axis=1).astype(np.float32).toarray()
        if self.backend == "faiss":
            scores, indices = self.index.search(query_vector, top_k)
            score_values = scores[0]
            index_values = indices[0]
        else:
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=top_k)
            score_values = 1.0 - distances[0]
            index_values = indices[0]

        results: list[VectorSearchResult] = []
        for score, index in zip(score_values, index_values):
            if index < 0:
                continue
            results.append(VectorSearchResult(chunk=self.chunks[int(index)], score=float(score)))
        return results
