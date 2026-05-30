"""BM25 keyword retrieval over chunk text."""

from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import CHUNKS_JSONL_PATH, KEYWORD_INDEX_PATH, VECTOR_METADATA_PATH


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass
class KeywordSearchResult:
    chunk: dict
    score: float
    source: str = "keyword_bm25"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25KeywordStore:
    def __init__(self, bm25, chunks: list[dict], tokenized_corpus: list[list[str]]):
        self.bm25 = bm25
        self.chunks = chunks
        self.tokenized_corpus = tokenized_corpus

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25KeywordStore":
        tokenized_corpus = [tokenize(chunk.get("search_text") or chunk["text"]) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            print("rank-bm25 not installed; using built-in BM25 fallback.")
            bm25 = SimpleBM25(tokenized_corpus)

        return cls(bm25=bm25, chunks=chunks, tokenized_corpus=tokenized_corpus)

    def save(self, path: Path = KEYWORD_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                file,
            )

    @classmethod
    def load(cls, path: Path = KEYWORD_INDEX_PATH) -> "BM25KeywordStore":
        if not path.exists():
            raise SystemExit(f"Missing BM25 index: {path}. Run `python -m ingestion.build_indexes`.")
        try:
            with path.open("rb") as file:
                payload = pickle.load(file)
        except ModuleNotFoundError:
            print("BM25 pickle uses an unavailable package; rebuilding keyword index from chunks.")
            chunks = load_chunks_for_rebuild()
            return cls.build(chunks)
        return cls(
            bm25=payload["bm25"],
            chunks=payload["chunks"],
            tokenized_corpus=payload["tokenized_corpus"],
        )

    def search(self, query: str, top_k: int = 8) -> list[KeywordSearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[KeywordSearchResult] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(KeywordSearchResult(chunk=self.chunks[int(index)], score=score))
        return results


class SimpleBM25:
    """Small BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(document) for document in corpus], dtype=np.float32)
        self.avg_doc_length = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.term_frequencies = [Counter(document) for document in corpus]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self.corpus)
        document_frequency: Counter[str] = Counter()
        for document in self.corpus:
            document_frequency.update(set(document))
        return {
            term: float(np.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        if not query_tokens or self.avg_doc_length == 0:
            return scores

        for index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            denominator_base = self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score = 0.0
            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + denominator_base
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores[index] = score
        return scores


def load_chunks_for_rebuild() -> list[dict]:
    source_path = VECTOR_METADATA_PATH if VECTOR_METADATA_PATH.exists() else CHUNKS_JSONL_PATH
    if not source_path.exists():
        raise SystemExit("Cannot rebuild BM25 index because chunk metadata is missing.")
    rows: list[dict] = []
    with source_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                import json

                rows.append(json.loads(line))
    return rows
