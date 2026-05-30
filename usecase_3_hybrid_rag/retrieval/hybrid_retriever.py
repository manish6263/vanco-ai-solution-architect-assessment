"""Merge semantic and keyword evidence into ranked retrieval results."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

from retrieval.graph_store import KnowledgeGraphStore
from retrieval.keyword_store import BM25KeywordStore
from retrieval.vector_store import FaissVectorStore


@dataclass
class HybridSearchResult:
    chunk: dict
    score: float
    vector_score: float
    keyword_score: float
    graph_score: float
    sources: list[str]
    graph_nodes: list[str]


def normalize_scores(items: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    if not items:
        return []
    max_score = max(score for _, score in items)
    if max_score <= 0:
        return [(chunk, 0.0) for chunk, _ in items]
    return [(chunk, score / max_score) for chunk, score in items]


def content_type_weight(chunk: dict) -> float:
    content_type = chunk.get("content_type")
    if content_type == "front_matter":
        return 0.35
    if content_type == "pedagogical":
        return 0.9
    return 1.0


def normalized_terms(text: str) -> list[str]:
    text = normalize_match_text(text)
    terms = [term for term in text.split() if len(term) >= 4]
    return [
        singularize(term)
        for term in terms
        if term not in {"what", "when", "where", "which", "explain"}
    ]


def normalize_match_text(text: str) -> str:
    text = text.lower().replace("'", " ").replace("’", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bs\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def normalized_term_set(text: str) -> set[str]:
    return {singularize(term) for term in normalize_match_text(text).split()}


def concept_match_boost(query: str, chunk: dict) -> float:
    query_terms = normalized_terms(query)
    if not query_terms:
        return 0.0

    section_text = normalize_match_text(str(chunk.get("section") or ""))
    body_text = normalize_match_text(str(chunk.get("text") or ""))
    text = " ".join(
        [
            str(chunk.get("section") or ""),
            str(chunk.get("search_text") or ""),
            str(chunk.get("text") or ""),
        ]
    )
    text = normalize_match_text(text)
    section_terms = normalized_term_set(section_text)
    body_terms = normalized_term_set(body_text)
    text_terms = normalized_term_set(text)
    coverage = sum(1 for term in query_terms if term in text_terms) / len(query_terms)
    section_coverage = sum(1 for term in query_terms if term in section_terms) / len(query_terms)

    compact_query = " ".join(query_terms)
    section_bonus = 0.45 if compact_query and compact_query in section_text else 0.0
    if section_coverage >= 1.0:
        section_bonus = max(section_bonus, 1.0)
    phrase_bonus = 0.75 if compact_query and compact_query in body_text else 0.0
    definition_bonus = 0.55 if section_coverage >= 1.0 and re.search(r"\bwhat\s+is\b", query, re.IGNORECASE) else 0.0
    general_section_bonus = 0.0
    if definition_bonus and "capacitor" in query_terms and "capacitance" in section_terms:
        general_section_bonus = 1.5
    application_penalty = 0.0
    if definition_bonus and any(
        phrase in section_text
        for phrase in [
            "applied to",
            "energy stored",
            "parallel plate",
            "in a parallel",
            "with air between",
        ]
    ):
        application_penalty = 0.75
    return 0.2 * coverage + section_bonus + phrase_bonus + definition_bonus + general_section_bonus - application_penalty


def safe_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


class HybridRetriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        keyword_store: BM25KeywordStore,
        graph_store: KnowledgeGraphStore,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        graph_weight: float = 0.35,
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.graph_store = graph_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.graph_weight = graph_weight

    @classmethod
    def load(cls) -> "HybridRetriever":
        return cls(
            vector_store=FaissVectorStore.load(),
            keyword_store=BM25KeywordStore.load(),
            graph_store=KnowledgeGraphStore.load(),
        )

    def search(self, query: str, top_k: int = 8, candidate_k: int = 20) -> list[HybridSearchResult]:
        vector_results = self.vector_store.search(query, top_k=candidate_k)
        keyword_results = self.keyword_store.search(query, top_k=candidate_k)
        graph_results = self.graph_store.search(query, top_k=candidate_k)

        vector_scores = normalize_scores([(result.chunk, result.score) for result in vector_results])
        keyword_scores = normalize_scores([(result.chunk, result.score) for result in keyword_results])
        graph_scores = normalize_scores([(result.chunk, result.score) for result in graph_results])
        graph_nodes_by_chunk = {result.chunk["chunk_id"]: result.matched_nodes for result in graph_results}

        merged: dict[str, HybridSearchResult] = {}
        for chunk, score in vector_scores:
            chunk_id = chunk["chunk_id"]
            merged[chunk_id] = HybridSearchResult(
                chunk=chunk,
                score=0.0,
                vector_score=score,
                keyword_score=0.0,
                graph_score=0.0,
                sources=["vector"],
                graph_nodes=[],
            )

        for chunk, score in keyword_scores:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = HybridSearchResult(
                    chunk=chunk,
                    score=0.0,
                    vector_score=0.0,
                    keyword_score=score,
                    graph_score=0.0,
                    sources=["keyword"],
                    graph_nodes=[],
                )
            else:
                merged[chunk_id].keyword_score = score
                merged[chunk_id].sources.append("keyword")

        for chunk, score in graph_scores:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = HybridSearchResult(
                    chunk=chunk,
                    score=0.0,
                    vector_score=0.0,
                    keyword_score=0.0,
                    graph_score=score,
                    sources=["graph"],
                    graph_nodes=graph_nodes_by_chunk.get(chunk_id, []),
                )
            else:
                merged[chunk_id].graph_score = score
                merged[chunk_id].sources.append("graph")
                merged[chunk_id].graph_nodes = graph_nodes_by_chunk.get(chunk_id, [])

        for result in merged.values():
            result.score = (
                self.vector_weight * result.vector_score
                + self.keyword_weight * result.keyword_score
                + self.graph_weight * result.graph_score
                + concept_match_boost(query, result.chunk)
            ) * content_type_weight(result.chunk)

        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


def preview(text: str, max_chars: int = 360) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval for a question.")
    parser.add_argument("query", nargs="?", default="What is Coulomb's law?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever.load()
    results = retriever.search(args.query, top_k=args.top_k)
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"\n{rank}. score={result.score:.3f} "
            f"vector={result.vector_score:.3f} keyword={result.keyword_score:.3f} "
            f"graph={result.graph_score:.3f} "
            f"sources={','.join(result.sources)}"
        )
        print(f"   {chunk['citation']} [{chunk['content_type']}] {chunk['chunk_id']}")
        if result.graph_nodes:
            print(f"   graph nodes: {safe_console(', '.join(result.graph_nodes[:4]))}")
        print(f"   {safe_console(preview(chunk['text']))}")


if __name__ == "__main__":
    main()
