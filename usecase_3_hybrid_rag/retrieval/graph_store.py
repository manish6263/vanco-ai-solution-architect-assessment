"""Knowledge graph construction and graph-based retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from config import GRAPH_GRAPHML_PATH, GRAPH_JSON_PATH
from retrieval.keyword_store import tokenize


STOP_TERMS = {
    "about",
    "after",
    "also",
    "because",
    "between",
    "chapter",
    "charge",
    "charges",
    "could",
    "current",
    "example",
    "figure",
    "field",
    "force",
    "from",
    "given",
    "into",
    "other",
    "point",
    "shown",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "where",
    "which",
    "with",
}

GRAPH_QUERY_STOP_TERMS = STOP_TERMS | {
    "and",
    "are",
    "for",
    "define",
    "does",
    "explain",
    "give",
    "how",
    "is",
    "laws",
    "mean",
    "the",
    "why",
    "what",
    "when",
    "where",
    "which",
    "who",
}

GENERIC_PHYSICS_TERMS = {
    "charge",
    "charges",
    "current",
    "electric",
    "field",
    "fields",
    "force",
    "forces",
    "law",
    "magnetic",
    "potential",
}


FORMULA_LINE_PATTERN = re.compile(
    r"[=\u2248\u221d\u00b1\u00d7\u00f7\u221a\u03a3\u222b]|(?:\b[A-Za-z]\s*=\s*)"
)


@dataclass
class GraphSearchResult:
    chunk: dict
    score: float
    matched_nodes: list[str]
    source: str = "graph"


def node_id(kind: str, value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r"[^a-z0-9._ -]+", "", value)
    return f"{kind}:{value[:120]}"


def add_node(graph: nx.Graph, kind: str, value: str, **attrs) -> str:
    identifier = node_id(kind, value)
    if not graph.has_node(identifier):
        graph.add_node(identifier, kind=kind, label=value, **attrs)
    return identifier


def add_edge(graph: nx.Graph, source: str, target: str, relation: str, weight: float = 1.0) -> None:
    if graph.has_edge(source, target):
        graph[source][target]["weight"] = float(graph[source][target].get("weight", 1.0)) + weight
        relations = set(str(graph[source][target].get("relation", "")).split("|"))
        relations.add(relation)
        graph[source][target]["relation"] = "|".join(sorted(item for item in relations if item))
    else:
        graph.add_edge(source, target, relation=relation, weight=weight)


def normalize_label(text: str) -> str:
    text = text.replace("â€™", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def candidate_concepts(chunk: dict, limit: int = 14) -> list[str]:
    concepts: list[str] = []
    for source in [chunk.get("section"), *chunk.get("headings", [])]:
        if source:
            concepts.append(clean_concept(str(source)))

    text = normalize_label(chunk.get("search_text") or chunk.get("text") or "")
    phrases = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\b", text)
    concepts.extend(clean_concept(phrase) for phrase in phrases)

    tokens = tokenize(text)
    for token in tokens:
        if len(token) >= 6 and token not in STOP_TERMS:
            concepts.append(token)

    seen: set[str] = set()
    cleaned: list[str] = []
    for concept in concepts:
        concept = clean_concept(concept)
        key = concept.lower()
        if not concept or len(concept) < 4 or key in STOP_TERMS or key in seen:
            continue
        seen.add(key)
        cleaned.append(concept)
        if len(cleaned) >= limit:
            break
    return cleaned


def clean_concept(text: str) -> str:
    text = normalize_label(text)
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    text = re.sub(r"\b\d+\b$", "", text)
    return text.strip(" -:;,.")


def extract_formulas(chunk: dict, limit: int = 8) -> list[str]:
    formulas = list(chunk.get("formula_candidates") or [])
    for line in str(chunk.get("text") or "").splitlines():
        line = normalize_label(line)
        if 5 <= len(line) <= 180 and FORMULA_LINE_PATTERN.search(line):
            formulas.append(line)

    seen: set[str] = set()
    result: list[str] = []
    for formula in formulas:
        key = formula.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(formula)
        if len(result) >= limit:
            break
    return result


class KnowledgeGraphStore:
    def __init__(self, graph: nx.Graph, chunks: dict[str, dict]):
        self.graph = graph
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[dict]) -> "KnowledgeGraphStore":
        graph = nx.Graph()
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        document_node = add_node(graph, "document", "NCERT Class 12 Physics Part 1")

        for chunk in chunks:
            chunk_node = add_node(
                graph,
                "chunk",
                chunk["chunk_id"],
                chunk_id=chunk["chunk_id"],
                citation=chunk.get("citation", ""),
                content_type=chunk.get("content_type", ""),
            )
            add_edge(graph, document_node, chunk_node, "HAS_CHUNK", weight=0.25)

            page_node = add_node(graph, "page", f"Page {chunk['page_number']}", page_number=chunk["page_number"])
            add_edge(graph, page_node, chunk_node, "HAS_CHUNK", weight=1.0)
            add_edge(graph, document_node, page_node, "HAS_PAGE", weight=0.2)

            section = chunk.get("section")
            if section:
                section_node = add_node(graph, "section", normalize_label(section))
                add_edge(graph, section_node, chunk_node, "HAS_EVIDENCE", weight=2.0)
                add_edge(graph, section_node, page_node, "MENTIONED_ON_PAGE", weight=0.5)

            for concept in candidate_concepts(chunk):
                concept_node = add_node(graph, "concept", concept)
                add_edge(graph, concept_node, chunk_node, "MENTIONED_IN", weight=1.0)
                if section:
                    add_edge(graph, section_node, concept_node, "HAS_CONCEPT", weight=0.5)

            for formula in extract_formulas(chunk):
                formula_node = add_node(graph, "formula", formula)
                add_edge(graph, formula_node, chunk_node, "APPEARS_IN", weight=1.25)
                if section:
                    add_edge(graph, section_node, formula_node, "USES_FORMULA", weight=0.5)

        return cls(graph=graph, chunks=chunks_by_id)

    def save(
        self,
        json_path: Path = GRAPH_JSON_PATH,
        graphml_path: Path = GRAPH_GRAPHML_PATH,
    ) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": json_graph.node_link_data(self.graph, edges="links"),
            "chunks": self.chunks,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        graphml_graph = nx.Graph()
        for node, attrs in self.graph.nodes(data=True):
            graphml_graph.add_node(node, **{key: str(value) for key, value in attrs.items()})
        for source, target, attrs in self.graph.edges(data=True):
            graphml_graph.add_edge(source, target, **{key: str(value) for key, value in attrs.items()})
        nx.write_graphml(graphml_graph, graphml_path)

    @classmethod
    def load(cls, json_path: Path = GRAPH_JSON_PATH) -> "KnowledgeGraphStore":
        if not json_path.exists():
            raise SystemExit(f"Missing graph index: {json_path}. Run `python -m ingestion.build_indexes`.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        graph = json_graph.node_link_graph(payload["graph"], edges="links")
        return cls(graph=graph, chunks=payload["chunks"])

    def search(self, query: str, top_k: int = 8) -> list[GraphSearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        distinctive_terms = {
            term
            for term in query_terms
            if term not in GRAPH_QUERY_STOP_TERMS and term not in GENERIC_PHYSICS_TERMS
        }
        required_terms = distinctive_terms or (query_terms - GRAPH_QUERY_STOP_TERMS) or query_terms

        matched_nodes: list[str] = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") not in {"section", "concept", "formula"}:
                continue
            label_terms = set(tokenize(str(attrs.get("label", ""))))
            if required_terms & label_terms:
                matched_nodes.append(node)

        scores: dict[str, float] = {}
        evidence_nodes: dict[str, list[str]] = {}
        for matched_node in matched_nodes:
            label_terms = set(tokenize(str(self.graph.nodes[matched_node].get("label", ""))))
            overlap = len(required_terms & label_terms) / max(len(required_terms), 1)
            if overlap <= 0:
                continue
            for neighbor in self.graph.neighbors(matched_node):
                neighbor_attrs = self.graph.nodes[neighbor]
                if neighbor_attrs.get("kind") == "chunk":
                    chunk_id = neighbor_attrs["chunk_id"]
                    edge_weight = float(self.graph[matched_node][neighbor].get("weight", 1.0))
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + overlap * edge_weight
                    evidence_nodes.setdefault(chunk_id, []).append(str(self.graph.nodes[matched_node].get("label")))
                    continue
                for second_hop in self.graph.neighbors(neighbor):
                    second_attrs = self.graph.nodes[second_hop]
                    if second_attrs.get("kind") != "chunk":
                        continue
                    chunk_id = second_attrs["chunk_id"]
                    edge_weight = float(self.graph[neighbor][second_hop].get("weight", 1.0))
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + overlap * edge_weight * 0.45
                    evidence_nodes.setdefault(chunk_id, []).append(str(self.graph.nodes[matched_node].get("label")))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            GraphSearchResult(
                chunk=self.chunks[chunk_id],
                score=float(score),
                matched_nodes=sorted(set(evidence_nodes.get(chunk_id, [])))[:6],
            )
            for chunk_id, score in ranked
            if chunk_id in self.chunks
        ]
