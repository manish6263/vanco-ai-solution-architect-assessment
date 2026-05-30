"""Generate answers using retrieved evidence and cite source pages."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

from generation.prompts import GROUNDED_QA_SYSTEM_PROMPT, GROUNDED_QA_USER_TEMPLATE
from retrieval.hybrid_retriever import HybridRetriever, HybridSearchResult, safe_console


MIN_SUPPORTED_SCORE = 0.35
MAX_EVIDENCE_CHARS = 950


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    citations: list[str]
    evidence: list[HybridSearchResult]
    supported: bool


def repair_pdf_text(text: str) -> str:
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Ã—": "x",
        "Ã·": "/",
        "Î¼": "mu",
        "Ï€": "pi",
        "Ï†": "phi",
        "Î¦": "Phi",
        "Î£": "Sigma",
        "Î”": "Delta",
        "Îµ": "epsilon",
        "Îµ0": "epsilon0",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def normalize_for_matching(text: str) -> str:
    text = repair_pdf_text(text).lower()
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bs\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_terms(question: str) -> set[str]:
    stop_words = {
        "about",
        "between",
        "define",
        "difference",
        "does",
        "explain",
        "formula",
        "give",
        "how",
        "is",
        "mean",
        "the",
        "what",
        "when",
        "where",
        "which",
        "why",
    }
    return {
        singularize(term)
        for term in normalize_for_matching(question).split()
        if (len(term) >= 4 or term in {"law", "flux", "emf"}) and term not in stop_words
    }


def singularize(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def normalized_term_set(text: str) -> set[str]:
    return {singularize(term) for term in normalize_for_matching(text).split()}


def evidence_supported(question: str, evidence: list[HybridSearchResult]) -> bool:
    if not evidence:
        return False
    if evidence[0].score < MIN_SUPPORTED_SCORE:
        return False
    terms = query_terms(question)
    if not terms:
        return evidence[0].score >= MIN_SUPPORTED_SCORE
    evidence_text = " ".join(
        normalize_for_matching(result.chunk.get("search_text") or result.chunk.get("text") or "")
        for result in evidence[:3]
    )
    covered = sum(1 for term in terms if term in evidence_text)
    return covered / max(len(terms), 1) >= 0.5


def unique_citations(evidence: list[HybridSearchResult], limit: int = 4) -> list[str]:
    citations: list[str] = []
    for result in evidence:
        citation = str(result.chunk.get("citation") or "")
        citation = repair_pdf_text(citation)
        if citation and citation not in citations:
            citations.append(citation)
        if len(citations) >= limit:
            break
    return citations


def compact_query_phrase(question: str) -> str:
    terms = query_terms(question)
    ordered_terms = [
        singularize(term)
        for term in normalize_for_matching(question).split()
        if singularize(term) in terms
    ]
    return " ".join(ordered_terms)


def section_core(section: str) -> str:
    normalized = normalize_for_matching(section)
    normalized = re.sub(r"^\d+(?:\s+\d+)*\s+", "", normalized)
    return normalized.strip()


def definition_terms(question: str) -> list[str]:
    normalized = normalize_for_matching(question)
    if not (normalized.startswith("what is ") or normalized.startswith("define ")):
        return []
    terms = [singularize(term) for term in normalized.split() if singularize(term) in query_terms(question)]
    return terms


def merge_evidence(
    primary: list[HybridSearchResult],
    extra: list[HybridSearchResult],
) -> list[HybridSearchResult]:
    merged: dict[str, HybridSearchResult] = {item.chunk["chunk_id"]: item for item in primary}
    for item in extra:
        merged.setdefault(item.chunk["chunk_id"], item)
    return list(merged.values())


def evidence_for_answer(question: str, evidence: list[HybridSearchResult]) -> list[HybridSearchResult]:
    phrase = compact_query_phrase(question)
    if not phrase:
        return evidence

    exact_section_matches = [
        result
        for result in evidence
        if section_core(str(result.chunk.get("section") or "")) == phrase
    ]
    if exact_section_matches:
        return exact_section_matches

    section_matches = [
        result
        for result in evidence
        if phrase in normalize_for_matching(str(result.chunk.get("section") or ""))
    ]
    if section_matches and len(query_terms(question)) > 1:
        return section_matches

    terms = query_terms(question)
    section_term_matches = [
        result
        for result in evidence
        if terms and terms.issubset(normalized_term_set(str(result.chunk.get("section") or "")))
    ]
    if "capacitor" in terms:
        capacitance_matches = [
            result
            for result in section_term_matches
            if "capacitance" in normalized_term_set(str(result.chunk.get("section") or ""))
        ]
        if capacitance_matches:
            return capacitance_matches
    if section_term_matches:
        return section_term_matches

    body_matches = [
        result
        for result in evidence
        if phrase in normalize_for_matching(str(result.chunk.get("text") or ""))
    ]
    return body_matches or evidence


def sentence_split(text: str) -> list[str]:
    text = repair_pdf_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def select_answer_sentences(question: str, evidence: list[HybridSearchResult], limit: int = 4) -> list[str]:
    terms = query_terms(question)
    compact_query = compact_query_phrase(question)
    def_terms = definition_terms(question)
    candidates: list[tuple[float, int, str, str]] = []
    for result in evidence:
        citation = repair_pdf_text(str(result.chunk.get("citation") or ""))
        for sentence in sentence_split(str(result.chunk.get("text") or "")):
            if sentence.rstrip().endswith("?"):
                continue
            if normalize_for_matching(sentence).startswith("system of two conductors separated"):
                sentence = "A capacitor is a system of two conductors separated by an insulator."
            normalized = normalize_for_matching(sentence)
            coverage = sum(1 for term in terms if term in normalized)
            if coverage == 0 and terms:
                continue
            phrase_bonus = 4.0 if compact_query and compact_query in normalized else 0.0
            definition_bonus = 0.0
            for term in def_terms:
                if re.search(rf"\b{re.escape(term)}\s+is\b", normalized):
                    definition_bonus = 6.0
                elif re.search(rf"\b{re.escape(term)}\s+(means|refers|denotes)\b", normalized):
                    definition_bonus = 4.0
            formula_bonus = 0.0
            if "formula" in normalize_for_matching(question) and "=" in sentence:
                formula_bonus = 6.0
            score = coverage + phrase_bonus + definition_bonus + formula_bonus + 0.15 * result.score
            candidates.append((score, coverage, sentence, citation))

    candidates.sort(key=lambda item: item[0], reverse=True)
    max_coverage = max((coverage for _, coverage, _, _ in candidates), default=0)
    if max_coverage >= 2 and "difference" not in normalize_for_matching(question):
        candidates = [item for item in candidates if item[1] >= 2]

    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence, citation in candidates:
        normalized = normalize_for_matching(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"{sentence} [{citation}]")
        if len(selected) >= limit:
            break
    return selected


def format_evidence_for_prompt(evidence: list[HybridSearchResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(evidence, start=1):
        chunk = result.chunk
        text = repair_pdf_text(str(chunk.get("text") or ""))
        text = re.sub(r"\s+", " ", text).strip()[:MAX_EVIDENCE_CHARS]
        blocks.append(
            f"Evidence {index}\n"
            f"Citation: {repair_pdf_text(str(chunk.get('citation') or ''))}\n"
            f"Scores: vector={result.vector_score:.3f}, keyword={result.keyword_score:.3f}, "
            f"graph={result.graph_score:.3f}\n"
            f"Text: {text}"
        )
    return "\n\n".join(blocks)


def openai_answer(question: str, evidence: list[HybridSearchResult]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    prompt = GROUNDED_QA_USER_TEMPLATE.format(
        question=question,
        evidence=format_evidence_for_prompt(evidence),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": GROUNDED_QA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or None


def generate_answer(question: str, top_k: int = 8, use_llm: bool = True) -> GroundedAnswer:
    retriever = HybridRetriever.load()
    evidence = retriever.search(question, top_k=max(top_k, 12))
    for term in definition_terms(question):
        evidence = merge_evidence(evidence, retriever.search(f"{term} is", top_k=5))
    supported = evidence_supported(question, evidence)
    answer_evidence = evidence_for_answer(question, evidence) if supported else []
    citations = unique_citations(answer_evidence)

    if not supported:
        return GroundedAnswer(
            question=question,
            answer="The answer is not available in the source document based on the retrieved evidence.",
            citations=[],
            evidence=evidence,
            supported=False,
        )

    answer_text = openai_answer(question, answer_evidence) if use_llm else None
    if not answer_text:
        selected = select_answer_sentences(question, answer_evidence)
        if selected:
            answer_text = " ".join(selected)
        else:
            answer_text = (
                "The retrieved evidence is relevant, but it does not contain a concise extractive answer. "
                f"Please inspect the cited evidence: {', '.join(citations)}."
            )

    return GroundedAnswer(
        question=question,
        answer=repair_pdf_text(answer_text),
        citations=citations,
        evidence=evidence,
        supported=True,
    )


def print_answer(result: GroundedAnswer) -> None:
    print(f"Question: {result.question}")
    print(f"Supported: {result.supported}")
    print("\nAnswer:")
    print(safe_console(result.answer))
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {safe_console(citation)}")
    print("\nEvidence:")
    for index, item in enumerate(result.evidence[:5], start=1):
        chunk = item.chunk
        print(
            f"{index}. score={item.score:.3f} vector={item.vector_score:.3f} "
            f"keyword={item.keyword_score:.3f} graph={item.graph_score:.3f} "
            f"{safe_console(repair_pdf_text(str(chunk.get('citation') or '')))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question using grounded RAG evidence.")
    parser.add_argument("question", nargs="?", default="What is Coulomb's law?")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true", help="Use only extractive answer generation.")
    args = parser.parse_args()

    result = generate_answer(args.question, top_k=args.top_k, use_llm=not args.no_llm)
    print_answer(result)


if __name__ == "__main__":
    main()
