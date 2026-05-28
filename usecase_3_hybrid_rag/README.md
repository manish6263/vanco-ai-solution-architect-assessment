# Use Case 3: Hybrid RAG Application for NCERT Physics PDF

## Objective

Build a grounded RAG application that answers questions from the NCERT Class 12 Physics Part 1 PDF using vector retrieval, graph retrieval, semantic search, and keyword search.

## Planned Architecture

```text
Physics PDF
    -> PDF parsing with page/section metadata
    -> section-aware chunking
    -> vector index for semantic retrieval
    -> keyword index for BM25 retrieval
    -> knowledge graph for concepts/formulas/sections
    -> hybrid retrieval and reranking
    -> grounded answer generation with citations
    -> live QA interface with evidence display
```

## Critical Behavior

The application must not behave like a generic chatbot. Answers must be grounded in the source PDF. If the retrieved evidence does not support an answer, the system should state that the information is not available in the source document.

## Components

Planned components:

- PDF parser: PyMuPDF or pdfplumber
- Chunking: page-aware and heading-aware chunks
- Vector DB/index: FAISS or Chroma
- Keyword search: BM25
- Graph DB/representation: Neo4j preferred, with a documented fallback if needed
- Backend/API: FastAPI or Streamlit-native backend
- UI: Streamlit or lightweight web app
- LLM: configurable provider through environment variables

## Graph Design

Planned node types:

- Chapter
- Section
- Topic
- Concept
- Formula
- Definition
- Page

Planned edge types:

- `CONTAINS`
- `MENTIONED_ON_PAGE`
- `RELATED_TO`
- `USES_FORMULA`
- `DEFINED_IN`

## Retrieval Design

For each query:

1. Retrieve semantic matches from the vector index.
2. Retrieve exact/token matches from BM25.
3. Extract concepts and query graph neighbors.
4. Merge and rerank evidence.
5. Build a constrained prompt using only retrieved evidence.
6. Return answer, citations, and visible retrieval evidence.

## Evaluation Plan

Prepare a test set covering:

- Factual questions
- Conceptual questions
- Formula-based questions
- Comparison questions
- Multi-section questions
- Unsupported questions

Track:

- Retrieval quality
- Citation correctness
- Groundedness
- Latency
- Failure modes

## Deliverables

- [ ] Ingestion pipeline
- [ ] Vector index
- [ ] Keyword index
- [ ] Graph database/export
- [ ] Live QA app
- [ ] Evidence display
- [ ] Example questions and answers
- [ ] Architecture diagram
- [ ] Screenshots/logs
- [ ] Limitations and improvement plan

