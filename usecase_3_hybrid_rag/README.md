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
- Graph DB/representation: NetworkX persisted graph, with Neo4j export path documented as the production replacement
- Backend/API: Streamlit-native backend for live demo
- UI: Streamlit
- LLM: configurable provider through environment variables

## Repository Layout

```text
usecase_3_hybrid_rag/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- app/
|   `-- streamlit_app.py
|-- ingestion/
|   |-- parse_pdf.py
|   |-- chunking.py
|   |-- build_indexes.py
|   `-- ingest.py
|-- retrieval/
|   |-- vector_store.py
|   |-- keyword_store.py
|   |-- graph_store.py
|   `-- hybrid_retriever.py
|-- generation/
|   |-- prompts.py
|   `-- answer.py
|-- evaluation/
|   |-- sample_questions.md
|   `-- evaluate_retrieval.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- indexes/
|-- graph/
|-- reports/
`-- screenshots/
```

## Setup

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional transformer embeddings:

```bash
pip install -r requirements-embeddings.txt
```

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional transformer embeddings:

```bash
pip install -r requirements-embeddings.txt
```

## Source PDF

Option A, download directly with the helper:

```bash
python -m ingestion.download_pdf
python -m ingestion.validate_pdf
```

Option B, download manually from:

```text
https://www.drishtiias.com/images/pdf/NCERT-Class-12-Physics-Part-1.pdf
```

Place it at:

```text
usecase_3_hybrid_rag/data/raw/ncert_physics_part1.pdf
```

Then validate it:

```bash
python -m ingestion.validate_pdf
```

## Planned Commands

Current ingestion commands:

```bash
python -m ingestion.validate_pdf
python -m ingestion.parse_pdf
python -m ingestion.chunking
python -m ingestion.ingest
```

Planned commands that will become active as implementation progresses:

```bash
python -m evaluation.evaluate_retrieval
streamlit run app/streamlit_app.py
```

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
