# Use Case 3 Report: Hybrid RAG for NCERT Physics

## Summary

This report will document the ingestion pipeline, hybrid retrieval architecture, grounding strategy, evaluation results, limitations, and improvement plan.

## Architecture

```text
NCERT Physics PDF
  -> page parser
  -> page/heading-aware chunker
  -> FAISS vector index
  -> BM25 keyword index
  -> knowledge graph
  -> hybrid retriever and reranker
  -> grounded answer generator
  -> Streamlit live demo
```

## Status

- [ ] PDF parser
- [ ] Chunker
- [ ] Vector index
- [ ] Keyword index
- [ ] Graph index
- [ ] Hybrid retriever
- [ ] Grounded answer generation
- [ ] Live demo
- [ ] Evaluation

