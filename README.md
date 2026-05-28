# Vanco AI Solution Architect Assessment

This repository contains the complete submission package for the Vanco AI Solution Architect technical assessment.

The assessment has three required use cases:

1. Grocery sales forecasting with external events
2. American Sign Language detection with a live webcam demo
3. Hybrid RAG application for the NCERT Class 12 Physics Part 1 PDF

The work is organized as one repository with clearly separated folders so reviewers can inspect, run, and discuss each system independently.

## Repository Structure

```text
.
├── README.md
├── SUBMISSION.md
├── DISCLOSURE.md
├── reports/
├── diagrams/
├── usecase_1_forecasting/
├── usecase_2_asl_detection/
└── usecase_3_hybrid_rag/
```

## Use Cases

### 1. Grocery Sales Forecasting

Folder: `usecase_1_forecasting/`

Goal: Build a Kaggle forecasting solution for store/product-family sales using the Corporacion Favorita dataset and external event features.

Expected outputs:

- Training and inference notebook or scripts
- Time-aware validation/backtesting
- Feature engineering and explainability
- Kaggle submission file
- Kaggle leaderboard screenshot
- Pipeline diagram
- Short report covering limitations and improvement plan

### 2. American Sign Language Detection

Folder: `usecase_2_asl_detection/`

Goal: Collect a custom ASL image dataset, annotate hand bounding boxes, train an object detection model, and demonstrate live webcam inference.

Expected outputs:

- Custom dataset summary
- Annotation samples
- Trained detection model
- Evaluation metrics
- Webcam demo script
- Architecture diagram
- Live demo instructions

### 3. Hybrid RAG for NCERT Physics PDF

Folder: `usecase_3_hybrid_rag/`

Goal: Build a grounded question-answering application over the NCERT Class 12 Physics Part 1 PDF using vector retrieval, graph retrieval, semantic search, and keyword search.

Expected outputs:

- PDF ingestion pipeline
- Vector database/index
- Graph database or graph representation
- Keyword search index
- Hybrid retrieval and answer-generation app
- Citations and retrieval evidence display
- Architecture diagram
- Live demo instructions

## Quick Start

Each use case has its own README with setup and execution instructions:

- `usecase_1_forecasting/README.md`
- `usecase_2_asl_detection/README.md`
- `usecase_3_hybrid_rag/README.md`

## Final Submission Checklist

See `SUBMISSION.md` for the detailed checklist used to verify the final package before submission.

## Disclosure

External resources, pretrained models, tutorials, public code references, and AI-assisted work are documented in `DISCLOSURE.md`.

