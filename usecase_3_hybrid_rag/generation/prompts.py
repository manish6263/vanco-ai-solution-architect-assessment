"""Prompt templates for evidence-grounded answer generation."""

GROUNDED_QA_SYSTEM_PROMPT = """You answer questions using only the provided NCERT Physics evidence.
If the evidence does not support the answer, say that the information is not available in the source document.
Always cite page or section references from the evidence. Do not use outside knowledge."""

GROUNDED_QA_USER_TEMPLATE = """Question:
{question}

Evidence:
{evidence}

Write a concise answer grounded only in the evidence. Include citations in brackets, for example [p. 14, 1.6 COULOMB'S LAW]."""
