"""
Script: evaluate_retrieval.py
Run a precision/recall evaluation against the loaded knowledge base.

Usage:
    python scripts/evaluate_retrieval.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.evaluation_service import EvaluationService
from app.models.schemas import EvalRequest, EvalSample
from app.core.logging import setup_logging

setup_logging()

# ── Define evaluation samples ────────────────────────────────────────────────
SAMPLES = [
    EvalSample(
        question="What is machine learning?",
        relevant_doc_ids=[],
    ),
    EvalSample(
        question="How does a neural network work?",
        relevant_doc_ids=[],
    ),
    EvalSample(
        question="What are transformers in deep learning?",
        relevant_doc_ids=[],
    ),
    EvalSample(
        question="Explain retrieval augmented generation",
        relevant_doc_ids=[],
    ),
    EvalSample(
        question="What is the difference between precision and recall?",
        relevant_doc_ids=[],
    ),
]

if __name__ == "__main__":
    svc = EvaluationService()
    request = EvalRequest(samples=SAMPLES, k=5)
    report = svc.evaluate(request)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(report.summary)
    print("\n── Aggregate Metrics ──")
    print(json.dumps(report.metrics.model_dump(), indent=2))
    print("\n── Per-Sample Results ──")
    for s in report.per_sample:
        print(f"  Q: {s['question'][:60]}")
        print(
            f"     P@K={s['precision_at_k']:.3f}  "
            f"R@K={s['recall_at_k']:.3f}  "
            f"F1={s['f1_at_k']:.3f}  "
            f"MRR={s['mrr']:.3f}  "
            f"t={s['retrieval_time_ms']:.1f}ms"
        )
    print("=" * 60)
