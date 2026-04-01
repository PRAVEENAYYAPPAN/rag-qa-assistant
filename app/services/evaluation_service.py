"""
Evaluation service – precision/recall-based retrieval quality assessment.

Metrics computed (per-sample and aggregate):
  - Precision@K
  - Recall@K
  - F1@K
  - Mean Reciprocal Rank (MRR)
  - Normalised Discounted Cumulative Gain (NDCG@K)
"""

from __future__ import annotations

import math
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import EvalRequest, EvalResponse, EvalMetrics
from app.services.vector_store import get_vector_store

log = get_logger(__name__)
settings = get_settings()


class EvaluationService:
    """Offline retrieval evaluation with standard IR metrics."""

    def __init__(self) -> None:
        self._store = get_vector_store()

    def evaluate(self, request: EvalRequest) -> EvalResponse:
        """Run evaluation on a set of labelled query samples."""
        k = request.k
        per_sample: list[dict[str, Any]] = []
        timings: list[float] = []

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0

        for sample in request.samples:
            t0 = time.perf_counter()
            results = self._store.search(sample.question, top_k=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)

            retrieved_ids = [
                r["metadata"].get("id", r["content"][:64]) for r in results
            ]
            relevant_set = set(sample.relevant_doc_ids)

            prec = self._precision_at_k(retrieved_ids, relevant_set, k)
            rec = self._recall_at_k(retrieved_ids, relevant_set, k)
            f1 = self._f1(prec, rec)
            mrr = self._mrr(retrieved_ids, relevant_set)
            ndcg = self._ndcg_at_k(retrieved_ids, relevant_set, k)

            total_precision += prec
            total_recall += rec
            total_f1 += f1
            total_mrr += mrr
            total_ndcg += ndcg

            per_sample.append(
                {
                    "question": sample.question,
                    "precision_at_k": round(prec, 4),
                    "recall_at_k": round(rec, 4),
                    "f1_at_k": round(f1, 4),
                    "mrr": round(mrr, 4),
                    "ndcg": round(ndcg, 4),
                    "retrieval_time_ms": round(elapsed_ms, 2),
                    "retrieved_ids": retrieved_ids[:k],
                    "relevant_ids": list(relevant_set),
                }
            )

        n = len(request.samples) or 1
        metrics = EvalMetrics(
            precision_at_k=round(total_precision / n, 4),
            recall_at_k=round(total_recall / n, 4),
            f1_at_k=round(total_f1 / n, 4),
            mrr=round(total_mrr / n, 4),
            ndcg=round(total_ndcg / n, 4),
            avg_retrieval_time_ms=round(sum(timings) / len(timings), 2),
            total_samples=len(request.samples),
        )

        summary = (
            f"Evaluated {len(request.samples)} samples at K={k}: "
            f"P@K={metrics.precision_at_k:.3f}, "
            f"R@K={metrics.recall_at_k:.3f}, "
            f"F1={metrics.f1_at_k:.3f}, "
            f"MRR={metrics.mrr:.3f}, "
            f"NDCG={metrics.ndcg:.3f}"
        )
        log.info(summary)

        return EvalResponse(metrics=metrics, per_sample=per_sample, summary=summary)

    # ── Metric helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not retrieved:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for r in top_k if r in relevant)
        return hits / len(top_k)

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 1.0   # undefined → 1.0 by convention
        top_k = retrieved[:k]
        hits = sum(1 for r in top_k if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        for i, r in enumerate(retrieved, 1):
            if r in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        def dcg(items: list[str]) -> float:
            total = 0.0
            for i, item in enumerate(items[:k], 1):
                if item in relevant:
                    total += 1.0 / math.log2(i + 1)
            return total

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        if idcg == 0:
            return 0.0
        return dcg(retrieved) / idcg
