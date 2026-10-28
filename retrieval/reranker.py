"""
retrieval/reranker.py

Cross-encoder reranking as a second-pass filter.

Why reranking is essential:
  - Embedding models encode query and document independently (bi-encoders).
    They are fast but lose fine-grained query-document interaction signals.
  - A cross-encoder scores the (query, document) pair jointly, giving far
    more accurate relevance scores at the cost of speed.
  - Running a cross-encoder on 20 candidates → keeping top 5 is the standard
    production pattern (retrieve-then-rerank).

This module uses a local HuggingFace cross-encoder (ms-marco-MiniLM-L-6-v2)
which runs locally without an additional API cost.

To switch to a hosted reranker (Cohere, Jina), replace _rerank_local() with
an API call and update configs/settings.py.
"""

import time
from typing import List, Tuple

from langchain_core.documents import Document

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("retrieval.reranker")

# Lazy-loaded to avoid import cost when reranker is disabled
_cross_encoder = None


def _load_model():
    """
    Load the cross-encoder lazily.

    The model is cached in the module-level _cross_encoder variable so that
    it should only be loaded once per running Python process.
    """
    global _cross_encoder

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            f"Loading reranker model: {settings.RERANKER_MODEL}"
        )

        load_start_ns = time.perf_counter_ns()

        _cross_encoder = CrossEncoder(settings.RERANKER_MODEL)

        load_latency_ms = (
            time.perf_counter_ns() - load_start_ns
        ) / 1_000_000

        logger.info(
            f"Reranker model loaded in {load_latency_ms:.1f} ms"
        )

    return _cross_encoder


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = settings.TOP_K_RERANK,
) -> List[Document]:
    """
    Rerank documents using a cross-encoder and return the top_k.

    If reranking is disabled in settings or the document list is already
    small (<= top_k), returns the input list unchanged.
    """
    if not settings.RERANKER_ENABLED:
        logger.info("Reranker disabled, skipping")
        return docs[:top_k]

    if len(docs) <= top_k:
        logger.info(
            f"Only {len(docs)} docs, skipping rerank (top_k={top_k})"
        )
        return docs

    rerank_start_ns = time.perf_counter_ns()

    try:
        # -------------------------------------------------------------
        # Model loading
        # -------------------------------------------------------------
        model = _load_model()

        # -------------------------------------------------------------
        # Prepare query-document pairs
        # -------------------------------------------------------------
        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        # -------------------------------------------------------------
        # Cross-encoder inference
        # -------------------------------------------------------------
        prediction_start_ns = time.perf_counter_ns()

        scores = model.predict(pairs)

        prediction_latency_ms = (
            time.perf_counter_ns() - prediction_start_ns
        ) / 1_000_000

        # -------------------------------------------------------------
        # Sort documents by cross-encoder relevance score
        # -------------------------------------------------------------
        ranked: List[Tuple[Document, float]] = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = [
            doc
            for doc, _ in ranked[:top_k]
        ]

        top_scores = [
            score
            for _, score in ranked[:top_k]
        ]

        total_rerank_latency_ms = (
            time.perf_counter_ns() - rerank_start_ns
        ) / 1_000_000

        logger.info(
            f"Reranked {len(docs)} → {len(top_docs)} docs. "
            f"Top score: {round(float(top_scores[0]), 3)}"
        )

        logger.info(
            f"Reranker timing | "
            f"prediction={prediction_latency_ms:.1f} ms | "
            f"total={total_rerank_latency_ms:.1f} ms"
        )

        return top_docs

    except Exception as exc:
        total_rerank_latency_ms = (
            time.perf_counter_ns() - rerank_start_ns
        ) / 1_000_000

        logger.warning(
            f"Reranker failed ({exc}), "
            f"falling back to original order. "
            f"Elapsed={total_rerank_latency_ms:.1f} ms"
        )

        return docs[:top_k]