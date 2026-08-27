"""
rag/pipeline.py

The orchestration layer — the only file that imports from all other modules.
Every step is logged via logger, not trace methods.

Pipeline steps:
    1. Retrieve  — hybrid dense + BM25 search
    2. Rerank    — cross-encoder second pass
    3. Prompt    — build grounded context block
    4. Generate  — call LLM
    5. Validate  — hallucination guardrail check
    6. Decide    — RAGAS-based decision layer
                   (Accept/Retry/Fallback/Reject)
    7. Return    — structured response with trace metadata

P0-002:
    RETRY and FALLBACK decisions are actively executed instead of
    merely being logged.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from configs.settings import settings
from retrieval.retriever import HybridRetriever
from retrieval.reranker import rerank
from rag.prompt import (
    build_context_block,
    RAG_PROMPT,
    SYSTEM_MESSAGE,
)
from guardrails.hallucination_filter import check_response
from observability.logger import get_logger
from rag.query_rewriter import rewrite_with_logging
from rag.decision_layer import DecisionLayer
from evaluation.ragas_eval import run_ragas_evaluation


logger = get_logger("rag.pipeline")


@dataclass
class RAGSession:
    session_id: str
    retriever: HybridRetriever
    all_chunks: List[Document] = field(default_factory=list)


# ── LLM factory ────────────────────────────────────────────────────────────────


def _get_llm(
    model: Optional[str] = None,
) -> ChatOpenAI:
    """
    Create an LLM instance.

    If model is provided, it overrides the default configured model.
    This is used by the FALLBACK decision.
    """

    model = model or settings.LLM_MODEL

    return ChatOpenAI(
        model=model,
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# ── Response model ────────────────────────────────────────────────────────────


@dataclass
class RAGResponse:
    answer: str
    is_grounded: bool
    grounding_score: float
    sources: List[Dict]
    trace_id: str
    refused: bool = False
    decision_metadata: Optional[Dict] = None


# ── Source helper ──────────────────────────────────────────────────────────────


def _build_sources(
    reranked_docs: List[Document],
) -> List[Dict]:
    """
    Build the source payload returned to the API.
    """

    return [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "chunk_text": doc.page_content,
        }
        for doc in reranked_docs
    ]


# ── Answer generation ─────────────────────────────────────────────────────────


def _generate_answer(
    question: str,
    docs: List[Document],
    model: Optional[str] = None,
) -> tuple[str, bool, float, str]:
    """
    Build context, call the LLM, and run the hallucination filter.

    model:
        Optional model override used by FALLBACK.
    """

    context_block = build_context_block(
        docs
    )

    if (
        not context_block
        or len(context_block.strip()) < 10
    ):
        logger.warning(
            f"Very small context block "
            f"({len(context_block)} chars) "
            f"for question: {question[:80]}"
        )

    prompt_text = RAG_PROMPT.format(
        context=context_block,
        question=question,
    )

    llm = _get_llm(model)

    messages = [
        SystemMessage(
            content=SYSTEM_MESSAGE
        ),
        HumanMessage(
            content=prompt_text
        ),
    ]

    callbacks = []

    if (
        settings.LANGCHAIN_TRACING
        and settings.LANGSMITH_API_KEY
    ):
        from langchain_core.tracers import (
            LangChainTracer,
        )

        callbacks.append(
            LangChainTracer(
                project_name=settings.LANGSMITH_PROJECT
            )
        )

    response = llm.invoke(
        messages,
        config={
            "callbacks": callbacks
        }
        if callbacks
        else {},
    )

    raw_answer = response.content

    logger.debug(
        f"LLM generated "
        f"{len(raw_answer)} chars of answer"
    )

    (
        is_grounded,
        grounding_score,
        final_answer,
    ) = check_response(
        raw_answer,
        docs,
    )

    logger.debug(
        f"Grounding check: "
        f"{is_grounded}, "
        f"score={grounding_score:.2f}"
    )

    if not is_grounded:
        logger.warning(
            "Answer not grounded in context"
        )

    return (
        final_answer,
        is_grounded,
        grounding_score,
        context_block,
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_pipeline(
    session: RAGSession,
    question: str,
    eval_scores: Optional[
        Dict[str, float]
    ] = None,
    trace_id: Optional[str] = None,
) -> RAGResponse:
    """
    Execute the complete RAG pipeline.

    P0-002 behavior:

    RETRY
        Increase retrieval breadth, retrieve again,
        rerank again, regenerate the answer, and evaluate
        the new answer through the decision layer.

    FALLBACK
        Switch to settings.FALLBACK_LLM_MODEL and regenerate
        the answer using the existing retrieved context.
    """

    trace_id = (
        trace_id
        or str(uuid.uuid4())
    )

    # ── Query rewriting ──────────────────────────────────────────────────────

    (
        rewritten_query,
        rewrite_metadata,
    ) = rewrite_with_logging(
        question
    )

    logger.debug(
        f"Query rewrite: "
        f"ambiguity="
        f"{rewrite_metadata.get('ambiguity_score'):.2f}, "
        f"was_rewritten="
        f"{rewrite_metadata.get('was_rewritten')}"
    )

    try:

        # ── Step 1: Initial retrieval ────────────────────────────────────────

        docs, scores = (
            session.retriever.retrieve(
                rewritten_query
            )
        )

        if not docs:

            logger.warning(
                f"[{session.session_id}] "
                f"No docs retrieved for: "
                f"{question[:80]}"
            )

            return RAGResponse(
                answer=settings.REFUSAL_MESSAGE,
                is_grounded=False,
                grounding_score=0.0,
                sources=[],
                trace_id=trace_id,
                refused=True,
                decision_metadata=None,
            )

        logger.debug(
            f"Retrieved {len(docs)} docs"
        )

        # ── Step 2: Rerank ───────────────────────────────────────────────────

        reranked_docs = rerank(
            question,
            docs,
            top_k=settings.TOP_K_RERANK,
        )

        logger.debug(
            f"Reranked to "
            f"{len(reranked_docs)} docs"
        )

        # ── Step 3-5: Generate + validate ───────────────────────────────────

        (
            final_answer,
            is_grounded,
            grounding_score,
            _context_block,
        ) = _generate_answer(
            question,
            reranked_docs,
        )

        decision_metadata = None

        # ── Step 6: Decision Layer ───────────────────────────────────────────

        if eval_scores:

            try:

                decision_layer = (
                    DecisionLayer()
                )

                max_attempts = (
                    settings.RAGAS_MAX_RETRY_ATTEMPTS
                )

                attempt = 1

                while True:

                    # ── Evaluate current answer ─────────────────────────────

                    (
                        decided_answer,
                        current_decision,
                        should_return,
                    ) = (
                        decision_layer.evaluate_and_decide(
                            answer=final_answer,
                            scores=eval_scores,
                            attempt=attempt,
                            max_attempts=max_attempts,
                        )
                    )

                    decision_metadata = (
                        current_decision
                    )

                    decision_type = (
                        current_decision[
                            "decision_type"
                        ]
                    )

                    logger.info(
                        f"Decision: "
                        f"{decision_type} | "
                        f"Attempt: {attempt} | "
                        f"Faithfulness: "
                        f"{eval_scores.get('faithfulness', 0.0):.2f} | "
                        f"Relevance: "
                        f"{eval_scores.get('answer_relevancy', 0.0):.2f}"
                    )

                    # ── ACCEPT ──────────────────────────────────────────────

                    if decision_type == "ACCEPT":

                        final_answer = (
                            decided_answer
                        )

                        logger.info(
                            "Answer accepted by decision layer"
                        )

                        break

                    # ── REJECT ──────────────────────────────────────────────

                    if decision_type == "REJECT":

                        logger.warning(
                            f"Answer rejected "
                            f"after {attempt} attempts"
                        )

                        return RAGResponse(
                            answer=decided_answer,
                            is_grounded=False,
                            grounding_score=(
                                grounding_score
                                * 0.5
                            ),
                            sources=_build_sources(
                                reranked_docs
                            ),
                            trace_id=trace_id,
                            refused=True,
                            decision_metadata=(
                                decision_metadata
                            ),
                        )

                    # ── RETRY ────────────────────────────────────────────────

                    if decision_type == "RETRY":

                        retry_config = (
                            decision_layer.get_retry_config(
                                attempt=attempt
                            )
                        )

                        logger.info(
                            f"RETRY triggered | "
                            f"top_k="
                            f"{retry_config['top_k']} | "
                            f"threshold="
                            f"{retry_config['score_threshold']} | "
                            f"max_context="
                            f"{retry_config['max_context_chunks']}"
                        )

                        # Retrieve more documents.

                        (
                            retry_docs,
                            retry_scores,
                        ) = (
                            session.retriever.retrieve(
                                rewritten_query,
                                top_k=retry_config[
                                    "top_k"
                                ],
                                score_threshold=(
                                    retry_config[
                                        "score_threshold"
                                    ]
                                ),
                            )
                        )

                        if not retry_docs:

                            logger.warning(
                                "RETRY retrieval returned "
                                "no documents"
                            )

                            return RAGResponse(
                                answer=(
                                    settings.REFUSAL_MESSAGE
                                ),
                                is_grounded=False,
                                grounding_score=0.0,
                                sources=_build_sources(
                                    reranked_docs
                                ),
                                trace_id=trace_id,
                                refused=True,
                                decision_metadata=(
                                    decision_metadata
                                ),
                            )

                        # Limit context size before reranking.

                        retry_docs = retry_docs[
                            : retry_config[
                                "max_context_chunks"
                            ]
                        ]

                        # Rerank the expanded result set.

                        reranked_docs = rerank(
                            question,
                            retry_docs,
                            top_k=min(
                                settings.TOP_K_RERANK,
                                len(retry_docs),
                            ),
                        )

                        logger.info(
                            f"RETRY retrieved "
                            f"{len(retry_docs)} docs and "
                            f"reranked to "
                            f"{len(reranked_docs)} docs"
                        )

                        # Generate a new answer.

                        (
                            final_answer,
                            is_grounded,
                            grounding_score,
                            _context_block,
                        ) = _generate_answer(
                            question,
                            reranked_docs,
                        )

                        # ── Re-evaluate the NEW answer with RAGAS ─────────────
                        # The previous eval_scores belong to the previous answer.
                        # After RETRY we must evaluate the newly generated answer
                        # against the newly retrieved context before making the
                        # next decision.
                        retry_contexts = [
                            doc.page_content
                            for doc in reranked_docs
                            if doc.page_content
                        ]

                        if not retry_contexts:
                            logger.error(
                                "RAGAS re-evaluation skipped: "
                                "RETRY produced no usable contexts"
                            )

                            decision_metadata = dict(
                                decision_metadata or {}
                            )
                            decision_metadata[
                                "reevaluation_error"
                            ] = "No usable contexts after RETRY"

                            return RAGResponse(
                                answer=final_answer,
                                is_grounded=is_grounded,
                                grounding_score=grounding_score,
                                sources=_build_sources(reranked_docs),
                                trace_id=trace_id,
                                refused=not is_grounded,
                                decision_metadata=decision_metadata,
                            )

                        retry_test_case = {
                            "question": question,
                            "answer": final_answer,
                            "contexts": retry_contexts,
                        }

                        try:
                            retry_eval_result = run_ragas_evaluation(
                                [retry_test_case]
                            )
                        except Exception as eval_error:
                            logger.error(
                                f"RAGAS re-evaluation failed after RETRY: "
                                f"{eval_error}"
                            )

                            decision_metadata = dict(
                                decision_metadata or {}
                            )
                            decision_metadata[
                                "reevaluation_error"
                            ] = str(eval_error)

                            return RAGResponse(
                                answer=final_answer,
                                is_grounded=is_grounded,
                                grounding_score=grounding_score,
                                sources=_build_sources(reranked_docs),
                                trace_id=trace_id,
                                refused=not is_grounded,
                                decision_metadata=decision_metadata,
                            )

                        if (
                            retry_eval_result.get("error")
                            or retry_eval_result.get("evaluated_cases", 0) == 0
                            or retry_eval_result.get("faithfulness") is None
                            or retry_eval_result.get("answer_relevancy") is None
                        ):
                            error_message = retry_eval_result.get(
                                "error",
                                "RAGAS returned no complete evaluation scores",
                            )

                            logger.error(
                                "RAGAS re-evaluation returned invalid results: "
                                f"{error_message}"
                            )

                            decision_metadata = dict(
                                decision_metadata or {}
                            )
                            decision_metadata[
                                "reevaluation_error"
                            ] = error_message

                            return RAGResponse(
                                answer=final_answer,
                                is_grounded=is_grounded,
                                grounding_score=grounding_score,
                                sources=_build_sources(reranked_docs),
                                trace_id=trace_id,
                                refused=not is_grounded,
                                decision_metadata=decision_metadata,
                            )

                        eval_scores = {
                            "faithfulness": float(
                                retry_eval_result["faithfulness"]
                            ),
                            "answer_relevancy": float(
                                retry_eval_result["answer_relevancy"]
                            ),
                        }

                        logger.info(
                            "RAGAS re-evaluation after RETRY: "
                            f"Faithfulness: {eval_scores['faithfulness']:.2f} | "
                            f"Relevance: {eval_scores['answer_relevancy']:.2f}"
                        )

                        # Move to the next decision attempt. The next iteration
                        # now evaluates the NEW answer using its NEW RAGAS scores.
                        attempt += 1

                        if attempt > max_attempts:

                            attempt = (
                                max_attempts
                            )

                        continue

                    # ── FALLBACK ─────────────────────────────────────────────

                    if decision_type == "FALLBACK":

                        fallback_model = (
                            decision_layer.get_fallback_model()
                        )

                        if not fallback_model:

                            logger.warning(
                                "FALLBACK requested but "
                                "no fallback model is configured"
                            )

                            return RAGResponse(
                                answer=(
                                    settings.REFUSAL_MESSAGE
                                ),
                                is_grounded=False,
                                grounding_score=0.0,
                                sources=_build_sources(
                                    reranked_docs
                                ),
                                trace_id=trace_id,
                                refused=True,
                                decision_metadata=(
                                    decision_metadata
                                ),
                            )

                        logger.info(
                            f"FALLBACK triggered. "
                            f"Switching to model: "
                            f"{fallback_model}"
                        )

                        # Generate using the fallback model.

                        (
                            final_answer,
                            is_grounded,
                            grounding_score,
                            _context_block,
                        ) = _generate_answer(
                            question,
                            reranked_docs,
                            model=fallback_model,
                        )

                        decision_metadata = dict(
                            decision_metadata
                        )

                        decision_metadata[
                            "fallback_model"
                        ] = fallback_model

                        decision_metadata[
                            "fallback_executed"
                        ] = True

                        logger.info(
                            "Fallback model successfully "
                            "generated a new answer"
                        )

                        # Fallback is intentionally one-shot.
                        # We do not repeatedly switch models.

                        break

                    # ── Unknown decision ────────────────────────────────────

                    logger.warning(
                        f"Unknown decision type: "
                        f"{decision_type}"
                    )

                    final_answer = (
                        decided_answer
                    )

                    break

            except Exception as e:

                logger.error(
                    f"Decision layer error "
                    f"(returning current answer): {e}"
                )

                decision_metadata = None

        # ── Step 7: Build response ───────────────────────────────────────────

        sources = _build_sources(
            reranked_docs
        )

        return RAGResponse(
            answer=final_answer,
            is_grounded=is_grounded,
            grounding_score=round(
                grounding_score,
                3,
            ),
            sources=sources,
            trace_id=trace_id,
            refused=not is_grounded,
            decision_metadata=decision_metadata,
        )

    except Exception as exc:

        logger.error(
            f"Pipeline error for session "
            f"{session.session_id}: {exc}"
        )

        raise