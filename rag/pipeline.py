"""
rag/pipeline.py

The orchestration layer — the only file that imports from all other modules.
Every step is logged via logger, not trace methods.

Pipeline steps:
    1. Retrieve  — hybrid dense + BM25 search
    2. Rerank    — cross-encoder second pass
    3. Prompt    — build grounded context block
    4. Generate  — call LLM (Groq free tier)
    5. Validate  — hallucination guardrail check
    6. Decide    — RAGAS-based decision layer (Accept/Retry/Fallback)
    7. Return    — structured response with trace metadata
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from configs.settings import settings
from retrieval.retriever import HybridRetriever
from retrieval.reranker import rerank
from rag.prompt import build_context_block, RAG_PROMPT, SYSTEM_MESSAGE
from guardrails.hallucination_filter import check_response
from observability.logger import get_logger
from rag.query_rewriter import rewrite_with_logging
from rag.decision_layer import DecisionLayer

logger = get_logger("rag.pipeline")


# ── Session object ─────────────────────────────────────────────────────────────

@dataclass
class RAGSession:
    session_id:  str
    retriever:   HybridRetriever
    all_chunks:  List[Document] = field(default_factory=list)


# ── LLM factory (Groq) ────────────────────────────────────────────────────

def _get_llm(model: Optional[str] = None) -> ChatOpenAI:
    """
    Get LLM instance. Supports overriding model for fallback scenarios.
    
    Args:
        model: Optional model override (for fallback). If None, uses settings.LLM_MODEL
    
    Returns:
        ChatOpenAI instance
    """
    model = model or settings.LLM_MODEL
    return ChatOpenAI(
        model=model,
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# ── Response model ─────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer:          str
    is_grounded:     bool
    grounding_score: float
    sources:         List[Dict]
    trace_id:        str
    refused:         bool = False
    decision_metadata: Optional[Dict] = None  # From decision layer


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    session: RAGSession, 
    question: str,
    eval_scores: Optional[Dict[str, float]] = None,
) -> RAGResponse:
    """
    Full RAG pipeline with integrated decision layer.
    
    Args:
        session: RAGSession with retriever
        question: User question
        eval_scores: Optional RAGAS scores (faithfulness, answer_relevancy)
                    If provided, triggers decision layer logic
    
    Returns:
        RAGResponse with answer, grounding score, sources, and decision metadata
    """
    trace_id = str(session.session_id)
    
    # ─── Query Rewriting ──────────────────────────────────────────────────────
    rewritten_query, rewrite_metadata = rewrite_with_logging(question)
    
    logger.debug(
        f"Query rewrite: ambiguity={rewrite_metadata.get('ambiguity_score'):.2f}, "
        f"was_rewritten={rewrite_metadata.get('was_rewritten')}"
    )

    try:
        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        docs, scores = session.retriever.retrieve(rewritten_query)

        if not docs:
            logger.warning(f"[{session.session_id}] No docs retrieved for: {question[:80]}")
            return RAGResponse(
                answer=settings.REFUSAL_MESSAGE,
                is_grounded=False,
                grounding_score=0.0,
                sources=[],
                trace_id=trace_id,
                refused=True,
                decision_metadata=None,
            )

        logger.debug(f"Retrieved {len(docs)} docs")

        # ── Step 2: Rerank ────────────────────────────────────────────────────
        reranked_docs = rerank(question, docs, top_k=settings.TOP_K_RERANK)
        logger.debug(f"Reranked to {len(reranked_docs)} docs")

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        context_block = build_context_block(reranked_docs)
        
        if not context_block or len(context_block.strip()) < 10:
            logger.warning(
                f"Very small context block ({len(context_block)} chars) for question: {question[:80]}"
            )
        
        prompt_text = RAG_PROMPT.format(context=context_block, question=question)

        # ── Step 4: Generate (Groq) ───────────────────────────────────────────
        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=prompt_text),
        ]

        callbacks = []
        if settings.LANGCHAIN_TRACING and settings.LANGSMITH_API_KEY:
            from langchain_core.tracers import LangChainTracer
            callbacks.append(LangChainTracer(project_name=settings.LANGSMITH_PROJECT))

        response   = llm.invoke(messages, config={"callbacks": callbacks} if callbacks else {})
        raw_answer = response.content

        logger.debug(f"LLM generated {len(raw_answer)} chars of answer")

        # ── Step 5: Validate (Hallucination Filter) ──────────────────────────
        is_grounded, grounding_score, final_answer = check_response(raw_answer, reranked_docs)
        logger.debug(f"Grounding check: {is_grounded}, score={grounding_score:.2f}")

        if not is_grounded:
            logger.warning("Answer not grounded in context")

        # ── Step 6: Decision Layer (RAGAS-based Accept/Retry/Fallback) ───────
        decision_metadata = None
        
        # Only run decision layer if RAGAS scores provided
        if eval_scores:
            try:
                decision_layer = DecisionLayer()
                answer, decision_metadata, should_return = decision_layer.evaluate_and_decide(
                    answer=final_answer,
                    scores=eval_scores,
                    attempt=1,
                    max_attempts=settings.RAGAS_MAX_RETRY_ATTEMPTS,
                )
                
                logger.info(
                    f"Decision: {decision_metadata['decision_type']} | "
                    f"Faithfulness: {eval_scores.get('faithfulness', 0.0):.2f} | "
                    f"Relevance: {eval_scores.get('answer_relevancy', 0.0):.2f}"
                )
                
                # ❌ REJECT: Too many retries, return with low confidence
                if decision_metadata['decision_type'] == 'REJECT':
                    logger.warning(f"Answer rejected after {decision_metadata['attempt']} attempts")
                    return RAGResponse(
                        answer=final_answer,
                        is_grounded=False,
                        grounding_score=grounding_score * 0.5,
                        sources=_build_sources(reranked_docs),
                        trace_id=trace_id,
                        refused=True,
                        decision_metadata=decision_metadata,
                    )
                
                # 🔄 RETRY: Poor faithfulness, get more context
                elif decision_metadata['decision_type'] == 'RETRY':
                    logger.info(f"Retry triggered (attempt {decision_metadata['attempt']})")
                    logger.warning("RETRY decision made but retry loop not yet implemented")
                
                # 🔀 FALLBACK: Poor relevance, try better model
                elif decision_metadata['decision_type'] == 'FALLBACK':
                    logger.info("Fallback triggered")
                    logger.warning("FALLBACK decision made but model switching not yet implemented")
                
                # ✅ ACCEPT: Quality is good
                elif decision_metadata['decision_type'] == 'ACCEPT':
                    logger.info("Answer accepted by decision layer")
                    
            except Exception as e:
                logger.error(f"Decision layer error (will skip decision and return answer as-is): {e}")
                decision_metadata = None

        # ── Step 7: Build sources payload ─────────────────────────────────────
        sources = _build_sources(reranked_docs)

        return RAGResponse(
            answer=final_answer,
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 3),
            sources=sources,
            trace_id=trace_id,
            refused=not is_grounded,
            decision_metadata=decision_metadata,
        )

    except Exception as exc:
        logger.error(f"Pipeline error for session {session.session_id}: {exc}")
        raise


def _build_sources(reranked_docs: List[Document]) -> List[Dict]:
    """Build sources payload from reranked documents."""
    return [
        {
            "chunk_id":   doc.metadata.get("chunk_id"),
            "source":     doc.metadata.get("source"),
            "page":       doc.metadata.get("page"),
            "chunk_text": doc.page_content,
        }
        for doc in reranked_docs
    ]