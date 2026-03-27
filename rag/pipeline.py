"""
rag/pipeline.py

The orchestration layer — the only file that imports from all other modules.
Every step is logged via RAGTrace so failures are fully debuggable.

Pipeline steps:
    1. Retrieve  — hybrid dense + BM25 search
    2. Rerank    — cross-encoder second pass
    3. Prompt    — build grounded context block
    4. Generate  — call LLM (Groq free tier)
    5. Validate  — hallucination guardrail check
    6. Return    — structured response with trace metadata
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from configs import settings
from retrieval.retriever import HybridRetriever
from retrieval.reranker import rerank
from rag.prompt import build_context_block, RAG_PROMPT, SYSTEM_MESSAGE
from guardrails.hallucination_filter import check_response
from observability.logger import get_logger, RAGTrace

logger = get_logger("rag.pipeline")


# ── Session object ─────────────────────────────────────────────────────────────

@dataclass
class RAGSession:
    session_id:  str
    retriever:   HybridRetriever
    all_chunks:  List[Document] = field(default_factory=list)


# ── LLM factory (Groq) ────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# ── Response model ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer:          str
    is_grounded:     bool
    grounding_score: float
    sources:         List[Dict]
    trace_id:        str
    refused:         bool = False


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(session: RAGSession, question: str) -> RAGResponse:
    trace = RAGTrace(session_id=session.session_id, question=question)

    try:
        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        docs, scores = session.retriever.retrieve(question)

        if not docs:
            logger.warning(f"[{session.session_id}] No docs retrieved for: {question[:80]}")
            trace.set_refused()
            trace.finish()
            return RAGResponse(
                answer=settings.REFUSAL_MESSAGE,
                is_grounded=False,
                grounding_score=0.0,
                sources=[],
                trace_id=trace.trace_id,
                refused=True,
            )

        trace.set_retrieval(docs, scores)

        # ── Step 2: Rerank ────────────────────────────────────────────────────
        reranked_docs = rerank(question, docs, top_k=settings.TOP_K_RERANK)
        trace.set_reranked(reranked_docs)

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        context_block = build_context_block(reranked_docs)
        prompt_text   = RAG_PROMPT.format(context=context_block, question=question)
        trace.set_prompt(prompt_text)

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

        # ── Step 5: Validate ──────────────────────────────────────────────────
        is_grounded, grounding_score, final_answer = check_response(raw_answer, reranked_docs)
        trace.set_response(final_answer, grounding_score)

        if not is_grounded:
            trace.set_refused()

        # ── Step 6: Build sources payload ─────────────────────────────────────
        sources = [
            {
                "chunk_id":   doc.metadata.get("chunk_id"),
                "source":     doc.metadata.get("source"),
                "page":       doc.metadata.get("page"),
                "chunk_text": doc.page_content,
            }
            for doc in reranked_docs
        ]

        trace.finish()

        return RAGResponse(
            answer=final_answer,
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 3),
            sources=sources,
            trace_id=trace.trace_id,
            refused=not is_grounded,
        )

    except Exception as exc:
        trace.set_error(str(exc))
        trace.finish()
        logger.error(f"Pipeline error for session {session.session_id}: {exc}")
        raise