"""
rag/query_rewriter.py

Query rewriting layer for DocMind RAG system.
Converts vague/ambiguous user queries into specific, detailed queries for improved retrieval.
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("rag.query_rewriter")


# ─────────────────────────────────────────────────────────────────────────────
# AMBIGUITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_ambiguity(query: str) -> float:
    """
    Detect how ambiguous/vague a query is.
    """
    if not query or len(query.strip()) == 0:
        return 1.0
    
    query_lower = query.lower().strip()
    ambiguity_score = 0.0
    
    # ── Signal 1: Query length ────────────────────────────────────────────
    if len(query_lower) < settings.QUERY_MIN_LENGTH:
        ambiguity_score += 0.3
    elif len(query_lower) < 15:
        ambiguity_score += 0.15
    
    # ── Signal 2: Vague words ─────────────────────────────────────────────
    vague_words = [
        "something", "anything", "things", "stuff", "what",
        "about", "like", "tell me", "give me", "show me",
        "how", "why", "so", "etc", "other",
        "generally", "usually", "sometimes", "anyway",
        "it", "its",
        "this", "that", "they", "them", "these", "those", "their"
    ]
    vague_count = sum(1 for word in vague_words if f" {word} " in f" {query_lower} ")
    if vague_count > 0:
        ambiguity_score += min(0.3, vague_count * 0.1)
    
    # ── Signal 3: Question structure ───────────────────────────────────────
    question_starters = ["what", "how", "why", "can you", "could you", "do you"]
    for starter in question_starters:
        if query_lower.startswith(starter):
            words_after = len(query_lower.split())
            if words_after <= 4:
                ambiguity_score += 0.25
            break
    
    # ── Signal 4: Specificity indicators ──────────────────────────────────
    specific_indicators = [
        "specific", "define", "explain", "describe", "list", "show",
        "diagnosis", "treatment", "symptom", "disease",
        "algorithm", "technical", "implementation", "how does"
    ]
    specific_count = sum(1 for indicator in specific_indicators if indicator in query_lower)
    if specific_count > 0:
        ambiguity_score -= min(0.2, specific_count * 0.05)
    
    ambiguity_score = max(0.0, min(1.0, ambiguity_score))
    logger.debug(f"Query ambiguity: '{query[:50]}...' = {ambiguity_score:.2f}")
    return ambiguity_score


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITING
# ─────────────────────────────────────────────────────────────────────────────

def _get_rewrite_llm() -> ChatOpenAI:
    """
    Get LLM instance for query rewriting using LLM_API_KEY.
    """
    api_key = settings.LLM_API_KEY or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is missing or empty for query rewriting.")

    return ChatOpenAI(
        model=settings.QUERY_REWRITE_MODEL,
        api_key=api_key,
        base_url=settings.LLM_API_BASE,
        openai_api_key=api_key,
        openai_api_base=settings.LLM_API_BASE,
        temperature=0.3,
        max_tokens=150,
    )


def rewrite_query(query: str) -> Tuple[str, Optional[str]]:
    """
    Rewrite a vague query to be more specific and detailed.
    """
    system_prompt = """You are a query optimization assistant. Your task is to rewrite vague or ambiguous user queries into specific, detailed queries that will improve document retrieval.

Rules:
1. Maintain the original intent - do not change what the user is asking for
2. Add specifics and context - expand the query with relevant details
3. Keep it concise - one sentence maximum, under 150 characters
4. No preamble - just output the rewritten query, nothing else
5. If the query is already clear and specific, return it unchanged"""

    user_prompt = f"""Rewrite this query to be more specific and detailed for document retrieval:

Query: {query}

Rewritten query:"""

    try:
        llm = _get_rewrite_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        
        if not rewritten or len(rewritten) < 3:
            error_msg = f"LLM returned invalid rewrite: '{rewritten}'"
            logger.warning(f"Query rewrite fallback: {error_msg}")
            return query, error_msg
        
        if len(rewritten) > settings.QUERY_MAX_REWRITE_LENGTH:
            rewritten = rewritten[:settings.QUERY_MAX_REWRITE_LENGTH].rsplit(' ', 1)[0] + "?"
        
        logger.debug(f"Successfully rewrote: '{query[:40]}...' → '{rewritten[:40]}...'")
        return rewritten, None
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {str(e)}"
        logger.warning(f"Query rewrite skipped due to {error_type}. Using original query.")
        return query, error_msg


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL REWRITING
# ─────────────────────────────────────────────────────────────────────────────

def should_rewrite(query: str, threshold: float = settings.QUERY_AMBIGUITY_THRESHOLD) -> bool:
    """
    Decide whether a query needs rewriting.
    """
    ambiguity_score = detect_ambiguity(query)
    
    if ambiguity_score > threshold:
        logger.debug(f"Query needs rewrite: ambiguity {ambiguity_score:.2f} > {threshold}")
        return True
    
    if len(query.strip()) < settings.QUERY_MIN_LENGTH:
        logger.debug(f"Query needs rewrite: too short ({len(query.strip())} < {settings.QUERY_MIN_LENGTH})")
        return True
    
    logger.debug(f"Query clear enough: ambiguity {ambiguity_score:.2f} <= {threshold}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT WITH LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_with_logging(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Rewrite query with comprehensive logging and error tracking.
    """
    start_time = time.time()
    rewrite_id = str(uuid.uuid4())[:8]
    
    ambiguity_score = detect_ambiguity(query)
    needs_rewrite = should_rewrite(query)
    
    rewrite_error = None
    rewrite_error_type = None
    
    if needs_rewrite:
        rewritten_query, rewrite_error = rewrite_query(query)
        was_rewritten = (rewritten_query != query and rewrite_error is None)
        
        if rewrite_error:
            rewrite_error_type = rewrite_error.split(":")[0] if ":" in rewrite_error else "Unknown"
    else:
        rewritten_query = query
        was_rewritten = False
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    metadata = {
        "rewrite_id": rewrite_id,
        "original_query": query,
        "rewritten_query": rewritten_query,
        "was_rewritten": was_rewritten,
        "ambiguity_score": round(ambiguity_score, 3),
        "timestamp": datetime.now().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2),
        "rewrite_error": rewrite_error,
        "rewrite_error_type": rewrite_error_type,
    }
    
    if was_rewritten:
        logger.info(
            f"[{rewrite_id}] Query rewritten: "
            f"'{query[:40]}...' → '{rewritten_query[:40]}...' "
            f"(ambiguity: {ambiguity_score:.2f})"
        )
    elif rewrite_error:
        logger.warning(
            f"[{rewrite_id}] Query rewrite failed ({rewrite_error_type}): "
            f"'{query[:40]}...' (ambiguity: {ambiguity_score:.2f}). "
            f"Returning original query."
        )
    else:
        logger.debug(
            f"[{rewrite_id}] Query clear enough, no rewrite needed: "
            f"'{query[:40]}...' (ambiguity: {ambiguity_score:.2f})"
        )
    
    return rewritten_query, metadata