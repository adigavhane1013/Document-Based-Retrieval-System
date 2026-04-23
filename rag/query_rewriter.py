"""
rag/query_rewriter.py

Query rewriting layer for DocMind RAG system.
Converts vague/ambiguous user queries into specific, detailed queries for improved retrieval.

Key Concepts:
  - Ambiguity Detection: Identify queries that need rewriting
  - Query Rewriting: Use LLM to make queries specific and detailed
  - Conditional Rewriting: Only rewrite when necessary
  - Logging: Track original vs rewritten queries for analysis

Example:
  Original: "What about symptoms?"
  Rewritten: "What are the symptoms of mild cognitive impairment in early stages?"
"""

import logging
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
    
    Uses multiple signals:
      1. Query length (too short = more ambiguous)
      2. Vague words ("something", "about", "like", "what", "tell me")
      3. Missing specifics (no entities, no context)
      4. Question words that are vague
    
    Args:
        query: User query string
    
    Returns:
        Ambiguity score from 0.0 (very clear) to 1.0 (very vague)
    
    Examples:
        "What about symptoms?" → 0.85 (very ambiguous)
        "What are symptoms of MCI?" → 0.3 (fairly clear)
        "List the three types of cognitive impairment" → 0.1 (very clear)
    """
    if not query or len(query.strip()) == 0:
        return 1.0  # Empty query is maximally ambiguous
    
    query_lower = query.lower().strip()
    ambiguity_score = 0.0
    
    # ── Signal 1: Query length ────────────────────────────────────────────
    # Short queries are often vague
    if len(query_lower) < settings.QUERY_MIN_LENGTH:
        ambiguity_score += 0.3
    elif len(query_lower) < 15:
        ambiguity_score += 0.15
    
    # ── Signal 2: Vague words ─────────────────────────────────────────────
    vague_words = [
        "something", "anything", "things", "stuff", "what",
        "about", "like", "tell me", "give me", "show me",
        "how", "why", "so", "etc", "other",
        "generally", "usually", "sometimes", "anyway"
    ]
    vague_count = sum(1 for word in vague_words if f" {word} " in f" {query_lower} ")
    if vague_count > 0:
        ambiguity_score += min(0.3, vague_count * 0.1)
    
    # ── Signal 3: Question structure ───────────────────────────────────────
    # Questions starting with "what", "how", "why" without specifics
    question_starters = ["what", "how", "why", "can you", "could you", "do you"]
    for starter in question_starters:
        if query_lower.startswith(starter):
            # If it's just a question starter with few words = vague
            words_after = len(query_lower.split())
            if words_after <= 4:
                ambiguity_score += 0.25
            break
    
    # ── Signal 4: Specificity indicators ──────────────────────────────────
    # Presence of specific details reduces ambiguity
    specific_indicators = [
        "specific", "define", "explain", "describe", "list", "show",
        "define", "diagnosis", "treatment", "symptom", "disease",
        "algorithm", "technical", "implementation", "how does"
    ]
    specific_count = sum(1 for indicator in specific_indicators if indicator in query_lower)
    if specific_count > 0:
        ambiguity_score -= min(0.2, specific_count * 0.05)
    
    # Clamp to [0, 1]
    ambiguity_score = max(0.0, min(1.0, ambiguity_score))
    
    logger.debug(f"Query ambiguity: '{query[:50]}...' = {ambiguity_score:.2f}")
    return ambiguity_score


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITING
# ─────────────────────────────────────────────────────────────────────────────

def _get_rewrite_llm() -> ChatOpenAI:
    """
    Get LLM instance for query rewriting.
    Uses faster, cheaper model for this task.
    """
    return ChatOpenAI(
        model=settings.QUERY_REWRITE_MODEL,
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        temperature=0.3,  # Lower temperature for consistency
        max_tokens=150,   # Keep rewrites concise
    )


def rewrite_query(query: str) -> str:
    """
    Rewrite a vague query to be more specific and detailed.
    
    Uses LLM with a focused prompt to expand the query while maintaining intent.
    
    Args:
        query: Original vague query
    
    Returns:
        Rewritten, more specific query
    
    Raises:
        Exception: If LLM call fails
    
    Example:
        Input: "What about symptoms?"
        Output: "What are the symptoms of mild cognitive impairment in early stages?"
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
        
        # Validate the rewritten query
        if not rewritten or len(rewritten) < 3:
            logger.warning(f"LLM returned invalid rewrite: '{rewritten}'. Using original.")
            return query
        
        # Ensure it's not too long
        if len(rewritten) > settings.QUERY_MAX_REWRITE_LENGTH:
            rewritten = rewritten[:settings.QUERY_MAX_REWRITE_LENGTH].rsplit(' ', 1)[0] + "?"
        
        logger.debug(f"Rewrote: '{query}' → '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}. Using original query.")
        return query


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL REWRITING
# ─────────────────────────────────────────────────────────────────────────────

def should_rewrite(query: str, threshold: float = settings.QUERY_AMBIGUITY_THRESHOLD) -> bool:
    """
    Decide whether a query needs rewriting.
    
    A query is rewritten if:
      1. Ambiguity score exceeds threshold, OR
      2. Query is too short
    
    Args:
        query: User query to evaluate
        threshold: Ambiguity score threshold (default from settings)
    
    Returns:
        True if query should be rewritten, False otherwise
    
    Example:
        >>> should_rewrite("What about symptoms?", threshold=0.6)
        True  # ambiguity score 0.85 > 0.6
        
        >>> should_rewrite("What are MCI symptoms?", threshold=0.6)
        False  # ambiguity score 0.4 < 0.6
    """
    ambiguity_score = detect_ambiguity(query)
    
    # Rule 1: Check if ambiguity exceeds threshold
    if ambiguity_score > threshold:
        logger.debug(f"Query needs rewrite: ambiguity {ambiguity_score:.2f} > {threshold}")
        return True
    
    # Rule 2: Check minimum length
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
    Rewrite query with comprehensive logging.
    
    This is the main entry point. It:
      1. Detects if rewriting is needed
      2. Rewrites if necessary
      3. Logs all metadata
      4. Returns both rewritten query and metadata
    
    Args:
        query: Original user query
    
    Returns:
        Tuple of (rewritten_query, metadata_dict) where metadata includes:
          - rewrite_id: UUID for tracking
          - original_query: The original user query
          - rewritten_query: The final query (may be same as original)
          - was_rewritten: Boolean flag
          - ambiguity_score: Score before rewriting
          - timestamp: When rewriting occurred
    
    Example:
        >>> rewritten, metadata = rewrite_with_logging("What about symptoms?")
        >>> print(rewritten)
        "What are the symptoms of mild cognitive impairment in early stages?"
        >>> print(metadata['was_rewritten'])
        True
    """
    start_time = time.time()
    rewrite_id = str(uuid.uuid4())[:8]  # Short ID for logging
    
    # Step 1: Check if rewriting needed
    ambiguity_score = detect_ambiguity(query)
    needs_rewrite = should_rewrite(query)
    
    # Step 2: Rewrite if needed
    if needs_rewrite:
        rewritten_query = rewrite_query(query)
        was_rewritten = rewritten_query != query
    else:
        rewritten_query = query
        was_rewritten = False
    
    # Step 3: Calculate metrics
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Step 4: Build metadata
    metadata = {
        "rewrite_id": rewrite_id,
        "original_query": query,
        "rewritten_query": rewritten_query,
        "was_rewritten": was_rewritten,
        "ambiguity_score": round(ambiguity_score, 3),
        "timestamp": datetime.now().isoformat(),
        "elapsed_ms": round(elapsed_ms, 2),
    }
    
    # Step 5: Log the operation
    if was_rewritten:
        logger.info(
            f"[{rewrite_id}] Query rewritten: "
            f"'{query[:40]}...' → '{rewritten_query[:40]}...' "
            f"(ambiguity: {ambiguity_score:.2f})"
        )
    else:
        logger.debug(
            f"[{rewrite_id}] Query kept as-is: '{query[:40]}...' "
            f"(ambiguity: {ambiguity_score:.2f})"
        )
    
    return rewritten_query, metadata