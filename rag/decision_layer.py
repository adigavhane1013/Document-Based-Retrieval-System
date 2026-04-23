"""
rag/decision_layer.py

RAGAS score-based decision layer for DocMind RAG system.
Makes intelligent decisions: Accept/Retry/Fallback based on evaluation scores.

Key Concepts:
  - Decision Types: ACCEPT (good answer), RETRY (poor answer, try again), 
                   FALLBACK (poor answer, use better model), REJECT (max retries)
  - Score Thresholds: Faithfulness >= 0.70, Relevance >= 0.65
  - Retry Strategy: Increase retrieval top-k, lower score threshold
  - Fallback Strategy: Use better LLM if available
  - Logging: Track all decisions for analysis

Example:
  Input: Answer with faithfulness=0.45 (too low)
  Decision: RETRY with more context
  Output: New answer with improved faithfulness
"""

import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import uuid

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("rag.decision_layer")


# ─────────────────────────────────────────────────────────────────────────────
# DECISION LAYER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DecisionLayer:
    """
    Makes decisions on whether to accept, retry, or fallback responses.
    
    Decision Logic:
      1. If attempt > max_attempts → REJECT (give up)
      2. Else if faithfulness < threshold AND retries available → RETRY (get better context)
      3. Else if relevance < threshold → FALLBACK (use better model)
      4. Else → ACCEPT (good answer)
    """
    
    def __init__(self):
        """Initialize the decision layer with thresholds from settings."""
        self.faithfulness_threshold = settings.RAGAS_FAITHFULNESS_THRESHOLD
        self.relevance_threshold = settings.RAGAS_RELEVANCE_THRESHOLD
        self.max_retry_attempts = settings.RAGAS_MAX_RETRY_ATTEMPTS
        self.retry_mode = settings.RAGAS_RETRY_MODE
        self.fallback_enabled = settings.RAGAS_FALLBACK_ENABLED
        
        logger.info(
            f"DecisionLayer initialized: "
            f"faithfulness_threshold={self.faithfulness_threshold}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"max_retries={self.max_retry_attempts}"
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # SCORE EVALUATION
    # ─────────────────────────────────────────────────────────────────────
    
    def should_retry(self, scores: Dict[str, float], attempt: int) -> bool:
        """
        Decide if we should retry based on scores.
        
        Retry if:
          1. Faithfulness below threshold, AND
          2. Haven't exceeded max retry attempts
        
        Args:
            scores: Dict with 'faithfulness', 'answer_relevancy', etc.
            attempt: Current attempt number (1-based)
        
        Returns:
            True if should retry, False otherwise
        """
        faithfulness = scores.get("faithfulness", 0.0)
        
        # Check if faithfulness is below threshold
        if faithfulness < self.faithfulness_threshold:
            # Check if we haven't exceeded max retries
            if attempt < self.max_retry_attempts:
                logger.debug(
                    f"Should retry: faithfulness {faithfulness:.2f} "
                    f"< threshold {self.faithfulness_threshold} "
                    f"(attempt {attempt}/{self.max_retry_attempts})"
                )
                return True
            else:
                logger.warning(
                    f"Max retries reached: faithfulness {faithfulness:.2f} "
                    f"still below threshold"
                )
                return False
        
        return False
    
    def should_fallback(self, scores: Dict[str, float]) -> bool:
        """
        Decide if we should fallback to better model.
        
        Fallback if:
          1. Fallback enabled in settings, AND
          2. Relevance below threshold
        
        Args:
            scores: Dict with evaluation scores
        
        Returns:
            True if should fallback, False otherwise
        """
        if not self.fallback_enabled:
            return False
        
        relevance = scores.get("answer_relevancy", 0.0)
        
        if relevance < self.relevance_threshold:
            logger.debug(
                f"Should fallback: relevance {relevance:.2f} "
                f"< threshold {self.relevance_threshold}"
            )
            return True
        
        return False
    
    # ─────────────────────────────────────────────────────────────────────
    # RETRY LOGIC
    # ─────────────────────────────────────────────────────────────────────
    
    def get_retry_config(self) -> Dict[str, Any]:
        """
        Get configuration for retry strategy.
        
        Returns dict with modified parameters for better retrieval:
          - Increase top_k for broader search
          - Lower similarity threshold to accept more results
          - Increase context max chars for more detail
        
        Returns:
            Dict with retry configuration
        """
        config = {
            "top_k_dense": settings.TOP_K_DENSE + settings.RETRY_RETRIEVAL_TOP_K_INCREASE,
            "top_k_sparse": settings.TOP_K_SPARSE + settings.RETRY_RETRIEVAL_TOP_K_INCREASE,
            "retrieval_score_threshold": settings.RETRY_LOWERED_THRESHOLD,
            "context_max_chars": settings.RETRY_CONTEXT_MAX_CHARS_INCREASE,
        }
        
        logger.debug(f"Retry config: {config}")
        return config
    
    # ─────────────────────────────────────────────────────────────────────
    # FALLBACK LOGIC
    # ─────────────────────────────────────────────────────────────────────
    
    def get_fallback_model(self) -> Optional[str]:
        """
        Get fallback LLM model name.
        
        Checks instance variable first (for testing), then settings.
        
        Returns:
            Model name string, or None if not available
        """
        # Check instance variable first (allows test override)
        if hasattr(self, 'FALLBACK_LLM_MODEL') and self.FALLBACK_LLM_MODEL:
            fallback_model = self.FALLBACK_LLM_MODEL
        else:
            fallback_model = settings.FALLBACK_LLM_MODEL
        
        main_model = settings.LLM_MODEL
        
        # Don't fallback to same model
        if fallback_model and fallback_model != main_model:
            logger.info(f"Falling back from {main_model} to {fallback_model}")
            return fallback_model
        else:
            logger.warning(f"Fallback model not configured or same as main")
            return None
    
    # ─────────────────────────────────────────────────────────────────────
    # MAIN DECISION POINT
    # ─────────────────────────────────────────────────────────────────────
    
    def evaluate_and_decide(
        self,
        answer: str,
        scores: Dict[str, float],
        attempt: int = 1,
        max_attempts: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        Evaluate response and decide: accept/retry/fallback/reject.
        
        Decision tree:
          1. If attempt > max_attempts → REJECT
          2. Else if faithfulness too low AND retries available → RETRY
          3. Else if relevance too low AND fallback available → FALLBACK
          4. Else → ACCEPT
        
        Args:
            answer: The generated answer
            scores: RAGAS evaluation scores
            attempt: Current attempt number (1-based)
            max_attempts: Override max attempts (default from settings)
        
        Returns:
            Tuple of (answer, decision_metadata, should_return_now)
            where should_return_now=True means pipeline should stop here
        
        Example:
            >>> answer, metadata, should_return = decision_layer.evaluate_and_decide(
            ...     answer="The answer is...",
            ...     scores={"faithfulness": 0.45, "answer_relevancy": 0.70},
            ...     attempt=1
            ... )
            >>> if metadata["decision_type"] == "RETRY":
            ...     # Get more context and regenerate
            >>> elif metadata["decision_type"] == "ACCEPT":
            ...     # Return this answer
        """
        decision_id = str(uuid.uuid4())[:8]
        max_attempts = max_attempts or self.max_retry_attempts
        
        # Extract scores
        faithfulness = scores.get("faithfulness", 0.0)
        relevance = scores.get("answer_relevancy", 0.0)
        
        # ── Decision Logic ──────────────────────────────────────────────
        decision_type = "UNKNOWN"
        reason = ""
        should_return = False
        metadata_extra = {}
        
        # Decision 1: Check if max retries exceeded FIRST
        if attempt > max_attempts:
            decision_type = "REJECT"
            reason = f"Max retries ({max_attempts}) exceeded"
            should_return = True
            logger.warning(f"[{decision_id}] Decision: REJECT (max retries)")
        
        # Decision 2: Check if we should retry (if retries still available)
        elif faithfulness < self.faithfulness_threshold and attempt <= max_attempts:
            decision_type = "RETRY"
            reason = f"Faithfulness {faithfulness:.2f} < {self.faithfulness_threshold}"
            metadata_extra["retry_config"] = self.get_retry_config()
            metadata_extra["retry_retrieval_increase"] = settings.RETRY_RETRIEVAL_TOP_K_INCREASE
            should_return = False  # Don't return yet, pipeline will retry
        
        # Decision 3: Check if we should fallback
        elif relevance < self.relevance_threshold:
            if self.fallback_enabled:
                decision_type = "FALLBACK"
                reason = f"Relevance {relevance:.2f} < {self.relevance_threshold}"
                fallback_model = self.get_fallback_model()
                if fallback_model:
                    metadata_extra["fallback_model"] = fallback_model
                    should_return = False  # Pipeline will regenerate with better model
                else:
                    # Can't fallback, just accept (no better model available)
                    decision_type = "ACCEPT"
                    reason = f"Relevance {relevance:.2f} < {self.relevance_threshold}, but no fallback model available"
                    should_return = True
            else:
                # Fallback disabled, just accept
                decision_type = "ACCEPT"
                reason = f"Relevance {relevance:.2f} < {self.relevance_threshold}, but fallback disabled"
                should_return = True
        
        # Decision 4: Scores are good
        else:
            decision_type = "ACCEPT"
            reason = (
                f"Faithfulness {faithfulness:.2f} >= {self.faithfulness_threshold}, "
                f"Relevance {relevance:.2f} >= {self.relevance_threshold}"
            )
            should_return = True
        
        # ── Build Decision Metadata ────────────────────────────────────
        metadata = {
            "decision_id": decision_id,
            "decision_type": decision_type,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "reason": reason,
            "should_return": should_return,
            "original_answer": answer,
            "scores": {
                "faithfulness": round(faithfulness, 3),
                "answer_relevancy": round(relevance, 3),
            },
            "thresholds": {
                "faithfulness": self.faithfulness_threshold,
                "relevance": self.relevance_threshold,
            },
            "timestamp": datetime.now().isoformat(),
            **metadata_extra,  # Add retry_config, fallback_model, etc.
        }
        
        # ── Logging ─────────────────────────────────────────────────────
        log_msg = (
            f"[{decision_id}] Decision: {decision_type} | "
            f"Attempt {attempt}/{max_attempts} | "
            f"Faithfulness: {faithfulness:.2f} | "
            f"Relevance: {relevance:.2f}"
        )
        
        if decision_type == "ACCEPT":
            logger.info(f"{log_msg} ✅")
        elif decision_type == "RETRY":
            logger.info(f"{log_msg} 🔄")
        elif decision_type == "FALLBACK":
            logger.info(f"{log_msg} 🔀")
        elif decision_type == "REJECT":
            logger.warning(f"{log_msg} ❌")
        
        return answer, metadata, should_return
    
    # ─────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────
    
    def log_decision_history(self, decisions: List[Dict[str, Any]]) -> None:
        """
        Log summary of all decisions made in a single query processing.
        
        Args:
            decisions: List of decision metadata dicts from multiple attempts
        """
        if not decisions:
            return
        
        logger.info(
            f"Decision history: {len(decisions)} attempt(s), "
            f"final decision: {decisions[-1]['decision_type']}"
        )
        
        for i, decision in enumerate(decisions, 1):
            decision_type = decision.get("decision_type", "UNKNOWN")
            # Handle cases where scores might not be present
            if "scores" in decision and "faithfulness" in decision["scores"]:
                faith = decision["scores"]["faithfulness"]
                logger.debug(f"  Attempt {i}: {decision_type} (faith={faith:.2f})")
            else:
                logger.debug(f"  Attempt {i}: {decision_type}")
    
    def get_decision_summary(self, decision: Dict[str, Any]) -> str:
        """
        Get human-readable summary of decision.
        
        Args:
            decision: Decision metadata dict
        
        Returns:
            Human-readable string summarizing the decision
        """
        decision_type = decision["decision_type"]
        attempt = decision["attempt"]
        max_attempts = decision["max_attempts"]
        reason = decision["reason"]
        
        return (
            f"{decision_type} (Attempt {attempt}/{max_attempts}): {reason}"
        )