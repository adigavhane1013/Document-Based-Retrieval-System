"""
guardrails/hallucination_filter.py

Post-generation hallucination detection and refusal enforcement.

Two independent checks:

1. INSUFFICIENT_CONTEXT PATTERN
   If the LLM followed the prompt instructions, it outputs
   "INSUFFICIENT_CONTEXT: ..." when it cannot answer. We pattern-match
   this and convert it into a structured refusal.

2. CITATION COVERAGE SCORE
   Parse [SOURCE:chunk_id] citations in the answer and verify each
   cited chunk_id actually appears in the retrieved context. An answer
   that cites non-existent chunk IDs is hallucinating sources.
   If coverage < GROUNDING_MIN_SCORE, the response is suppressed.

These are independent of temperature — even at temperature=0, a model
can hallucinate if the context is missing the answer. These checks catch
that case structurally.
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("guardrails.hallucination_filter")

_INSUFFICIENT_PATTERN = re.compile(r"INSUFFICIENT_CONTEXT\s*:", re.IGNORECASE)
_CITATION_PATTERN     = re.compile(r"\[SOURCE:([a-zA-Z0-9_\-]+)\]")


def check_response(
    answer: str,
    retrieved_docs: List[Document],
) -> Tuple[bool, float, str]:
    """
    Validate a generated answer against the retrieved context.

    Returns:
        (is_grounded, grounding_score, final_answer)

        is_grounded     — True if the answer passes all checks
        grounding_score — float in [0, 1]; 1.0 = all citations verified
        final_answer    — cleaned answer, or refusal message
    """

    # ── Check 1: model self-reported it couldn't answer ──────────────────────
    if _INSUFFICIENT_PATTERN.search(answer):
        logger.info("Model reported INSUFFICIENT_CONTEXT")
        return False, 0.0, settings.REFUSAL_MESSAGE

    # ── Check 2: citation coverage ───────────────────────────────────────────
    valid_chunk_ids = {
        doc.metadata.get("chunk_id", "") for doc in retrieved_docs
    }
    cited_ids = set(_CITATION_PATTERN.findall(answer))

    if not cited_ids:
        # No citations at all — model ignored the citation instruction.
        # This is suspicious; score it low but don't auto-refuse here
        # (some answers are legitimately short and citation-free).
        grounding_score = 0.5
        logger.warning("Answer contains no citations — potential hallucination risk")
    else:
        hallucinated_citations = cited_ids - valid_chunk_ids
        verified_citations     = cited_ids & valid_chunk_ids

        if hallucinated_citations:
            logger.warning(
                f"Hallucinated citation IDs detected: {hallucinated_citations}. "
                f"Valid IDs: {valid_chunk_ids}"
            )

        grounding_score = (
            len(verified_citations) / len(cited_ids) if cited_ids else 0.0
        )

    if grounding_score < settings.GROUNDING_MIN_SCORE:
        logger.warning(
            f"Grounding score {grounding_score:.2f} < threshold {settings.GROUNDING_MIN_SCORE}. "
            f"Refusing response."
        )
        return False, grounding_score, settings.REFUSAL_MESSAGE

    logger.info(f"Response grounding score: {grounding_score:.2f} — accepted")
    return True, grounding_score, answer