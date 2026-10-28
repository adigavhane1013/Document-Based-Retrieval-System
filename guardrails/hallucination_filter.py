"""
guardrails/hallucination_filter.py

Post-generation citation validation and refusal enforcement.

Checks:
1. INSUFFICIENT_CONTEXT
   Detect when the model explicitly says it cannot answer.

2. EMPTY_RESPONSE
   Reject empty responses.

3. CITATION VALIDATION & CANONICALIZATION
   Verify that citation references map to retrieved documents.
   Supports [SOURCE:chunk_id], [chunk_id], and Unicode variants (【...】).
   Exact IDs are preferred; a unique 12+ character prefix is resolved
   back to the canonical chunk_id.
"""

import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from configs.settings import settings
from observability.logger import get_logger


logger = get_logger("guardrails.hallucination_filter")


# ── Patterns ──────────────────────────────────────────────────────────────────

_INSUFFICIENT_PATTERN = re.compile(
    r"INSUFFICIENT_CONTEXT\s*:",
    re.IGNORECASE,
)

# Robust citation regex:
# - Supports '[' and '【' (and their closing counterparts ']' and '】')
# - Strips zero-width spaces (\u200b, \ufeff) and unicode whitespace (\u202f, \u00a0)
# - Makes 'SOURCE:' optional so bare IDs like [decc69c0b065] and [SOURCE:decc69c0b065] match
_CITATION_PATTERN = re.compile(
    r"[\[\u3010][\s\u200b\ufeff\u202f\u00a0]*(?:SOURCE[\s\u200b\ufeff\u202f\u00a0]*:[\s\u200b\ufeff\u202f\u00a0]*)?([a-zA-Z0-9_.\-]+)[\s\u200b\ufeff\u202f\u00a0]*[\]\u3011]",
    re.IGNORECASE,
)


def _clean_zero_width_chars(text: str) -> str:
    """Strip zero-width and invisible unicode codepoints."""
    return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)


# ── Citation canonicalization ────────────────────────────────────────────────

MIN_CITATION_PREFIX_LENGTH = 12


def _build_lower_lookup(valid_chunk_ids: set[str]) -> Dict[str, str]:
    """Map lowercase chunk_id -> original chunk_id for case-insensitive lookup."""
    return {chunk_id.lower(): chunk_id for chunk_id in valid_chunk_ids}


def _resolve_citation_ids(
    cited_ids: List[str],
    valid_chunk_ids: set[str],
) -> Tuple[set[str], set[str]]:
    """
    Resolve generated citation IDs to canonical retrieved chunk IDs.

    Resolution order:
      1. Exact match (case-sensitive) -> accepted.
      2. Case-insensitive exact match -> accepted, canonical = stored chunk_id.
      3. Unique 12+ char prefix (case-insensitive) of a retrieved chunk -> accepted.
      4. Otherwise -> unresolved/invalid.
    """
    resolved: set[str] = set()
    unresolved: set[str] = set()

    lower_lookup = _build_lower_lookup(valid_chunk_ids)

    for cited_id in cited_ids:
        if cited_id in valid_chunk_ids:
            resolved.add(cited_id)
            continue

        # Case-insensitive exact match
        canonical_id = lower_lookup.get(cited_id.lower())
        if canonical_id is not None:
            resolved.add(canonical_id)
            logger.info(
                "Resolved case-insensitive citation '%s' -> '%s'",
                cited_id,
                canonical_id,
            )
            continue

        if len(cited_id) < MIN_CITATION_PREFIX_LENGTH:
            unresolved.add(cited_id)
            continue

        cited_lower = cited_id.lower()
        matches = [
            chunk_id
            for chunk_id in valid_chunk_ids
            if chunk_id.lower().startswith(cited_lower)
        ]

        if len(matches) == 1:
            canonical_id = matches[0]
            resolved.add(canonical_id)
            logger.info(
                "Resolved shortened citation '%s' -> '%s'",
                cited_id,
                canonical_id,
            )
        else:
            unresolved.add(cited_id)
            if len(matches) > 1:
                logger.warning(
                    "Ambiguous shortened citation '%s' matched %d chunks",
                    cited_id,
                    len(matches),
                )

    return resolved, unresolved


def _canonicalize_citations(
    answer: str,
    valid_chunk_ids: set[str],
) -> Tuple[str, set[str], set[str]]:
    """
    Resolve model-generated citations to canonical retrieved chunk IDs.

    Normalizes all recognized citation variants cleanly to [SOURCE:<canonical_id>].

    Only bracketed tokens that are either:
      * [SOURCE:<id>]  — explicitly marked citation
      * [<id>]         — bare ID that is NOT all-uppercase
    are treated as citations. Bracketed text with other labels (e.g.
    [NOTE: see docs], [Disambiguation: ..]) is left untouched in the
    output and excluded from the cited-IDs set so it cannot trigger a
    false refusal.
    """
    normalized_answer = _clean_zero_width_chars(answer or "")

    # Build the cited-IDs set, filtering out [WORD:..] labels.
    # All-uppercase bare brackets like [NOTE:..] are treated as non-citations.
    raw_cited_ids: set[str] = set()
    for match in _CITATION_PATTERN.finditer(normalized_answer):
        full = match.group(0)
        inner = match.group(1)

        if re.search(r"SOURCE\s*:", full, re.IGNORECASE):
            # Explicit [SOURCE:...] citation — always treat as citation
            raw_cited_ids.add(inner)
        elif inner.isupper():
            # [WORD:..] all-uppercase label — not a citation, ignore
            pass
        else:
            # Bare ID like [abc123] — treat as citation
            raw_cited_ids.add(inner)

    resolved_ids, unresolved_ids = _resolve_citation_ids(
        raw_cited_ids,
        valid_chunk_ids,
    )

    lower_lookup = _build_lower_lookup(valid_chunk_ids)

    def replace_citation(match: re.Match) -> str:
        full = match.group(0)
        inner = match.group(1)

        # Leave [WORD:..] uppercase labels alone
        if not re.search(r"SOURCE\s*:", full, re.IGNORECASE) and inner.isupper():
            return full

        generated_id = inner

        if generated_id in valid_chunk_ids:
            canonical_id = generated_id
        elif generated_id.lower() in lower_lookup:
            # Case-insensitive exact match
            canonical_id = lower_lookup[generated_id.lower()]
        else:
            matches = [
                chunk_id
                for chunk_id in valid_chunk_ids
                if (
                    len(generated_id) >= MIN_CITATION_PREFIX_LENGTH
                    and chunk_id.lower().startswith(generated_id.lower())
                )
            ]
            if len(matches) != 1:
                return match.group(0)
            canonical_id = matches[0]

        return f"[SOURCE:{canonical_id}]"

    normalized_answer = _CITATION_PATTERN.sub(replace_citation, normalized_answer)

    return normalized_answer, resolved_ids, unresolved_ids


# ── Main guardrail ────────────────────────────────────────────────────────────

def check_response(
    answer: str,
    retrieved_docs: List[Document],
) -> Tuple[bool, float, str]:
    """
    Validate a generated answer against retrieved context.

    Returns:
        (
            is_grounded,
            grounding_score,
            final_answer,
        )

    grounding_score semantics:
        Continuous value in [0.0, 1.0] equal to
        (resolved citation IDs) / (total cited IDs).
        0.0 when no citations were found at all (no signal, not "fully
        ungrounded"). 1.0 when every cited ID resolved to a retrieved
        chunk. Intermediate values represent partial citation coverage.
    """
    # ── Check 1: explicit insufficient context ───────────────────────────────
    if _INSUFFICIENT_PATTERN.search(answer or ""):
        logger.info("Model reported INSUFFICIENT_CONTEXT")
        return False, 0.0, settings.REFUSAL_MESSAGE

    # ── Check 2: empty answer ────────────────────────────────────────────────
    if not answer or not answer.strip():
        logger.warning("Empty answer received")
        return False, 0.0, settings.REFUSAL_MESSAGE

    # ── Build valid citation lookup ──────────────────────────────────────────
    documents_by_id: Dict[str, Document] = {}
    for document in retrieved_docs:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            documents_by_id[str(chunk_id)] = document

    valid_chunk_ids = set(documents_by_id.keys())

    # ── Check 3: citation validation + canonicalization ──────────────────────
    (
        normalized_answer,
        cited_ids,
        unresolved_citations,
    ) = _canonicalize_citations(
        answer,
        valid_chunk_ids,
    )

    # Zero cited IDs is the only refusal point: there is no signal at all,
    # so the answer cannot be presented as grounded.
    # If there are unresolved citations, we still accept the answer
    # but surface the partial coverage via a continuous score.
    if not cited_ids and not unresolved_citations:
        logger.warning("Answer contains no citations (no signal)")
        return False, 0.0, settings.REFUSAL_MESSAGE

    total_cited = len(cited_ids) + len(unresolved_citations)
    resolved_count = len(cited_ids)
    grounding_score = (
        resolved_count / total_cited
        if total_cited > 0
        else 0.0
    )

    if unresolved_citations:
        logger.warning(
            f"Partial citation coverage: {resolved_count}/{total_cited} "
            f"resolved. Unresolved: {unresolved_citations}. "
            f"Score={grounding_score:.2f} (not refusing)"
        )
    else:
        logger.debug(
            f"Validated {resolved_count}/{total_cited} citation IDs "
            f"(score={grounding_score:.2f})"
        )

    return True, grounding_score, normalized_answer