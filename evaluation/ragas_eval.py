# evaluation/ragas_eval.py

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union

from observability.logger import get_logger
from configs.settings import settings

logger = get_logger("evaluation.ragas")

from embeddings.embedding_model import get_embedding_model


# ─────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────

def _get_embeddings():
    return get_embedding_model()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    try:
        return default if (val is None or math.isnan(val) or math.isinf(val)) else float(val)
    except Exception:
        return default


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[(?:SOURCE|src):.*?\]", "", text, flags=re.IGNORECASE)
    return text.strip()


def _trim_contexts(contexts: List[str]) -> List[str]:
    max_contexts = getattr(settings, "RAGAS_MAX_CONTEXTS", 10)
    max_chars = getattr(settings, "RAGAS_CONTEXT_MAX_CHARS", 4500)

    contexts = list(dict.fromkeys(contexts))
    dropped_count = max(0, len(contexts) - max_contexts)
    selected = contexts[:max_contexts]

    truncated_count = sum(1 for ctx in selected if len(ctx) > max_chars)
    if truncated_count or dropped_count:
        logger.warning(
            f"_trim_contexts: dropped {dropped_count} context(s) beyond "
            f"RAGAS_MAX_CONTEXTS={max_contexts}, truncated {truncated_count} "
            f"context(s) beyond RAGAS_CONTEXT_MAX_CHARS={max_chars}. "
            f"RAGAS faithfulness will only see the retained portion — "
            f"a citation pointing past the cutoff can score as unfaithful "
            f"even though the full chunk supports it."
        )

    return [ctx[:max_chars] for ctx in selected]


def _get_http_status_code(error: Exception) -> Optional[int]:
    """Extract HTTP status code from structured exception attributes, if available."""
    for attr in ("status_code", "http_status", "code", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val
    # Check wrapped/chained cause
    cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    if cause is not None and cause is not error:
        for attr in ("status_code", "http_status", "code", "status"):
            val = getattr(cause, attr, None)
            if isinstance(val, int):
                return val
    return None


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error from Groq API."""
    # Prefer structured status code
    status = _get_http_status_code(error)
    if status == 429:
        return True
    # Fallback to message-based matching
    error_str = str(error).lower()
    return "rate_limit" in error_str or "rate limit" in error_str


def _is_api_or_rate_limit_error(error: Exception) -> bool:
    """Check if error is an API, quota, rate limit, authentication, or connection error.

    Prefers structured exception attributes (status_code, http_status) for HTTP
    status checks, falling back to message-based matching for non-numeric patterns.
    Numeric HTTP status codes are NOT matched against the error message string to
    avoid false positives (e.g. a message mentioning '500 items' being classified
    as a server error).
    """
    # 1. Structured HTTP status code check (preferred)
    status = _get_http_status_code(error)
    if status is not None:
        if status in (401, 403, 429, 500, 502, 503, 504):
            return True

    # 2. Exception type name check
    error_type = type(error).__name__.lower()
    type_patterns = [
        "rate_limit", "ratelimit", "quota", "authentication",
        "unauthorized", "forbidden", "connection", "timeout",
        "readtimeout", "connecttimeout",
    ]
    if any(p in error_type for p in type_patterns):
        return True

    # 3. Message-based matching (non-numeric patterns only)
    error_str = str(error).lower()
    message_patterns = [
        "rate_limit",
        "rate limit",
        "quota",
        "insufficient_quota",
        "tokens per minute",
        "requests per minute",
        "authentication",
        "unauthorized",
        "forbidden",
        "connection",
        "timeout",
        "readtimeout",
        "connecttimeout",
        "server error",
        "service unavailable",
    ]
    return any(p in error_str for p in message_patterns)


def _atomic_write_json(file_path: Union[Path, str], data: Any) -> None:
    """Atomically write JSON data to file to prevent corruption on interruption."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _is_json_parse_error(error: Exception) -> bool:
    """
    Check if error is a JSON parsing/output format error.

    Common patterns:
    - JSONDecodeError
    - OUTPUT_PARSING_FAILURE
    - "json" or "parse" in error message
    - Llama reasoning before JSON in response
    """
    error_str = str(error).lower()

    # Direct error type checks
    if "json" in error_str or "parse" in error_str or "parsing" in error_str:
        return True
    if "output_parsing" in error_str or "output parsing" in error_str:
        return True
    if "expected json" in error_str or "expected valid json" in error_str:
        return True

    # Check error class name
    error_type = type(error).__name__.lower()
    if "json" in error_type or "parse" in error_type:
        return True

    return False


def _exponential_backoff_wait(attempt: int, base_wait: int = 5) -> int:
    """
    Calculate exponential backoff wait time.

    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds (default 5)

    Returns:
        Time to wait in seconds (base_wait * 2^attempt)

    Examples:
        attempt=0: wait = 5 * 2^0 = 5 seconds
        attempt=1: wait = 5 * 2^1 = 10 seconds
        attempt=2: wait = 5 * 2^2 = 20 seconds
    """
    return base_wait * (2 ** attempt)


# ─────────────────────────────────────────────────────────────
# LLM SETUP (FINAL - GROQ ONLY)
# ─────────────────────────────────────────────────────────────
def _get_llm():
    from langchain_openai import ChatOpenAI

    api_key = settings.LLM_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    class GroqLLM(ChatOpenAI):
        """
        Groq LLM wrapper with rate limit resilience.
        Forces n=1 to avoid Groq API constraint issues.
        """

        def _force_n(self, kwargs):
            kwargs["n"] = 1
            return kwargs

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def generate(self, messages, stop=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super().generate(messages, stop=stop, **kwargs)

        async def agenerate(self, messages, stop=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return await super().agenerate(messages, stop=stop, **kwargs)

        def invoke(self, input, config=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super().invoke(input, config=config, **kwargs)

        async def ainvoke(self, input, config=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return await super().ainvoke(input, config=config, **kwargs)

    return GroqLLM(
        api_key=api_key,
        base_url=settings.RAGAS_API_BASE,
        model=settings.RAGAS_EVAL_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        # RAGAS's internal prompts (claim decomposition, NLI verification) require
        # strict JSON-only output. Llama-via-Groq tends to add reasoning text before
        # the JSON block, which breaks LangChain's JSON output parser
        # (OUTPUT_PARSING_FAILURE). Forcing JSON mode here is safe — this LLM
        # instance is only ever used internally by RAGAS, never to generate
        # user-facing answers.
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def run_ragas_evaluation(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    import ragas
    logger.info(f"RAGAS version: {ragas.__version__}")

    # prevent accidental OpenAI usage
    os.environ.pop("OPENAI_API_KEY", None)

    # ── Clean + Filter test cases ─────────────────────────────
    cleaned_cases = []
    for tc in test_cases:
        if not tc.get("answer") or not tc.get("contexts"):
            continue

        answer = _clean_answer(tc["answer"])
        contexts = _trim_contexts(tc["contexts"])

        if not answer or not contexts:
            continue

        cleaned_case = {
            "question": tc["question"],
            "answer": answer,
            "contexts": contexts,
        }
        # Retain metadata and ground truth fields if present
        for meta_key in (
            "reference",
            "reference_answer",
            "relevant_chunk_ids",
            "retrieved_chunk_ids",
            "chunk_hit",
            "chunk_recall",
            "source_document",
            "difficulty",
            "evaluation_type",
            "is_grounded",
            "grounding_score",
            "refused",
        ):
            if meta_key in tc:
                cleaned_case[meta_key] = tc[meta_key]

        cleaned_cases.append(cleaned_case)

    if not cleaned_cases:
        return {
            "error": "No valid cleaned test cases",
            "total_cases": len(test_cases),
            "evaluated_cases": 0,
        }

    logger.info(f"Evaluating {len(cleaned_cases)} cleaned cases")

    # ── Initialize LLM + embeddings ───────────────────────────
    llm = _get_llm()
    embeds = _get_embeddings()

    if llm is None:
        raise ValueError("LLM initialization failed (None)")

    # ── RAGAS Setup ───────────────────────────────────────────
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(langchain_llm=llm)
    ragas_embed = LangchainEmbeddingsWrapper(embeddings=embeds)

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embed
    )

    # ── Build dataset ─────────────────────────────────────────
    samples = [
        SingleTurnSample(
            user_input=tc["question"],
            response=tc["answer"],
            retrieved_contexts=tc["contexts"],
            reference=tc.get("reference") or tc.get("reference_answer") or None,
        )
        for tc in cleaned_cases
    ]

    import nest_asyncio
    nest_asyncio.apply()

    # ── ENHANCED: Retry with exponential backoff for ALL error types ───────────
    retry_count = getattr(settings, "RAGAS_RETRY_COUNT", 2)
    max_rate_limit_retries = 3  # Extra retries for rate limits
    max_json_parse_retries = 2  # Extra retries for JSON parse errors
    rate_limit_attempt = 0
    json_parse_attempt = 0
    general_attempt = 0
    last_exception = None
    df = None

    while general_attempt < retry_count:
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=[faithfulness, answer_relevancy],
                run_config=RunConfig(max_workers=1, timeout=180),
            )
            df = result.to_pandas()
            logger.info(f"RAGAS evaluation succeeded on attempt {general_attempt + 1}")
            break

        except Exception as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)
            is_json_error = _is_json_parse_error(e)
            error_type = type(e).__name__

            # ── Rate limit error: exponential backoff (separate counter) ──────
            if is_rate_limit and rate_limit_attempt < max_rate_limit_retries:
                wait_time = _exponential_backoff_wait(rate_limit_attempt)
                logger.warning(
                    f"Rate limit hit ({error_type}). "
                    f"Backoff attempt {rate_limit_attempt + 1}/{max_rate_limit_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                rate_limit_attempt += 1
                time.sleep(wait_time)
                continue  # Does NOT increment general_attempt

            # ── JSON parse error: exponential backoff (separate counter) ──────
            elif is_json_error and json_parse_attempt < max_json_parse_retries:
                wait_time = _exponential_backoff_wait(json_parse_attempt)
                logger.warning(
                    f"JSON parse error ({error_type}). "
                    f"This often happens when Llama adds reasoning before JSON. "
                    f"Backoff attempt {json_parse_attempt + 1}/{max_json_parse_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                json_parse_attempt += 1
                time.sleep(wait_time)
                continue  # Does NOT increment general_attempt

            # ── Exhausted rate-limit or JSON-parse retries: break ─────────────
            elif is_rate_limit or is_json_error:
                logger.error(
                    f"{error_type} persists after exhausting "
                    f"{'rate-limit' if is_rate_limit else 'JSON-parse'} retries. "
                    f"Will attempt per-metric partial evaluation if available."
                )
                break  # Exit loop to try partial results if they exist

            # ── Other errors: regular retry ────────────────────────────────────
            else:
                general_attempt += 1
                logger.warning(
                    f"RAGAS attempt {general_attempt}/{retry_count} failed ({error_type}): {e}"
                )
                time.sleep(2)  # Brief pause before next attempt
    else:
        # Loop completed without success (general_attempt reached retry_count)
        pass

    if df is None:
        if last_exception is not None:
            return {
                "error": f"Evaluation failed after retries. "
                         f"Last error ({type(last_exception).__name__}): "
                         f"{last_exception}",
                "evaluated_cases": 0,
            }
        return {
            "error": "Evaluation produced no results",
            "evaluated_cases": 0,
        }

    faith_vals: List[float] = []
    relev_vals: List[float] = []
    fully_failed_count = 0  # rows where BOTH metrics failed

    for i in range(len(df)):
        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        f_ok = f_val is not None and not math.isnan(f_val)
        r_ok = r_val is not None and not math.isnan(r_val)

        if f_ok:
            faith_vals.append(f_val)
        if r_ok:
            relev_vals.append(r_val)
        if not f_ok and not r_ok:
            fully_failed_count += 1
        if not f_ok:
            logger.warning(
                f"Faithfulness failed for case {i} (likely judge-LLM parse error) "
                f"— excluded from average"
            )
        if not r_ok:
            logger.warning(
                f"Answer relevancy failed for case {i} "
                f"— excluded from average"
            )

    if not faith_vals and not relev_vals:
        return {
            "error": "All evaluations failed for every metric",
            "evaluated_cases": 0,
            "failed_cases": fully_failed_count,
        }

    faith_avg = (sum(faith_vals) / len(faith_vals)) if faith_vals else None
    relev_avg = (sum(relev_vals) / len(relev_vals)) if relev_vals else None

    # ── Per question breakdown ────────────────────────────────
    per_question = []

    for i in range(len(df)):
        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        f_ok = f_val is not None and not math.isnan(f_val)
        r_ok = r_val is not None and not math.isnan(r_val)

        if not f_ok and not r_ok:
            continue  # nothing usable for this case at all

        case_meta = cleaned_cases[i]
        q_item = {
            "question": case_meta["question"],
            "faithfulness": round(_safe(f_val), 3) if f_ok else None,
            "answer_relevancy": round(_safe(r_val), 3) if r_ok else None,
        }

        # Populate additional metadata fields if provided
        ref_ans = case_meta.get("reference") or case_meta.get("reference_answer")
        if ref_ans:
            q_item["reference_answer"] = ref_ans
        if "relevant_chunk_ids" in case_meta:
            q_item["relevant_chunk_ids"] = case_meta["relevant_chunk_ids"]
        if "retrieved_chunk_ids" in case_meta:
            q_item["retrieved_chunk_ids"] = case_meta["retrieved_chunk_ids"]
        if "chunk_hit" in case_meta:
            q_item["chunk_hit"] = case_meta["chunk_hit"]
        if "chunk_recall" in case_meta:
            q_item["chunk_recall"] = case_meta["chunk_recall"]
        if "source_document" in case_meta:
            q_item["source_document"] = case_meta["source_document"]
        if "difficulty" in case_meta:
            q_item["difficulty"] = case_meta["difficulty"]
        if "evaluation_type" in case_meta:
            q_item["evaluation_type"] = case_meta["evaluation_type"]
        if "is_grounded" in case_meta:
            q_item["is_grounded"] = case_meta["is_grounded"]
        if "grounding_score" in case_meta:
            q_item["grounding_score"] = case_meta["grounding_score"]

        per_question.append(q_item)

    # ── Final output ──────────────────────────────────────────
    success_count = len(per_question)
    logger.info(
        f"RAGAS evaluation complete: {success_count} successful, "
        f"{fully_failed_count} fully failed out of {len(cleaned_cases)} cases"
    )

    return {
        "evaluated_cases": len(per_question),
        "failed_cases": fully_failed_count,
        "total_cases": len(test_cases),
        "faithfulness": round(faith_avg, 3) if faith_avg is not None else None,
        "answer_relevancy": round(relev_avg, 3) if relev_avg is not None else None,
        "hallucination_rate": round(1.0 - faith_avg, 3) if faith_avg is not None else None,
        "per_question": per_question,
    }


# ─────────────────────────────────────────────────────────────
# BENCHMARK EVALUATION WORKFLOW
# ─────────────────────────────────────────────────────────────

def _get_company_kb_session():
    """Load or initialize the Company KB RAG session backed by Qdrant."""
    from vectorstore.vectordb import (
        load_company_kb_vectorstore,
        get_all_company_kb_documents,
        company_kb_exists,
    )
    from retrieval.retriever import HybridRetriever
    from rag.pipeline import RAGSession

    if not company_kb_exists():
        raise RuntimeError(
            "Company KB vectorstore not found. Please run Company KB ingestion first: "
            "python -m ingestion.company_kb"
        )

    kb_vectorstore = load_company_kb_vectorstore()
    if kb_vectorstore is None:
        raise RuntimeError("Failed to load Company KB vectorstore.")

    kb_chunks = get_all_company_kb_documents()
    if not kb_chunks:
        raise RuntimeError("No documents found in Company KB collection.")

    kb_retriever = HybridRetriever(
        vectorstore=kb_vectorstore,
        all_chunks=kb_chunks,
    )

    return RAGSession(
        session_id="company_kb",
        retriever=kb_retriever,
        all_chunks=kb_chunks,
    )


def _is_valid_metric_score(val: Any) -> bool:
    """Check if a metric score is a valid, non-null float."""
    if val is None:
        return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


def _has_both_ragas_metrics(entry: Optional[Dict[str, Any]]) -> bool:
    """Check if an entry has valid, non-null scores for both faithfulness and answer_relevancy."""
    if not entry:
        return False
    return (
        _is_valid_metric_score(entry.get("faithfulness"))
        and _is_valid_metric_score(entry.get("answer_relevancy"))
    )


def _reconcile_entry_status(entry: Dict[str, Any]) -> None:
    """Reconcile ragas_status flag with stored metric scores if they disagree."""
    has_faith = _is_valid_metric_score(entry.get("faithfulness"))
    has_relev = _is_valid_metric_score(entry.get("answer_relevancy"))
    if has_faith and has_relev:
        entry["ragas_status"] = "completed"
        entry["ragas_error"] = None
    elif has_faith or has_relev:
        entry["ragas_status"] = "partial"
        missing_metric = "answer_relevancy" if has_faith else "faithfulness"
        valid_metric = "faithfulness" if has_faith else "answer_relevancy"
        ragas_err = entry.get("ragas_error")
        if ragas_err:
            err_lower = str(ragas_err).lower()
            if err_lower.startswith(missing_metric):
                pass
            else:
                entry["ragas_error"] = None
        else:
            entry["ragas_error"] = None
    else:
        if entry.get("ragas_status") not in ("failed", "pending"):
            entry["ragas_status"] = "pending"


def _update_ragas_status(entry: Dict[str, Any], last_error: Optional[str] = None) -> None:
    """Update the ragas_status and ragas_error fields based on valid metric scores."""
    has_faith = _is_valid_metric_score(entry.get("faithfulness"))
    has_relev = _is_valid_metric_score(entry.get("answer_relevancy"))
    if has_faith and has_relev:
        entry["ragas_status"] = "completed"
        entry["ragas_error"] = None
    elif has_faith or has_relev:
        entry["ragas_status"] = "partial"
        if last_error:
            entry["ragas_error"] = last_error
    else:
        if last_error:
            entry["ragas_status"] = "failed"
            entry["ragas_error"] = last_error
        else:
            entry["ragas_status"] = "failed"
            entry["ragas_error"] = "Both faithfulness and answer_relevancy evaluations failed"


def _load_and_validate_dataset(dataset_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Load and validate the benchmark dataset.
    Supports:
      - JSON object with 'questions' list: {"questions": [...]}
      - Top-level JSON list: [...]
    Validates structure and non-empty questions with clear error messages.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in benchmark dataset file '{path}': {e}") from e

    if isinstance(raw_data, dict):
        if "questions" not in raw_data:
            raise ValueError(
                f"Invalid benchmark dataset format in '{path}': "
                f"Expected JSON object with 'questions' key, but found keys: {list(raw_data.keys())}"
            )
        questions_list = raw_data["questions"]
    elif isinstance(raw_data, list):
        questions_list = raw_data
    else:
        raise ValueError(
            f"Invalid benchmark dataset format in '{path}': "
            f"Expected JSON object or list, but got {type(raw_data).__name__}"
        )

    if not isinstance(questions_list, list) or len(questions_list) == 0:
        raise ValueError(
            f"Benchmark dataset in '{path}' contains no questions (must be a non-empty list)."
        )

    for idx, item in enumerate(questions_list):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question at index {idx} in '{path}' is not a JSON object: {type(item).__name__}"
            )
        q_text = item.get("question")
        if not q_text or not isinstance(q_text, str) or not q_text.strip():
            raise ValueError(
                f"Question at index {idx} in '{path}' is missing a valid non-empty 'question' string."
            )

    # Check for duplicate question text
    seen_questions: Dict[str, int] = {}
    for idx, item in enumerate(questions_list):
        q_text = item["question"].strip()
        if q_text in seen_questions:
            raise ValueError(
                f"Duplicate question text found in '{path}': "
                f"index {seen_questions[q_text]} and index {idx} both have question "
                f"'{q_text[:80]}{'...' if len(q_text) > 80 else ''}'. "
                f"Each question must be unique to prevent result overwrites."
            )
        seen_questions[q_text] = idx

    return questions_list


def _evaluate_single_metric_ragas(
    case_meta: Dict[str, Any],
    metric_name: str,
    evaluator_llm: Any,
    evaluator_embed: Any,
    error_details: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Evaluate a single RAGAS metric for one test case.

    Args:
        case_meta: Dict with 'question', 'answer', 'contexts', and optionally 'reference'.
        metric_name: Either 'faithfulness' or 'answer_relevancy'.
        evaluator_llm: Wrapped RAGAS LLM.
        evaluator_embed: Wrapped RAGAS embeddings.
        error_details: Optional dict to receive error message if evaluation fails or returns None.

    Returns:
        The metric score as a float, or None if evaluation failed.

    Raises:
        Re-raises API/quota/rate-limit errors so the caller can stop cleanly.
    """
    import nest_asyncio
    nest_asyncio.apply()

    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy

    sample = SingleTurnSample(
        user_input=case_meta["question"],
        response=case_meta["answer"],
        retrieved_contexts=case_meta["contexts"],
        reference=case_meta.get("reference") or case_meta.get("reference_answer") or None,
    )

    if metric_name == "faithfulness":
        metric_obj = Faithfulness(llm=evaluator_llm)
    elif metric_name == "answer_relevancy":
        metric_obj = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embed)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    retry_count = getattr(settings, "RAGAS_RETRY_COUNT", 2)
    max_rate_limit_retries = 3
    max_json_parse_retries = 2
    rate_limit_attempt = 0
    json_parse_attempt = 0
    general_attempt = 0
    df = None
    last_exception = None

    while general_attempt < retry_count:
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[metric_obj],
                run_config=RunConfig(max_workers=1, timeout=180),
                show_progress=False,
            )
            df = result.to_pandas()
            break
        except Exception as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)
            is_json_error = _is_json_parse_error(e)
            error_type = type(e).__name__

            if is_rate_limit and rate_limit_attempt < max_rate_limit_retries:
                wait_time = _exponential_backoff_wait(rate_limit_attempt)
                logger.warning(
                    f"RAGAS rate limit hit ({error_type}) for {metric_name}. "
                    f"Backoff attempt {rate_limit_attempt + 1}/{max_rate_limit_retries}. "
                    f"Waiting {wait_time}s..."
                )
                rate_limit_attempt += 1
                time.sleep(wait_time)
                continue  # Does NOT increment general_attempt
            elif is_json_error and json_parse_attempt < max_json_parse_retries:
                wait_time = _exponential_backoff_wait(json_parse_attempt)
                logger.warning(
                    f"RAGAS JSON parse error ({error_type}) for {metric_name}. "
                    f"Backoff attempt {json_parse_attempt + 1}/{max_json_parse_retries}. Waiting {wait_time}s..."
                )
                json_parse_attempt += 1
                time.sleep(wait_time)
                continue  # Does NOT increment general_attempt
            elif _is_api_or_rate_limit_error(e):
                logger.error(f"API/quota/rate-limit error during RAGAS {metric_name} evaluation: {e}")
                raise
            else:
                general_attempt += 1
                logger.warning(f"RAGAS {metric_name} attempt {general_attempt}/{retry_count} failed ({error_type}): {e}")
                time.sleep(2)

    if df is not None and len(df) > 0:
        val = df.iloc[0].get(metric_name)
        if _is_valid_metric_score(val):
            return round(_safe(val), 3)
        if error_details is not None:
            error_details["error"] = f"Judge-LLM returned NaN or empty output for {metric_name}"

    if last_exception:
        if _is_api_or_rate_limit_error(last_exception):
            raise last_exception
        if error_details is not None:
            error_details["error"] = f"{type(last_exception).__name__}: {last_exception}"
    elif error_details is not None and "error" not in error_details:
        error_details["error"] = f"Evaluation produced no result for {metric_name}"

    return None


def _compute_benchmark_summary(questions_list: List[Dict[str, Any]], results_by_q: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall summary metrics across all processed benchmark questions."""
    total_q = len(questions_list)
    final_per_question = [results_by_q[q.get("question", "")] for q in questions_list if q.get("question", "") in results_by_q]

    # Chunk hit rate and recall: exclude questions with no expected chunk IDs
    questions_with_expected = [
        item for item in final_per_question
        if item.get("relevant_chunk_ids") and len(item["relevant_chunk_ids"]) > 0
    ]
    denom_chunks = len(questions_with_expected)

    hits_count = sum(1 for item in questions_with_expected if item.get("chunk_hit"))
    chunk_hit_rate = round(hits_count / denom_chunks, 3) if denom_chunks else None

    recalls = [item["chunk_recall"] for item in questions_with_expected if _is_valid_metric_score(item.get("chunk_recall"))]
    avg_recall = round(sum(recalls) / len(recalls), 3) if recalls else None

    faith_vals = [
        item["faithfulness"]
        for item in final_per_question
        if item.get("rag_status") == "completed" and _is_valid_metric_score(item.get("faithfulness"))
    ]
    relev_vals = [
        item["answer_relevancy"]
        for item in final_per_question
        if item.get("rag_status") == "completed" and _is_valid_metric_score(item.get("answer_relevancy"))
    ]

    faith_avg = round(sum(faith_vals) / len(faith_vals), 3) if faith_vals else None
    relev_avg = round(sum(relev_vals) / len(relev_vals), 3) if relev_vals else None
    hallucination_rate = round(1.0 - faith_avg, 3) if faith_avg is not None else None

    # Count only questions whose RAG generation is completed and whose RAGAS evaluation has both metrics successfully recorded
    completed_questions = sum(
        1 for item in final_per_question
        if item.get("rag_status") == "completed"
        and _has_both_ragas_metrics(item)
    )

    evaluated_cases = sum(
        1 for item in final_per_question
        if _is_valid_metric_score(item.get("faithfulness")) or _is_valid_metric_score(item.get("answer_relevancy"))
    )

    failed_cases = sum(
        1 for item in final_per_question
        if item.get("rag_status") == "failed"
        or (
            item.get("rag_status") == "completed"
            and not _is_valid_metric_score(item.get("faithfulness"))
            and not _is_valid_metric_score(item.get("answer_relevancy"))
            and item.get("ragas_status") in ("failed", "completed")
        )
    )

    return {
        "total_questions": total_q,
        "completed_questions": completed_questions,
        "evaluated_cases": evaluated_cases,
        "failed_cases": failed_cases,
        "faithfulness": faith_avg,
        "answer_relevancy": relev_avg,
        "hallucination_rate": hallucination_rate,
        "chunk_retrieval_hit_rate": chunk_hit_rate,
        "average_chunk_recall": avg_recall,
    }


def run_benchmark_evaluation(
    dataset_path: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
    session: Optional[Any] = None,
    delay: float = 10.0,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run end-to-end benchmark evaluation with question-level checkpointing and configurable delays.

    Features:
    1. Question-level persistence: Saves immediately after each RAG generation and each RAGAS metric.
    2. Atomic writes: Uses temporary files and os.replace to prevent file corruption upon interruption.
    3. Resumable: Skips fully completed questions, resumes individual missing metrics.
    4. Configurable delay: Waits `delay` seconds between questions after saving.
    5. Clean stoppage: Catches API/quota/rate-limit errors, records failure status, preserves progress.
    6. Per-question status tracking: rag_status and ragas_status fields with pending/completed/failed states.
    7. Initialization error handling: Safely marks incomplete status if session or evaluator init fails.
    8. Preserves complete source metadata: Keeps original rag_response.sources for citations.
    """
    from rag.pipeline import run_pipeline

    if dataset_path is None:
        dataset_path = Path(__file__).parent / "benchmark_dataset.json"
    else:
        dataset_path = Path(dataset_path)

    questions_list = _load_and_validate_dataset(dataset_path)

    if output_path is None:
        output_path = Path(__file__).parent / "benchmark_results.json"
    else:
        output_path = Path(output_path)

    logger.info(f"Loaded {len(questions_list)} benchmark questions from {dataset_path.name}")

    # ── Fresh run safety: archive old results before starting ──
    results_by_q: Dict[str, Dict[str, Any]] = {}
    if not resume:
        if output_path.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_path = output_path.with_suffix(f".backup_{ts}.json")
            try:
                os.replace(output_path, archive_path)
                logger.info(f"Archived previous results to {archive_path.name}")
            except OSError as e:
                logger.warning(f"Could not archive old results ({e}). Overwriting in place.")
        # Initialize a fresh empty checkpoint immediately so old data cannot be confused
        _atomic_write_json(output_path, {
            "total_questions": len(questions_list),
            "completed_questions": 0,
            "evaluated_cases": 0,
            "failed_cases": 0,
            "faithfulness": None,
            "answer_relevancy": None,
            "hallucination_rate": None,
            "chunk_retrieval_hit_rate": None,
            "average_chunk_recall": None,
            "benchmark_status": "running",
            "checkpoint": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "run_mode": "fresh",
            },
            "per_question": [],
        })
    else:
        # Load existing checkpoint/results if resuming
        if output_path.exists():
            try:
                existing_data = json.loads(output_path.read_text(encoding="utf-8"))
                for entry in existing_data.get("per_question", []):
                    q = entry.get("question")
                    if q:
                        _reconcile_entry_status(entry)
                        results_by_q[q] = entry
                logger.info(f"Resuming: Loaded {len(results_by_q)} existing entries from {output_path.name}")
            except Exception as e:
                logger.warning(f"Could not load existing output file for resuming ({e}). Starting fresh.")

    def _save_checkpoint(status: str = "running", stopped_reason: Optional[str] = None):
        summary = _compute_benchmark_summary(questions_list, results_by_q)
        ordered_per_question = [
            results_by_q[q.get("question", "")]
            for q in questions_list
            if q.get("question", "") in results_by_q
        ]
        complete_results = {**summary}
        complete_results["benchmark_status"] = status
        if stopped_reason:
            complete_results["stopped_reason"] = stopped_reason
        complete_results["checkpoint"] = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "run_mode": "fresh" if not resume else "resumed",
        }
        complete_results["per_question"] = ordered_per_question
        _atomic_write_json(output_path, complete_results)

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: RAG GENERATION & RETRIEVAL (Question by Question)
    # ─────────────────────────────────────────────────────────────
    need_rag = [
        q for q in questions_list
        if q.get("question") not in results_by_q
        or results_by_q[q.get("question")].get("rag_status") != "completed"
    ]

    if need_rag:
        if session is None:
            logger.info("Initializing Company KB RAG session...")
            try:
                session = _get_company_kb_session()
            except Exception as e:
                logger.error(f"Failed to initialize Company KB session: {e}")
                _save_checkpoint(
                    status="incomplete",
                    stopped_reason=f"Company KB session initialization failed: {type(e).__name__}: {e}",
                )
                raise

        for idx, item in enumerate(questions_list, 1):
            q_text = item.get("question", "")
            if not q_text:
                continue

            existing_entry = results_by_q.get(q_text)
            if existing_entry and existing_entry.get("rag_status") == "completed":
                continue  # Already generated

            ref_ans = item.get("reference_answer", "")
            expected_chunk_ids = item.get("relevant_chunk_ids", [])
            src_doc = item.get("source_document", "")
            difficulty = item.get("difficulty", "")
            eval_type = item.get("evaluation_type", "")

            logger.info(f"[{idx}/{len(questions_list)}] [RAG] Running pipeline for: {q_text[:70]}...")

            try:
                rag_response = run_pipeline(session=session, question=q_text)
            except Exception as e:
                logger.error(f"Error during RAG pipeline for question {idx}: {e}")
                # Record failure status for this question regardless of error type
                entry = existing_entry or {"question": q_text}
                entry.update({
                    "question": q_text,
                    "reference_answer": ref_ans,
                    "source_document": src_doc,
                    "difficulty": difficulty,
                    "evaluation_type": eval_type,
                    "relevant_chunk_ids": expected_chunk_ids,
                    "rag_status": "failed",
                    "rag_error": f"{type(e).__name__}: {e}",
                    "ragas_status": "pending",
                    "faithfulness": None,
                    "answer_relevancy": None,
                })
                results_by_q[q_text] = entry
                _save_checkpoint(status="incomplete", stopped_reason=f"RAG generation halted on Q{idx}: {e}")
                if _is_api_or_rate_limit_error(e):
                    logger.error(f"Stopping benchmark cleanly due to API/rate-limit error: {e}")
                    return json.loads(output_path.read_text(encoding="utf-8"))
                raise

            retrieved_contexts = [
                src.get("chunk_text") or src.get("content") or ""
                for src in rag_response.sources
                if (src.get("chunk_text") or src.get("content"))
            ]
            retrieved_chunk_ids = [
                src["chunk_id"]
                for src in rag_response.sources
                if src.get("chunk_id")
            ]

            expected_set = set(expected_chunk_ids)
            retrieved_set = set(retrieved_chunk_ids)
            matched_chunks = list(expected_set.intersection(retrieved_set))

            # When relevant_chunk_ids is empty, chunk_recall = null, chunk_hit = False
            if expected_set:
                chunk_hit = bool(matched_chunks)
                chunk_recall = round(len(matched_chunks) / len(expected_set), 3)
            else:
                chunk_hit = False
                chunk_recall = None

            # When RAG is re-run, previous RAGAS scores are stale (they were
            # computed against a different answer and/or contexts).  Invalidate
            # them so the RAGAS evaluation stage will recompute from scratch.
            #
            # Preserve original sources metadata in JSON serializable format
            raw_sources = []
            if getattr(rag_response, "sources", None):
                for src in rag_response.sources:
                    if isinstance(src, dict):
                        raw_sources.append(dict(src))

            entry = {
                "question": q_text,
                "reference_answer": ref_ans,
                "generated_answer": rag_response.answer,
                "source_document": src_doc,
                "difficulty": difficulty,
                "evaluation_type": eval_type,
                "relevant_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "matched_chunk_ids": matched_chunks,
                "chunk_hit": chunk_hit,
                "chunk_recall": chunk_recall,
                "is_grounded": rag_response.is_grounded,
                "grounding_score": rag_response.grounding_score,
                "refused": rag_response.refused,
                "retrieved_contexts": retrieved_contexts,
                "sources": raw_sources,
                "faithfulness": None,
                "answer_relevancy": None,
                "rag_status": "completed",
                "rag_error": None,
                "ragas_status": "pending",
                "ragas_error": None,
            }

            results_by_q[q_text] = entry
            _save_checkpoint(status="running")
            logger.info(f"[{idx}/{len(questions_list)}] Saved RAG checkpoint for Q{idx}")

            if delay > 0 and idx < len(questions_list):
                logger.info(f"Waiting {delay}s delay before next question...")
                time.sleep(delay)

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: RAGAS EVALUATION (Question by Question, Metric by Metric)
    # ─────────────────────────────────────────────────────────────
    need_eval = [
        q for q in questions_list
        if q.get("question") in results_by_q
        and results_by_q[q.get("question")].get("rag_status") == "completed"
        and not _has_both_ragas_metrics(results_by_q[q.get("question")])
    ]

    if need_eval:
        logger.info(f"Initializing RAGAS evaluator LLM and embeddings for {len(need_eval)} questions...")
        try:
            llm = _get_llm()
            embeds = _get_embeddings()

            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            ragas_llm = LangchainLLMWrapper(langchain_llm=llm)
            ragas_embed = LangchainEmbeddingsWrapper(embeddings=embeds)
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS evaluator: {e}")
            _save_checkpoint(
                status="incomplete",
                stopped_reason=f"RAGAS evaluator initialization failed: {type(e).__name__}: {e}",
            )
            raise

        for idx, item in enumerate(questions_list, 1):
            q_text = item.get("question", "")
            if not q_text or q_text not in results_by_q:
                continue

            entry = results_by_q[q_text]
            if entry.get("rag_status") != "completed":
                continue
            if _has_both_ragas_metrics(entry):
                entry["ragas_status"] = "completed"
                continue  # Both metrics already done

            answer = _clean_answer(entry.get("generated_answer", ""))
            contexts = _trim_contexts(entry.get("retrieved_contexts", []))

            if not answer or not contexts:
                logger.warning(f"[{idx}/{len(questions_list)}] Skipping RAGAS for Q{idx}: empty answer or contexts")
                entry["ragas_status"] = "failed"
                entry["ragas_error"] = "Empty answer or contexts"
                results_by_q[q_text] = entry
                _save_checkpoint(status="running")
                continue

            case_meta = {
                "question": q_text,
                "answer": answer,
                "contexts": contexts,
                "reference": entry.get("reference_answer"),
            }

            # Determine which metrics still need evaluation
            metrics_to_eval = []
            if not _is_valid_metric_score(entry.get("faithfulness")):
                metrics_to_eval.append("faithfulness")
            if not _is_valid_metric_score(entry.get("answer_relevancy")):
                metrics_to_eval.append("answer_relevancy")

            if not metrics_to_eval:
                entry["ragas_status"] = "completed"
                entry["ragas_error"] = None
                results_by_q[q_text] = entry
                _save_checkpoint(status="running")
                continue

            logger.info(
                f"[{idx}/{len(questions_list)}] [RAGAS] Evaluating Q{idx} "
                f"metrics={metrics_to_eval}: {q_text[:70]}..."
            )

            for metric_name in metrics_to_eval:
                metric_err: Dict[str, Any] = {}
                try:
                    score = _evaluate_single_metric_ragas(
                        case_meta=case_meta,
                        metric_name=metric_name,
                        evaluator_llm=ragas_llm,
                        evaluator_embed=ragas_embed,
                        error_details=metric_err,
                    )
                    entry[metric_name] = score

                    if score is not None:
                        logger.info(
                            f"[{idx}/{len(questions_list)}] Saved {metric_name}={score} for Q{idx}"
                        )
                        _update_ragas_status(entry)
                    else:
                        err_reason = metric_err.get("error", f"{metric_name} returned None")
                        logger.warning(
                            f"[{idx}/{len(questions_list)}] Evaluation returned None for {metric_name} on Q{idx}: {err_reason}"
                        )
                        _update_ragas_status(entry, last_error=f"{metric_name} failed: {err_reason}")

                    results_by_q[q_text] = entry
                    _save_checkpoint(status="running")
                except Exception as e:
                    logger.error(f"Error during RAGAS {metric_name} for question {idx}: {e}")
                    if _is_api_or_rate_limit_error(e):
                        _update_ragas_status(entry, last_error=f"{metric_name}: {type(e).__name__}: {e}")
                        results_by_q[q_text] = entry
                        _save_checkpoint(status="incomplete", stopped_reason=f"RAGAS {metric_name} halted on Q{idx}: {e}")
                        logger.error(f"Stopping benchmark cleanly due to API/rate-limit error: {e}")
                        return json.loads(output_path.read_text(encoding="utf-8"))
                    # Non-API error: record failure for this metric, keep others
                    _update_ragas_status(entry, last_error=f"{metric_name}: {type(e).__name__}: {e}")
                    results_by_q[q_text] = entry
                    _save_checkpoint(status="running")

            if delay > 0 and idx < len(questions_list):
                logger.info(f"Waiting {delay}s delay before next evaluation...")
                time.sleep(delay)

    # Determine final status: only 'completed' if every valid question has completed RAG and both valid metrics
    all_complete = True
    incomplete_reasons = []
    for item in questions_list:
        q_text = item.get("question", "")
        if not q_text:
            continue
        entry = results_by_q.get(q_text)
        if not entry:
            all_complete = False
            incomplete_reasons.append(f"Q '{q_text[:50]}...': no results")
        elif entry.get("rag_status") != "completed":
            all_complete = False
            incomplete_reasons.append(f"Q '{q_text[:50]}...': rag_status={entry.get('rag_status')}")
        elif not _has_both_ragas_metrics(entry):
            all_complete = False
            missing = []
            if not _is_valid_metric_score(entry.get("faithfulness")):
                missing.append("faithfulness")
            if not _is_valid_metric_score(entry.get("answer_relevancy")):
                missing.append("answer_relevancy")
            incomplete_reasons.append(f"Q '{q_text[:50]}...': missing valid metric(s) {missing}")

    if all_complete:
        _save_checkpoint(status="completed")
        logger.info(f"Benchmark evaluation completed successfully! Results saved to {output_path}")
    else:
        reason = f"{len(incomplete_reasons)} question(s) not fully complete: {'; '.join(incomplete_reasons[:5])}"
        if len(incomplete_reasons) > 5:
            reason += f" ... and {len(incomplete_reasons) - 5} more"
        _save_checkpoint(status="incomplete", stopped_reason=reason)
        logger.warning(f"Benchmark saved with incomplete status: {reason}")

    return json.loads(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on benchmark dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to benchmark_dataset.json (defaults to evaluation/benchmark_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save benchmark_results.json (defaults to evaluation/benchmark_results.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Delay in seconds between questions during RAG generation and RAGAS evaluation (default: 10.0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing output file; start fresh from beginning",
    )
    args = parser.parse_args()

    run_benchmark_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        delay=args.delay,
        resume=not args.no_resume,
    )