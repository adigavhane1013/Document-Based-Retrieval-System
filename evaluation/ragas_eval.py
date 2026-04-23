# evaluation/ragas_eval.py

import math
import os
import re
import time
from typing import Any, Dict, List

from observability.logger import get_logger
from configs import settings

logger = get_logger("evaluation.ragas")

from langchain_ollama import OllamaEmbeddings


# ─────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────

def _get_embeddings():
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.EMBEDDING_API_BASE,
    )


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
    text = re.sub(r"\[SOURCE:.*?\]", "", text)
    return text.strip()


def _trim_contexts(contexts: List[str]) -> List[str]:
    max_contexts = getattr(settings, "RAGAS_MAX_CONTEXTS", 3)
    max_chars = getattr(settings, "RAGAS_CONTEXT_MAX_CHARS", 500)

    contexts = list(dict.fromkeys(contexts))
    return [ctx[:max_chars] for ctx in contexts[:max_contexts]]


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error from Groq API."""
    error_str = str(error).lower()
    return "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str


def _exponential_backoff_wait(attempt: int, base_wait: int = 5) -> int:
    """
    Calculate exponential backoff wait time.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds
    
    Returns:
        Time to wait in seconds (base_wait * 2^attempt)
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
        base_url=settings.LLM_API_BASE,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
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

        cleaned_cases.append({
            "question": tc["question"],
            "answer": answer,
            "contexts": contexts,
        })

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
        )
        for tc in cleaned_cases
    ]

    import nest_asyncio
    nest_asyncio.apply()

    # ── UPDATED: Enhanced retry mechanism with rate limit handling ────────────
    retry_count = getattr(settings, "RAGAS_RETRY_COUNT", 2)
    max_rate_limit_retries = 3  # Allow extra retries for rate limit specifically
    last_exception = None
    rate_limit_attempt = 0

    for attempt in range(retry_count):
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=[faithfulness, answer_relevancy],
                run_config=RunConfig(max_workers=1, timeout=180),
            )
            df = result.to_pandas()
            break
            
        except Exception as e:
            is_rate_limit = _is_rate_limit_error(e)
            
            if is_rate_limit and rate_limit_attempt < max_rate_limit_retries:
                # Special handling for rate limit errors
                wait_time = _exponential_backoff_wait(rate_limit_attempt)
                logger.warning(
                    f"Rate limit hit (attempt {rate_limit_attempt + 1}). "
                    f"Waiting {wait_time} seconds before retry..."
                )
                rate_limit_attempt += 1
                time.sleep(wait_time)
                continue  # Retry without incrementing main attempt counter
            
            # Regular error handling
            logger.error(f"RAGAS attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(2)
    else:
        return {
            "error": f"Evaluation failed after {retry_count} attempts: {last_exception}",
            "evaluated_cases": 0,
        }

    # ── Process results ───────────────────────────────────────
    valid_rows = []
    failed_count = 0

    for i in range(len(df)):
        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        if f_val is None or r_val is None or math.isnan(f_val) or math.isnan(r_val):
            failed_count += 1
            continue

        valid_rows.append((f_val, r_val))

    if not valid_rows:
        return {
            "error": "All evaluations failed",
            "evaluated_cases": 0,
            "failed_cases": failed_count,
        }

    faith_avg = sum(f for f, _ in valid_rows) / len(valid_rows)
    relev_avg = sum(r for _, r in valid_rows) / len(valid_rows)

    # ── Per question breakdown ────────────────────────────────
    per_question = []
    idx = 0

    for i in range(len(df)):
        if idx >= len(valid_rows):
            break

        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        if f_val is None or r_val is None or math.isnan(f_val) or math.isnan(r_val):
            continue

        per_question.append({
            "question": cleaned_cases[i]["question"],
            "faithfulness": round(_safe(f_val), 3),
            "answer_relevancy": round(_safe(r_val), 3),
        })

        idx += 1

    # ── Final output ──────────────────────────────────────────
    return {
        "evaluated_cases": len(valid_rows),
        "failed_cases": failed_count,
        "total_cases": len(test_cases),
        "faithfulness": round(faith_avg, 3),
        "answer_relevancy": round(relev_avg, 3),
        "hallucination_rate": round(1.0 - faith_avg, 3),
        "per_question": per_question,
    }