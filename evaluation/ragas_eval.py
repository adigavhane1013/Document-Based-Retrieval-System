"""
evaluation/ragas_eval.py

RAGAS evaluation using Groq (llama-3.3-70b-versatile) as the eval LLM.
Same key as RAG answering — no extra API keys needed.
Metrics: Faithfulness + Answer Relevancy only.
No ground truth required.
"""
import math
import os
import time
from typing import Any, Dict, List
from observability.logger import get_logger

logger = get_logger("evaluation.ragas")


def _safe(val, default=0.0):
    try:
        return default if (val is None or math.isnan(val) or math.isinf(val)) else float(val)
    except Exception:
        return default


def _get_llm():
    """Groq LLM via langchain_openai — same key as RAG pipeline."""
    from langchain_openai import ChatOpenAI
    from configs import settings

    api_key = settings.LLM_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your .env file")

    return ChatOpenAI(
        model=settings.RAGAS_EVAL_MODEL,
        openai_api_key=api_key,
        openai_api_base=settings.RAGAS_API_BASE,
        temperature=0.0,
        max_tokens=1024,
        model_kwargs={"n": 1},
    )


def _get_embeddings():
    """Ollama local embeddings — free, no API call."""
    from langchain_ollama import OllamaEmbeddings
    from configs import settings
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.EMBEDDING_API_BASE,
    )


def run_ragas_evaluation(
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run RAGAS Faithfulness + Answer Relevancy evaluation.
    Each test case: { question, answer, contexts: [str, ...] }
    """
    import ragas
    logger.info(f"RAGAS version: {ragas.__version__}")

    # Block RAGAS from falling back to OpenAI
    os.environ.pop("OPENAI_API_KEY", None)

    # Filter out refused/empty answers
    valid = [
        tc for tc in test_cases
        if tc.get("answer")
        and "cannot answer" not in tc["answer"].lower()
        and tc.get("contexts")
    ]

    if not valid:
        return {
            "error":           "No valid (non-refused) test cases to evaluate",
            "total_cases":     len(test_cases),
            "evaluated_cases": 0,
            "refused_cases":   len(test_cases),
        }

    logger.info(f"Evaluating {len(valid)}/{len(test_cases)} valid cases")

    llm    = _get_llm()
    embeds = _get_embeddings()

    try:
        # ── New API: ragas >= 0.2 ─────────────────────────────────────────────
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.run_config import RunConfig
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm   = LangchainLLMWrapper(langchain_llm=llm)
        ragas_embed = LangchainEmbeddingsWrapper(embeddings=embeds)

        faithfulness     = Faithfulness(llm=ragas_llm)
        answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embed)

        samples = [
            SingleTurnSample(
                user_input=tc["question"],
                response=tc["answer"],
                retrieved_contexts=tc["contexts"],
            )
            for tc in valid
        ]

        import nest_asyncio
        nest_asyncio.apply()

        # max_workers=1 forces sequential calls — avoids hitting Groq rate limits
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[faithfulness, answer_relevancy],
            run_config=RunConfig(max_workers=1, timeout=120),
        )
        df = result.to_pandas()

    except ImportError:
        # ── Old API fallback: ragas 0.1.x ─────────────────────────────────────
        logger.warning("Falling back to old RAGAS API (ragas < 0.2)")
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness as f_metric, answer_relevancy as r_metric

        f_metric.llm = llm
        r_metric.llm = llm
        r_metric.embeddings = embeds

        import nest_asyncio
        nest_asyncio.apply()

        data = {
            "question": [tc["question"] for tc in valid],
            "answer":   [tc["answer"]   for tc in valid],
            "contexts": [tc["contexts"] for tc in valid],
        }
        result = ragas_evaluate(Dataset.from_dict(data), metrics=[f_metric, r_metric])
        df     = result.to_pandas()

    # ── Column helpers ─────────────────────────────────────────────────────────
    faith_col = "faithfulness"     if "faithfulness"     in df.columns else None
    relev_col = "answer_relevancy" if "answer_relevancy" in df.columns else None

    if not faith_col:
        for col in df.columns:
            if col.lower().endswith("faithfulness"): faith_col = col
    if not relev_col:
        for col in df.columns:
            if col.lower().endswith("answer_relevancy"): relev_col = col

    faith_avg = _safe(df[faith_col].mean()) if faith_col else 0.0
    relev_avg = _safe(df[relev_col].mean()) if relev_col else 0.0

    per_question = [
        {
            "question":         valid[i]["question"],
            "faithfulness":     round(_safe(df.iloc[i][faith_col] if faith_col else 0), 3),
            "answer_relevancy": round(_safe(df.iloc[i][relev_col] if relev_col else 0), 3),
        }
        for i in range(len(valid))
    ]

    scores: Dict[str, Any] = {
        "evaluated_cases":    len(valid),
        "total_cases":        len(test_cases),
        "refused_cases":      len(test_cases) - len(valid),
        "faithfulness":       round(faith_avg, 3),
        "answer_relevancy":   round(relev_avg, 3),
        "hallucination_rate": round(1.0 - faith_avg, 3),
        "per_question":       per_question,
    }

    logger.info(
        f"RAGAS done: faithfulness={scores['faithfulness']}, "
        f"relevancy={scores['answer_relevancy']}, "
        f"hallucination={scores['hallucination_rate']}"
    )
    return scores