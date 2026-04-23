"""
evaluation/cli_eval.py

Cleaned CLI evaluation script.
Fixes:
- Input cleaning (answers + contexts)
- Weak query filtering
- Context limiting
- Better reporting (including failed cases)
"""

import json
import logging
import re
from pathlib import Path
from colorama import init, Fore, Style
from configs.settings import settings
from evaluation.ragas_eval import run_ragas_evaluation

# Initialize colorama
init(autoreset=True)

# Clean logging
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger("cli_eval")

SESSIONS_FILE = settings.STORAGE_DIR / "sessions.json"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def print_header(title: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(60)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 60}\n")


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[SOURCE:.*?\]", "", text)
    return text.strip()


def _is_valid_question(q: str) -> bool:
    if not q:
        return False
    q = q.strip()

    # Reject very short / vague queries
    if len(q.split()) < 3:
        return False

    return True


def _prepare_contexts(contexts):
    max_contexts = getattr(settings, "RAGAS_MAX_CONTEXTS", 3)
    max_chars = getattr(settings, "RAGAS_CONTEXT_MAX_CHARS", 500)

    # Remove duplicates
    contexts = list(dict.fromkeys(contexts))

    trimmed = []
    for ctx in contexts[:max_contexts]:
        trimmed.append(ctx[:max_chars])

    return trimmed


# ─────────────────────────────────────────────────────────────
# MAIN CLI
# ─────────────────────────────────────────────────────────────

def run_cli_eval():
    print_header("RAGAS TERMINAL EVALUATION")

    if not SESSIONS_FILE.exists():
        print(f"{Fore.RED}No sessions found. Upload a document and ask a question first.")
        return

    try:
        sessions = json.loads(SESSIONS_FILE.read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(f"{Fore.RED}Failed to parse sessions.json: {e}")
        return

    if not sessions:
        print(f"{Fore.YELLOW}No active sessions found in storage.")
        return

    test_cases = []

    # ── Collect + Clean Test Cases ────────────────────────────
    for session_id, data in sessions.items():
        messages = data.get("messages", [])

        for msg in messages:
            if msg.get("refused", False):
                continue

            question = msg.get("question", "")
            answer = msg.get("answer", "")
            contexts = msg.get("contexts", [])

            if not question or not answer or not contexts:
                continue

            if not _is_valid_question(question):
                continue

            cleaned_answer = _clean_answer(answer)
            cleaned_contexts = _prepare_contexts(contexts)

            if not cleaned_answer or not cleaned_contexts:
                continue

            test_cases.append({
                "question": question.strip(),
                "answer": cleaned_answer,
                "contexts": cleaned_contexts,
                "session_id": session_id,
                "document": data.get("documents", ["Unknown"])[0],
            })

    if not test_cases:
        print(f"{Fore.RED}No valid cleaned test cases found.")
        return

    # ── Cap evaluation size ───────────────────────────────────
    MAX_EVAL = 5  # safer for stability
    if len(test_cases) > MAX_EVAL:
        print(f"{Fore.YELLOW}Found {len(test_cases)} valid answers. Capping to {MAX_EVAL} most recent...")
        test_cases = test_cases[-MAX_EVAL:]
    else:
        print(f"{Fore.GREEN}Found {len(test_cases)} valid answers to evaluate.")

    print(f"{Fore.CYAN}Starting RAGAS Evaluation... (30-60 seconds)\n")

    # ── Run Evaluation ────────────────────────────────────────
    try:
        results = run_ragas_evaluation(test_cases)

        if "error" in results:
            print(f"{Fore.RED}Evaluation Error: {results['error']}")
            return

        faith = results.get("faithfulness", 0.0) * 100
        relev = results.get("answer_relevancy", 0.0) * 100
        halluc = results.get("hallucination_rate", 1.0) * 100
        failed = results.get("failed_cases", 0)

        print_header("EVALUATION RESULTS")

        print(f"{Style.BRIGHT}Total Evaluated Cases: {Fore.WHITE}{results.get('evaluated_cases', 0)}")
        print(f"{Style.BRIGHT}Failed Cases:          {Fore.WHITE}{failed}")
        print(f"{Style.BRIGHT}Total Input Cases:     {Fore.WHITE}{results.get('total_cases', 0)}\n")

        # Color logic
        f_color = Fore.GREEN if faith >= 80 else Fore.YELLOW if faith >= 50 else Fore.RED
        r_color = Fore.GREEN if relev >= 80 else Fore.YELLOW if relev >= 50 else Fore.RED
        h_color = Fore.GREEN if halluc <= 20 else Fore.YELLOW if halluc <= 50 else Fore.RED

        print(f"{Style.BRIGHT}Faithfulness Metric:       {f_color}{faith:.1f}%")
        print(f"{Style.BRIGHT}Answer Relevancy Metric:   {r_color}{relev:.1f}%")
        print(f"{Style.BRIGHT}Hallucination Rate Metric: {h_color}{halluc:.1f}%\n")

        print(f"{Style.BRIGHT}--- Per Question Breakdown ---")

        for pq in results.get("per_question", []):
            q_faith = pq.get("faithfulness", 0) * 100
            q_relev = pq.get("answer_relevancy", 0) * 100

            q_faith_color = Fore.GREEN if q_faith >= 80 else Fore.YELLOW if q_faith >= 50 else Fore.RED
            q_relev_color = Fore.GREEN if q_relev >= 80 else Fore.YELLOW if q_relev >= 50 else Fore.RED

            print(f"\n{Fore.CYAN}Q: {pq['question']}")
            print(f"   Faithfulness: {q_faith_color}{q_faith:.1f}%{Fore.RESET} | Relevancy: {q_relev_color}{q_relev:.1f}%")

        print("\n")

    except ImportError:
        print(f"{Fore.RED}Missing dependencies. Install: ragas datasets langchain-openai nest-asyncio colorama")
    except Exception as e:
        print(f"{Fore.RED}Fatal evaluation error: {e}")


if __name__ == "__main__":
    run_cli_eval()