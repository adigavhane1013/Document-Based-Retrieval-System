"""
evaluation/cli_eval.py

A command-line script to evaluate RAG sessions directly from the terminal.
Iterates over stored chat sessions, extracts the last N questions and their contexts,
and evaluates them using RAGAS.
"""

import json
import logging
from pathlib import Path
from colorama import init, Fore, Style

from configs import settings
from evaluation.ragas_eval import run_ragas_evaluation

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure logging to output cleanly
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger("cli_eval")

SESSIONS_FILE = settings.STORAGE_DIR / "sessions.json"

def print_header(title: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(60)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 60}\n")

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
    
    # Collect test cases from all sessions
    for session_id, data in sessions.items():
        messages = data.get("messages", [])
        for msg in messages:
            if not msg.get("refused", True) and msg.get("answer"):
                # Ensure contexts exist, else cannot evaluate accurately
                contexts = msg.get("contexts", [])
                if not contexts:
                    continue
                
                test_cases.append({
                    "question": msg["question"],
                    "answer": msg["answer"],
                    "contexts": contexts,
                    "session_id": session_id,
                    "document": data.get("documents", ["Unknown Document"])[0]
                })

    if not test_cases:
        print(f"{Fore.RED}No valid answered questions with context found across any session.")
        return

    # Cap to avoid extreme timeouts
    MAX_EVAL = 8
    if len(test_cases) > MAX_EVAL:
        print(f"{Fore.YELLOW}Found {len(test_cases)} valid answers. Capping to {MAX_EVAL} most recent to respect API limits...")
        test_cases = test_cases[-MAX_EVAL:]
    else:
        print(f"{Fore.GREEN}Found {len(test_cases)} valid answers to evaluate.")

    print(f"{Fore.CYAN}Starting RAGAS Evaluation... (This may take 30-60 seconds)\n")
    
    try:
        results = run_ragas_evaluation(test_cases)
        
        if "error" in results:
            print(f"{Fore.RED}Evaluation Error: {results['error']}")
            return

        faith = results.get("faithfulness", 0.0) * 100
        relev = results.get("answer_relevancy", 0.0) * 100
        halluc = results.get("hallucination_rate", 1.0) * 100
        
        print_header("EVALUATION RESULTS")
        
        print(f"{Style.BRIGHT}Total Evaluated Cases: {Fore.WHITE}{results.get('evaluated_cases', 0)}")
        print(f"{Style.BRIGHT}Total Refused Cases:   {Fore.WHITE}{results.get('refused_cases', 0)}\n")

        # Color coding logic
        f_color = Fore.GREEN if faith >= 80 else Fore.YELLOW if faith >= 50 else Fore.RED
        r_color = Fore.GREEN if relev >= 80 else Fore.YELLOW if relev >= 50 else Fore.RED
        h_color = Fore.GREEN if halluc <= 20 else Fore.YELLOW if halluc <= 50 else Fore.RED
        
        print(f"{Style.BRIGHT}Faithfulness Metric:       {f_color}{faith:.1f}%")
        print(f"{Style.BRIGHT}Answer Relevancy Metric:   {r_color}{relev:.1f}%")
        print(f"{Style.BRIGHT}Hallucination Rate Metric: {h_color}{halluc:.1f}% \n")

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
        print(f"{Fore.RED}RAGAS dependencies missing. Please install: ragas datasets langchain-google-genai nest-asyncio colorama")
    except Exception as e:
        print(f"{Fore.RED}Fatal evaluation error: {e}")

if __name__ == "__main__":
    run_cli_eval()
