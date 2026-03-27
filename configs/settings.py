"""
configs/settings.py
Central configuration for the production RAG system.
All tunable parameters live here — never scatter magic numbers across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent.parent
STORAGE_DIR       = BASE_DIR / "storage"
VECTORSTORE_DIR   = BASE_DIR / "vectorstore"
LOG_DIR           = BASE_DIR / "logs"

for d in [STORAGE_DIR, VECTORSTORE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── LLM (Groq — free tier, used for RAG answering) ────────────────────────────
LLM_API_KEY       = os.getenv("GROQ_API_KEY")
LLM_API_BASE      = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Keep temperature 0.0 for faithful RAG — never exceed 0.3
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1024"))


# ── RAGAS Evaluation LLM (Groq — same key, free tier) ─────────────────────────
RAGAS_EVAL_MODEL  = os.getenv("RAGAS_EVAL_MODEL", "llama-3.3-70b-versatile")
RAGAS_API_BASE    = os.getenv("RAGAS_API_BASE", "https://api.groq.com/openai/v1")
RAGAS_MAX_TOKENS  = int(os.getenv("RAGAS_MAX_TOKENS", "512"))


# ── Embeddings (Ollama — local, free) ─────────────────────────────────────────
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_API_BASE   = os.getenv("EMBEDDING_API_BASE", "http://localhost:11434")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_CACHE_DIR  = STORAGE_DIR / "embedding_cache"
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)


# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNK_MIN_CHARS   = int(os.getenv("CHUNK_MIN_CHARS", "100"))


# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_DENSE       = int(os.getenv("TOP_K_DENSE", "20"))
TOP_K_SPARSE      = int(os.getenv("TOP_K_SPARSE", "20"))
TOP_K_RERANK      = int(os.getenv("TOP_K_RERANK", "5"))

RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.45"))
HYBRID_ALPHA      = float(os.getenv("HYBRID_ALPHA", "0.7"))


# ── Reranker ───────────────────────────────────────────────────────────────────
RERANKER_MODEL    = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED  = os.getenv("RERANKER_ENABLED", "true").lower() == "true"


# ── Guardrails ─────────────────────────────────────────────────────────────────
GROUNDING_MIN_SCORE  = float(os.getenv("GROUNDING_MIN_SCORE", "0.5"))
REFUSAL_MESSAGE      = "I cannot answer this based on the provided documents."


# ── Observability ──────────────────────────────────────────────────────────────
LANGSMITH_API_KEY    = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT    = os.getenv("LANGSMITH_PROJECT", "rag-production")
LANGCHAIN_TRACING    = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

LOG_LEVEL            = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE             = LOG_DIR / "rag.log"


# ── Upload Limits ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB     = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS   = {".pdf", ".txt", ".md", ".docx"}