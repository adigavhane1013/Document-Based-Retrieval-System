"""
configs/settings.py

Central configuration for the production RAG system.
Supports both:
- module-style access: from configs import settings
- instance-style access: from configs.settings import settings
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )

    # ── Paths ─────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ── LLM (Groq) ────────────────────────────────────────────
    LLM_API_KEY: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    LLM_API_BASE: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # ── RAGAS Evaluation (Groq) ───────────────────────────────
    RAGAS_EVAL_MODEL: str = "llama-3.3-70b-versatile"
    RAGAS_API_BASE: str = "https://api.groq.com/openai/v1"
    RAGAS_MAX_TOKENS: int = 1024
    RAGAS_MAX_CONTEXTS: int = 3
    RAGAS_CONTEXT_MAX_CHARS: int = 500
    RAGAS_RETRY_COUNT: int = 2
    RAGAS_TIMEOUT: int = 180

    # ── Embeddings (Ollama) ───────────────────────────────────
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_API_BASE: str = "http://localhost:11434"
    EMBEDDING_BATCH_SIZE: int = 64

    # ── Chunking ──────────────────────────────────────────────
    # FIXED: Increased from 512 to 1024 for better research paper chunking
    # Reason: Research papers with metadata (authors, affiliations) need larger chunks
    #         to avoid splitting semantic units (author lists, keywords, etc.)
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 256
    CHUNK_MIN_CHARS: int = 100

    # ── Retrieval ────────────────────────────────────────────
    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_RERANK: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.45
    HYBRID_ALPHA: float = 0.7

    # ── Reranker ─────────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_ENABLED: bool = True

    # ── Guardrails ───────────────────────────────────────────
    GROUNDING_MIN_SCORE: float = 0.5
    REFUSAL_MESSAGE: str = "I cannot answer this based on the provided documents."

    # ── Observability ────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "logs" / "rag.log"

    # LangSmith disabled for now
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    LANGCHAIN_TRACING: bool = False

    # ── Upload Limits ────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".docx"}

    # ── Query Rewriting (Phase 1 - Feature 1) ─────────────────
    ENABLE_QUERY_REWRITING: bool = True
    QUERY_AMBIGUITY_THRESHOLD: float = 0.6
    QUERY_MIN_LENGTH: int = 5
    QUERY_MAX_REWRITE_LENGTH: int = 150
    QUERY_REWRITE_MODEL: str = "llama-3.3-70b-versatile"

    # ── RAGAS Decision Layer (Phase 1 - Feature 2) ────────────
    RAGAS_FAITHFULNESS_THRESHOLD: float = 0.70
    RAGAS_RELEVANCE_THRESHOLD: float = 0.65
    RAGAS_MAX_RETRY_ATTEMPTS: int = 2
    RAGAS_RETRY_MODE: str = "better_retrieval"
    RAGAS_FALLBACK_ENABLED: bool = True
    RETRY_RETRIEVAL_TOP_K_INCREASE: int = 10
    RETRY_LOWERED_THRESHOLD: float = 0.30
    RETRY_CONTEXT_MAX_CHARS_INCREASE: int = 1000
    FALLBACK_LLM_MODEL: Optional[str] = None


# Instantiate once
settings = Settings()

# ── Module-level aliases so old imports still work ──────────
BASE_DIR = settings.BASE_DIR
STORAGE_DIR = settings.STORAGE_DIR
VECTORSTORE_DIR = settings.VECTORSTORE_DIR
LOG_DIR = settings.LOG_DIR

LLM_API_KEY = settings.LLM_API_KEY
LLM_API_BASE = settings.LLM_API_BASE
LLM_MODEL = settings.LLM_MODEL
LLM_TEMPERATURE = settings.LLM_TEMPERATURE
LLM_MAX_TOKENS = settings.LLM_MAX_TOKENS

RAGAS_EVAL_MODEL = settings.RAGAS_EVAL_MODEL
RAGAS_API_BASE = settings.RAGAS_API_BASE
RAGAS_MAX_TOKENS = settings.RAGAS_MAX_TOKENS
RAGAS_MAX_CONTEXTS = settings.RAGAS_MAX_CONTEXTS
RAGAS_CONTEXT_MAX_CHARS = settings.RAGAS_CONTEXT_MAX_CHARS
RAGAS_RETRY_COUNT = settings.RAGAS_RETRY_COUNT
RAGAS_TIMEOUT = settings.RAGAS_TIMEOUT

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_API_BASE = settings.EMBEDDING_API_BASE
EMBEDDING_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
CHUNK_MIN_CHARS = settings.CHUNK_MIN_CHARS

TOP_K_DENSE = settings.TOP_K_DENSE
TOP_K_SPARSE = settings.TOP_K_SPARSE
TOP_K_RERANK = settings.TOP_K_RERANK
RETRIEVAL_SCORE_THRESHOLD = settings.RETRIEVAL_SCORE_THRESHOLD
HYBRID_ALPHA = settings.HYBRID_ALPHA

RERANKER_MODEL = settings.RERANKER_MODEL
RERANKER_ENABLED = settings.RERANKER_ENABLED

GROUNDING_MIN_SCORE = settings.GROUNDING_MIN_SCORE
REFUSAL_MESSAGE = settings.REFUSAL_MESSAGE

LOG_LEVEL = settings.LOG_LEVEL
LOG_FILE = settings.LOG_FILE

LANGSMITH_API_KEY = settings.LANGSMITH_API_KEY
LANGSMITH_PROJECT = settings.LANGSMITH_PROJECT
LANGCHAIN_TRACING = settings.LANGCHAIN_TRACING

MAX_FILE_SIZE_MB = settings.MAX_FILE_SIZE_MB
ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS

# Query Rewriting aliases
ENABLE_QUERY_REWRITING = settings.ENABLE_QUERY_REWRITING
QUERY_AMBIGUITY_THRESHOLD = settings.QUERY_AMBIGUITY_THRESHOLD
QUERY_MIN_LENGTH = settings.QUERY_MIN_LENGTH
QUERY_MAX_REWRITE_LENGTH = settings.QUERY_MAX_REWRITE_LENGTH
QUERY_REWRITE_MODEL = settings.QUERY_REWRITE_MODEL

# RAGAS Decision Layer aliases
RAGAS_FAITHFULNESS_THRESHOLD = settings.RAGAS_FAITHFULNESS_THRESHOLD
RAGAS_RELEVANCE_THRESHOLD = settings.RAGAS_RELEVANCE_THRESHOLD
RAGAS_MAX_RETRY_ATTEMPTS = settings.RAGAS_MAX_RETRY_ATTEMPTS
RAGAS_RETRY_MODE = settings.RAGAS_RETRY_MODE
RAGAS_FALLBACK_ENABLED = settings.RAGAS_FALLBACK_ENABLED
RETRY_RETRIEVAL_TOP_K_INCREASE = settings.RETRY_RETRIEVAL_TOP_K_INCREASE
RETRY_LOWERED_THRESHOLD = settings.RETRY_LOWERED_THRESHOLD
RETRY_CONTEXT_MAX_CHARS_INCREASE = settings.RETRY_CONTEXT_MAX_CHARS_INCREASE
FALLBACK_LLM_MODEL = settings.FALLBACK_LLM_MODEL

# Extra paths
EMBEDDING_CACHE_DIR = settings.STORAGE_DIR / "embedding_cache"
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)

# Ensure required folders exist
for d in [settings.STORAGE_DIR, settings.VECTORSTORE_DIR, settings.LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)