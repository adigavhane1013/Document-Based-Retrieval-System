"""
configs/settings.py

Central configuration for the RAG production system.
"""

from pathlib import Path
from typing import Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Base directory
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # ─────────────────────────────────────────────────────────────────────────
    # Application paths
    # ─────────────────────────────────────────────────────────────────────────

    BASE_DIR: Path = BASE_DIR
    STORAGE_DIR: Path = BASE_DIR / "storage"
    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
    LOG_DIR: Path = BASE_DIR / "logs"
    COMPANY_DOCS_DIR: Path = BASE_DIR / "company_documents"

    # ─────────────────────────────────────────────────────────────────────────
    # LLM configuration
    # ─────────────────────────────────────────────────────────────────────────

    LLM_API_KEY: str = Field(default="", validation_alias="GROQ_API_KEY")
    LLM_API_BASE: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "qwen/qwen3.8-27b"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1536
    LLM_CONTEXT_MAX_TOKENS: int = 4096

    # ─────────────────────────────────────────────────────────────────────────
    # Qdrant
    # ─────────────────────────────────────────────────────────────────────────

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_TIMEOUT: int = 60
    COMPANY_KB_COLLECTION: str = "company_kb"

    # ─────────────────────────────────────────────────────────────────────────
    # JWT / Authentication
    # ─────────────────────────────────────────────────────────────────────────

    JWT_SECRET_KEY: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # RAGAS evaluation
    # ─────────────────────────────────────────────────────────────────────────

    RAGAS_EVAL_MODEL: str = "openai/gpt-oss-120b"
    RAGAS_API_BASE: str = "https://api.groq.com/openai/v1"
    RAGAS_MAX_TOKENS: int = 1024
    RAGAS_MAX_CONTEXTS: int = 10
    RAGAS_CONTEXT_MAX_CHARS: int = 4000
    RAGAS_RETRY_COUNT: int = 2
    RAGAS_TIMEOUT: int = 180

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 64

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────────────────────────────────

    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 128
    CHUNK_MIN_CHARS: int = 100

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    TOP_K_DENSE: int = 25
    TOP_K_SPARSE: int = 25
    TOP_K_RERANK: int = 12
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35
    HYBRID_ALPHA: float = 0.7

    # ─────────────────────────────────────────────────────────────────────────
    # Reranking
    # ─────────────────────────────────────────────────────────────────────────

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_ENABLED: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Refusal
    # ─────────────────────────────────────────────────────────────────────────

    REFUSAL_MESSAGE: str = "I cannot answer this based on the provided documents."

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOG_DIR / "rag.log"

    # ─────────────────────────────────────────────────────────────────────────
    # LangSmith
    # ─────────────────────────────────────────────────────────────────────────

    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None
    LANGCHAIN_TRACING: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # File ingestion
    # ─────────────────────────────────────────────────────────────────────────

    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: Set[str] = {
        ".txt",
        ".md",
        ".pdf",
        ".docx",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Query rewriting
    # ─────────────────────────────────────────────────────────────────────────

    ENABLE_QUERY_REWRITING: bool = True
    QUERY_AMBIGUITY_THRESHOLD: float = 0.4
    QUERY_MIN_LENGTH: int = 5
    QUERY_MAX_REWRITE_LENGTH: int = 150
    QUERY_REWRITE_MODEL: str = "openai/gpt-oss-20b"

    # ─────────────────────────────────────────────────────────────────────────
    # RAGAS decision layer
    # ─────────────────────────────────────────────────────────────────────────

    RAGAS_FAITHFULNESS_THRESHOLD: float = 0.7
    RAGAS_RELEVANCE_THRESHOLD: float = 0.65
    RAGAS_MAX_RETRY_ATTEMPTS: int = 2
    RAGAS_RETRY_MODE: str = "better_retrieval"
    RAGAS_FALLBACK_ENABLED: bool = True
    RETRY_RETRIEVAL_TOP_K_INCREASE: int = 10
    RETRY_LOWERED_THRESHOLD: float = 0.3
    RETRY_CONTEXT_MAX_CHARS_INCREASE: int = 1000
    FALLBACK_LLM_MODEL: str | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # OpenRouter
    # ─────────────────────────────────────────────────────────────────────────

    OPENROUTER_API_KEY: str | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Pydantic settings configuration
    # ─────────────────────────────────────────────────────────────────────────

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)