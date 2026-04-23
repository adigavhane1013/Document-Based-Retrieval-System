"""
embeddings/embedding_model.py

Free local embeddings using Ollama (nomic-embed-text model).
No API key required — runs 100% on your machine.

Setup (one time):
    1. Download Ollama from https://ollama.com/download
    2. Run in terminal: ollama pull nomic-embed-text
    3. Ollama runs as a background service automatically

Why nomic-embed-text:
    - Free, runs locally via Ollama
    - 768 dimensions, strong performance on retrieval benchmarks
    - Comparable to text-embedding-3-small for RAG use cases
    - No rate limits, no cost, no internet needed after first pull
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings

from configs import settings
from observability.logger import get_logger

logger = get_logger("embeddings.model")


# ── Disk cache ─────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """Simple SHA-256 keyed file cache for embedding vectors."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits   = 0
        self._misses = 0

    def _key(self, model: str, text: str) -> str:
        return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> List[float] | None:
        path = self.cache_dir / f"{self._key(model, text)}.json"
        if path.exists():
            self._hits += 1
            return json.loads(path.read_text())
        self._misses += 1
        return None

    def set(self, model: str, text: str, vector: List[float]) -> None:
        path = self.cache_dir / f"{self._key(model, text)}.json"
        path.write_text(json.dumps(vector))

    @property
    def stats(self):
        total = self._hits + self._misses
        rate  = (self._hits / total * 100) if total else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate_pct": round(rate, 1)}


_cache = EmbeddingCache(settings.EMBEDDING_CACHE_DIR)


# ── Embedding model ────────────────────────────────────────────────────────────

def get_embedding_model() -> OllamaEmbeddings:
    """
    Return a configured OllamaEmbeddings instance.
    Ollama must be running locally (it starts automatically after install).
    """
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,   # nomic-embed-text
        base_url=settings.EMBEDDING_API_BASE,  # http://localhost:11434
    )


def embed_texts_cached(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts with disk caching.
    Cache hits are returned immediately without calling Ollama.
    """
    model    = settings.EMBEDDING_MODEL
    results  = [None] * len(texts)
    to_embed = []

    for i, text in enumerate(texts):
        cached = _cache.get(model, text)
        if cached is not None:
            results[i] = cached
        else:
            to_embed.append((i, text))

    if to_embed:
        embedder    = get_embedding_model()
        batch_texts = [t for _, t in to_embed]
        vectors     = _embed_with_retry(embedder, batch_texts)

        for (orig_idx, text), vector in zip(to_embed, vectors):
            _cache.set(model, text, vector)
            results[orig_idx] = vector

    logger.info(f"Embedding stats: {_cache.stats}")
    return results


def _embed_with_retry(embedder: OllamaEmbeddings, texts: List[str],
                      retries: int = 3, delay: float = 2.0) -> List[List[float]]:
    for attempt in range(1, retries + 1):
        try:
            return embedder.embed_documents(texts)
        except Exception as exc:
            if attempt == retries:
                logger.error(f"Embedding failed after {retries} attempts: {exc}")
                raise
            logger.warning(f"Embedding attempt {attempt} failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay * attempt)
    raise RuntimeError("Unreachable")