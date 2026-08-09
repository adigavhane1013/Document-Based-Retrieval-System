"""
vectorstore/vectordb.py

Thin abstraction over Qdrant Cloud (replaces ChromaDB).

Design decisions:
  - Each session gets its own isolated Qdrant collection (not a global one).
    This prevents cross-session retrieval leakage when users ask questions
    about their own documents.
  - score_threshold is NOT set here. It is applied by the retriever layer
    after scores are inspected — this gives us visibility into what got
    filtered and why.
  - Collections persist in Qdrant Cloud, enabling cold-start rebuilds
    without re-ingesting (same behaviour as the old per-session Chroma dirs).
"""

from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from configs.settings import settings
from embeddings.embedding_model import get_embedding_model
from observability.logger import get_logger

logger = get_logger("vectorstore.db")


def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def _collection_name(session_id: str) -> str:
    return f"session_{session_id}"


def create_vectorstore(session_id: str, chunks: List[Document]) -> QdrantVectorStore:
    """
    Create a new Qdrant collection for a session and index the given chunks.
    Overwrites any existing collection for the same session_id.
    """
    collection_name = _collection_name(session_id)
    embeddings = get_embedding_model()

    logger.info(f"Creating Qdrant collection '{collection_name}' with {len(chunks)} chunks")

    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True,
        distance=qmodels.Distance.COSINE,
    )
    logger.info(f"Qdrant collection '{collection_name}' created")
    return store


def load_vectorstore(session_id: str) -> Optional[QdrantVectorStore]:
    """
    Load an existing Qdrant collection.
    Returns None if no collection exists for the given session_id.
    """
    collection_name = _collection_name(session_id)
    client = _client()

    if not client.collection_exists(collection_name):
        logger.warning(f"No Qdrant collection found for session {session_id}")
        return None

    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def add_to_vectorstore(session_id: str, chunks: List[Document]) -> QdrantVectorStore:
    """
    Add chunks to an existing session's Qdrant collection.
    If the session doesn't have one yet, creates it.
    """
    store = load_vectorstore(session_id)
    if store is None:
        return create_vectorstore(session_id, chunks)

    logger.info(f"Adding {len(chunks)} chunks to existing collection for session {session_id}")
    store.add_documents(chunks)
    return store


def delete_vectorstore(session_id: str) -> None:
    """Remove a session's Qdrant collection."""
    collection_name = _collection_name(session_id)
    client = _client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info(f"Deleted Qdrant collection for session {session_id}")