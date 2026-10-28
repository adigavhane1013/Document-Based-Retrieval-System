"""
vectorstore/vectordb.py

Thin abstraction over Qdrant Cloud.

Design decisions:
  - Each session gets its own isolated Qdrant collection.
    This prevents cross-session retrieval leakage when users ask
    questions about their own documents.

  - score_threshold is NOT set here. It is applied by the retriever
    layer after scores are inspected.

  - Collections persist in Qdrant Cloud, enabling cold-start rebuilds
    without re-ingesting documents.

  - Qdrant stores both page_content and metadata for every LangChain
    Document. The stored documents can therefore be reconstructed for
    BM25 retrieval after uploads and application restarts.
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


def _client(timeout: Optional[int] = None) -> QdrantClient:
    """Create and return a Qdrant client."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=timeout or getattr(settings, "QDRANT_TIMEOUT", 60),
    )


def _collection_name(session_id: str) -> str:
    """Return the Qdrant collection name for a session."""
    return f"session_{session_id}"


def create_vectorstore(
    session_id: str,
    chunks: List[Document],
) -> QdrantVectorStore:
    """
    Create a new Qdrant collection for a session and index the given chunks.

    Warning:
        force_recreate=True intentionally replaces an existing collection.
        This function should therefore only be used when creating a new
        session.
    """
    collection_name = _collection_name(session_id)
    embeddings = get_embedding_model()

    logger.info(
        f"Creating Qdrant collection '{collection_name}' "
        f"with {len(chunks)} chunks"
    )

    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True,
        distance=qmodels.Distance.COSINE,
    )

    logger.info(
        f"Qdrant collection '{collection_name}' created"
    )

    return store


def load_vectorstore(
    session_id: str,
) -> Optional[QdrantVectorStore]:
    """
    Load an existing Qdrant collection.

    Returns:
        QdrantVectorStore if the session collection exists,
        otherwise None.
    """
    collection_name = _collection_name(session_id)
    client = _client()

    if not client.collection_exists(collection_name):
        logger.warning(
            f"No Qdrant collection found for session {session_id}"
        )
        return None

    embeddings = get_embedding_model()

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def get_all_documents(session_id: str) -> List[Document]:
    """
    Retrieve all stored LangChain Documents for a session.

    Qdrant stores the original page_content and metadata as payload.
    This function reconstructs those Documents so they can be used by
    in-memory retrievers such as BM25.

    This is intentionally separate from load_vectorstore():
        - load_vectorstore() provides dense vector retrieval.
        - get_all_documents() reconstructs the complete document corpus
          required by BM25.

    Returns:
        A list containing every stored Document in the session.

    Raises:
        RuntimeError:
            If the session collection does not exist or the stored payload
            cannot be reconstructed.
    """
    collection_name = _collection_name(session_id)
    client = _client()

    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"No Qdrant collection found for session {session_id}"
        )

    documents: List[Document] = []
    next_page_offset = None

    logger.info(
        f"Loading all stored documents for session {session_id}"
    )

    while True:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            page_content = payload.get("page_content")
            metadata = payload.get("metadata", {})

            if page_content is None:
                logger.warning(
                    f"Skipping Qdrant point {point.id} in session "
                    f"{session_id}: missing page_content"
                )
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        if next_page_offset is None:
            break

    logger.info(
        f"Loaded {len(documents)} stored documents "
        f"for session {session_id}"
    )

    return documents


def add_to_vectorstore(
    session_id: str,
    chunks: List[Document],
) -> QdrantVectorStore:
    """
    Add chunks to an existing session's Qdrant collection.

    If the session does not have a collection yet, a new collection
    is created.
    """
    store = load_vectorstore(session_id)

    if store is None:
        return create_vectorstore(session_id, chunks)

    logger.info(
        f"Adding {len(chunks)} chunks to existing collection "
        f"for session {session_id}"
    )

    store.add_documents(chunks)

    return store


def delete_vectorstore(session_id: str) -> None:
    """Remove a session's Qdrant collection."""
    collection_name = _collection_name(session_id)
    client = _client()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

        logger.info(
            f"Deleted Qdrant collection for session {session_id}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Company Knowledge Base collection
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_KB_COLLECTION: str = getattr(settings, "COMPANY_KB_COLLECTION", "company_kb")


def company_kb_exists() -> bool:
    """Check whether the dedicated company_kb collection exists in Qdrant."""
    client = _client()
    return client.collection_exists(COMPANY_KB_COLLECTION)


def create_company_kb_vectorstore(
    chunks: List[Document],
    force_recreate: bool = False,
    ids: Optional[List[str]] = None,
) -> QdrantVectorStore:
    """
    Create or populate the dedicated company_kb Qdrant collection.

    Chunks are uploaded in batches of settings.EMBEDDING_BATCH_SIZE to avoid
    Qdrant write timeouts when indexing the full Company KB corpus.

    Args:
        chunks: List of LangChain Document chunks with policy/section metadata.
        force_recreate: If True, drops any existing collection and recreates it.
        ids: Optional deterministic point IDs (UUID strings) to avoid duplicates.

    Returns:
        QdrantVectorStore connected to the company_kb collection.
    """
    client = _client()
    embeddings = get_embedding_model()

    batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(
        f"Indexing {len(chunks)} chunks into dedicated Qdrant collection "
        f"'{COMPANY_KB_COLLECTION}' (force_recreate={force_recreate}) "
        f"in {total_batches} batch(es) of {batch_size}"
    )

    if force_recreate or not client.collection_exists(COMPANY_KB_COLLECTION):
        # Create the collection with the first batch, then upsert the rest.
        first_batch = chunks[:batch_size]
        first_ids = ids[:batch_size] if ids else None

        kwargs = {
            "documents": first_batch,
            "embedding": embeddings,
            "url": settings.QDRANT_URL,
            "api_key": settings.QDRANT_API_KEY,
            "collection_name": COMPANY_KB_COLLECTION,
            "force_recreate": True,
            "distance": qmodels.Distance.COSINE,
        }
        if first_ids:
            kwargs["ids"] = first_ids
        store = QdrantVectorStore.from_documents(**kwargs)
        logger.info(
            f"Qdrant collection '{COMPANY_KB_COLLECTION}' created with "
            f"{len(first_batch)} chunks (batch 1/{total_batches})"
        )

        for i in range(batch_size, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            batch_num = i // batch_size + 1
            store.add_documents(batch_chunks, ids=batch_ids)
            logger.info(
                f"Upserted batch {batch_num}/{total_batches} "
                f"({len(batch_chunks)} chunks) into '{COMPANY_KB_COLLECTION}'"
            )

        return store

    # Collection already exists and force_recreate is False: load and upsert in batches
    store = QdrantVectorStore(
        client=client,
        collection_name=COMPANY_KB_COLLECTION,
        embedding=embeddings,
    )
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = ids[i:i + batch_size] if ids else None
        batch_num = i // batch_size + 1
        store.add_documents(batch_chunks, ids=batch_ids)
        logger.info(
            f"Upserted batch {batch_num}/{total_batches} "
            f"({len(batch_chunks)} chunks) into '{COMPANY_KB_COLLECTION}'"
        )

    logger.info(
        f"Upserted {len(chunks)} chunks into existing '{COMPANY_KB_COLLECTION}'"
    )
    return store


def load_company_kb_vectorstore() -> Optional[QdrantVectorStore]:
    """
    Load the existing company_kb Qdrant collection.

    Returns:
        QdrantVectorStore if the company_kb collection exists, otherwise None.
    """
    client = _client()
    if not client.collection_exists(COMPANY_KB_COLLECTION):
        logger.warning(
            f"No Qdrant collection found for '{COMPANY_KB_COLLECTION}'"
        )
        return None

    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=COMPANY_KB_COLLECTION,
        embedding=embeddings,
    )


def get_all_company_kb_documents() -> List[Document]:
    """
    Retrieve all stored LangChain Documents from the company_kb collection.
    Used for reconstructing the in-memory BM25 index.
    """
    client = _client()
    if not client.collection_exists(COMPANY_KB_COLLECTION):
        raise RuntimeError(
            f"No Qdrant collection found for '{COMPANY_KB_COLLECTION}'"
        )

    documents: List[Document] = []
    next_page_offset = None

    logger.info(
        f"Loading all stored documents for '{COMPANY_KB_COLLECTION}'"
    )

    while True:
        points, next_page_offset = client.scroll(
            collection_name=COMPANY_KB_COLLECTION,
            limit=100,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            page_content = payload.get("page_content")
            metadata = payload.get("metadata", {})

            if page_content is None:
                logger.warning(
                    f"Skipping Qdrant point {point.id} in "
                    f"{COMPANY_KB_COLLECTION}: missing page_content"
                )
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        if next_page_offset is None:
            break

    logger.info(
        f"Loaded {len(documents)} stored documents "
        f"from '{COMPANY_KB_COLLECTION}'"
    )
    return documents