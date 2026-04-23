"""
retrieval/retriever.py

Hybrid dense + BM25 retrieval with RRF (Reciprocal Rank Fusion).
"""

from typing import List, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("retrieval.retriever")


def _reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    alpha: float = settings.HYBRID_ALPHA,
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Combine dense and sparse retrieval results using Reciprocal Rank Fusion.
    
    Args:
        dense_docs: Documents from dense (vector) retrieval
        sparse_docs: Documents from sparse (BM25) retrieval
        alpha: Weight for dense results (0.7 = 70% dense, 30% sparse)
        k: RRF parameter for ranking normalization
    
    Returns:
        List of (document, score) tuples sorted by combined score
    """
    scores = {}
    doc_map = {}
    
    def _key(doc):
        return doc.metadata.get("chunk_id", doc.page_content[:64])
    
    # Dense results: weighted by alpha (70% by default)
    for rank, doc in enumerate(dense_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + alpha * (1.0 / (k + rank + 1))
        doc_map[key] = doc
    
    # Sparse results: weighted by (1-alpha) (30% by default)
    for rank, doc in enumerate(sparse_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + (1 - alpha) * (1.0 / (k + rank + 1))
        doc_map[key] = doc
    
    # Sort by combined score
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[k], scores[k]) for k in sorted_keys]


class HybridRetriever:
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) search.
    
    Uses Reciprocal Rank Fusion to combine results:
    - Dense search (70%): Fast, semantic similarity via embeddings
    - Sparse search (30%): Keyword matching via BM25
    
    This gives both semantic understanding and keyword coverage.
    """
    
    def __init__(self, vectorstore: Chroma, all_chunks: List[Document]) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            vectorstore: Chroma vector database for dense retrieval
            all_chunks: All documents for BM25 indexing
        """
        self._vectorstore = vectorstore
        self._bm25 = BM25Retriever.from_documents(all_chunks, k=settings.TOP_K_SPARSE)

    def retrieve(self, query: str) -> Tuple[List[Document], List[float]]:
        """
        Retrieve documents using hybrid dense + BM25 search.
        
        Args:
            query: User question or search query
        
        Returns:
            Tuple of (documents, scores) where:
            - documents: List of retrieved Document objects
            - scores: List of fusion scores (0-1) for each document
        
        Logs:
            - Dense retrieval results and threshold filtering
            - Sparse (BM25) retrieval results
            - Final fused results
        """
        # ── Dense retrieval (vector search) ────────────────────────────────
        dense_results = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=settings.TOP_K_DENSE
        )
        top_raw = round(dense_results[0][1], 3) if dense_results else "n/a"
        logger.info(f"Dense retrieval: {len(dense_results)} docs, top score={top_raw}")
        
        # Filter by threshold to remove low-confidence matches
        dense_filtered = [
            (doc, score) for doc, score in dense_results
            if score >= settings.RETRIEVAL_SCORE_THRESHOLD
        ]
        
        if not dense_filtered:
            logger.warning(
                f"No docs passed threshold={settings.RETRIEVAL_SCORE_THRESHOLD} "
                f"(top={top_raw}) for: {query[:80]}"
            )
            return [], []
        
        dense_docs = [d for d, _ in dense_filtered]
        logger.info(f"{len(dense_docs)} docs passed threshold")
        
        # ── Sparse retrieval (BM25 keyword search) ────────────────────────────
        sparse_docs = self._bm25.invoke(query)
        logger.info(f"Sparse retrieval: {len(sparse_docs)} docs")
        
        # ── Fusion (combine dense + sparse) ───────────────────────────────────
        fused = _reciprocal_rank_fusion(dense_docs, sparse_docs)
        docs = [d for d, _ in fused]
        scores = [s for _, s in fused]
        logger.info(f"Final hybrid result: {len(docs)} docs")
        
        return docs, scores