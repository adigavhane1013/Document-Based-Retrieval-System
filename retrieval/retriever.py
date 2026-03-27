from typing import List, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from configs import settings
from observability.logger import get_logger

logger = get_logger("retrieval.retriever")

def _reciprocal_rank_fusion(dense_docs, sparse_docs, alpha=settings.HYBRID_ALPHA, k=60):
    scores = {}
    doc_map = {}
    def _key(doc):
        return doc.metadata.get("chunk_id", doc.page_content[:64])
    for rank, doc in enumerate(dense_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + alpha * (1.0 / (k + rank + 1))
        doc_map[key] = doc
    for rank, doc in enumerate(sparse_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + (1 - alpha) * (1.0 / (k + rank + 1))
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[k], scores[k]) for k in sorted_keys]

class HybridRetriever:
    def __init__(self, vectorstore, all_chunks):
        self._vectorstore = vectorstore
        self._bm25 = BM25Retriever.from_documents(all_chunks, k=settings.TOP_K_SPARSE)

    def retrieve(self, query):
        dense_results = self._vectorstore.similarity_search_with_relevance_scores(query, k=settings.TOP_K_DENSE)
        top_raw = round(dense_results[0][1], 3) if dense_results else "n/a"
        logger.info(f"Dense retrieval: {len(dense_results)} docs, top score={top_raw}")
        dense_filtered = [(doc, score) for doc, score in dense_results if score >= settings.RETRIEVAL_SCORE_THRESHOLD]
        if not dense_filtered:
            logger.warning(f"No docs passed threshold={settings.RETRIEVAL_SCORE_THRESHOLD} (top={top_raw}) for: {query[:80]}")
            return [], []
        dense_docs = [d for d, _ in dense_filtered]
        logger.info(f"{len(dense_docs)} docs passed threshold")
        sparse_docs = self._bm25.invoke(query)
        logger.info(f"Sparse retrieval: {len(sparse_docs)} docs")
        fused = _reciprocal_rank_fusion(dense_docs, sparse_docs)
        docs = [d for d, _ in fused]
        scores = [s for _, s in fused]
        logger.info(f"Final hybrid result: {len(docs)} docs")
        return docs, scores
