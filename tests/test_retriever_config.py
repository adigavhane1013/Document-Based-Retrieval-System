from unittest.mock import Mock

from langchain_core.documents import Document

from retrieval.config import RetrievalConfig
from retrieval.retriever import HybridRetriever


def test_retriever_uses_supplied_config():
    """Test that retriever respects config for dense/sparse top-k, threshold, and RRF dedup."""
    vectorstore = Mock()

    # Dense returns 2 docs: doc1 (high score), doc2 (low score)
    vectorstore.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="document 1", metadata={"chunk_id": "1"}), 0.80),
        (Document(page_content="document 2", metadata={"chunk_id": "2"}), 0.20),
    ]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._vectorstore = vectorstore

    bm25 = Mock()

    # BM25 returns 3 docs: doc3, doc4, doc1 (different order from dense)
    # This lets us verify top_k_sparse is respected independently
    bm25.invoke.return_value = [
        Document(page_content="document 3", metadata={"chunk_id": "3"}),
        Document(page_content="document 4", metadata={"chunk_id": "4"}),
        Document(page_content="document 1", metadata={"chunk_id": "1"}),
    ]

    retriever._bm25 = bm25

    config = RetrievalConfig(
        top_k_dense=2,
        top_k_sparse=1,  # Only first BM25 doc should be used
        score_threshold=0.50,
    )

    docs, scores = retriever.retrieve("test query", config)

    # ──────────────────────────────────────────────
    # Verify dense top-k
    # ──────────────────────────────────────────────
    vectorstore.similarity_search_with_relevance_scores.assert_called_once_with(
        "test query", k=2
    )

    # ──────────────────────────────────────────────
    # Verify sparse retrieval was called
    # ──────────────────────────────────────────────
    bm25.invoke.assert_called_once_with("test query")

    # ──────────────────────────────────────────────
    # Verify sparse top-k: only first BM25 doc (doc3) used
    # ──────────────────────────────────────────────
    # Dense passed: doc1 (0.80 >= 0.50), doc2 filtered (0.20 < 0.50)
    # BM25 sliced to top_k_sparse=1: only doc3
    # RRF fusion: doc1 (dense only), doc3 (sparse only)
    # Result should have 2 docs: doc1 and doc3
    assert len(docs) == 2, f"Expected 2 docs (doc1 from dense, doc3 from sparse), got {len(docs)}"
    chunk_ids = [doc.metadata["chunk_id"] for doc in docs]
    assert "1" in chunk_ids, "doc1 from dense should be present"
    assert "3" in chunk_ids, "doc3 from sparse (top_k_sparse=1) should be present"
    assert "4" not in chunk_ids, "doc4 should NOT be present (top_k_sparse=1)"
    assert "2" not in chunk_ids, "doc2 should NOT be present (filtered by threshold)"

    # ──────────────────────────────────────────────
    # Verify RRF deduplication: same doc in dense & sparse
    # ──────────────────────────────────────────────
    # Create a second test case where same doc appears in both
    bm25_dedup = Mock()
    bm25_dedup.invoke.return_value = [
        Document(page_content="document 1", metadata={"chunk_id": "1"}),  # Same as dense doc1
        Document(page_content="document 5", metadata={"chunk_id": "5"}),
    ]
    retriever._bm25 = bm25_dedup

    docs_dedup, scores_dedup = retriever.retrieve("test query", config)

    # Dense: doc1 (passes threshold)
    # Sparse top_k=1: doc1 only
    # RRF: doc1 appears in both -> should be deduplicated to single entry
    assert len(docs_dedup) == 1, f"Expected 1 deduped doc, got {len(docs_dedup)}"
    assert docs_dedup[0].metadata["chunk_id"] == "1"
    assert len(scores_dedup) == 1

    # ──────────────────────────────────────────────
    # Verify score threshold filters dense results
    # ──────────────────────────────────────────────
    # Already verified above: doc2 (score 0.20) not in results
    # because threshold = 0.50
    assert all(doc.metadata["chunk_id"] != "2" for doc in docs_dedup)