"""
ingestion/__init__.py
"""

from ingestion.loader import load_document
from ingestion.chunking import chunk_documents
from ingestion.company_kb import (
    chunk_company_document,
    load_and_chunk_company_documents,
    ingest_company_kb,
)

__all__ = [
    "load_document",
    "chunk_documents",
    "chunk_company_document",
    "load_and_chunk_company_documents",
    "ingest_company_kb",
]
