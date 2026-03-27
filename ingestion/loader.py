"""
ingestion/loader.py - Direct XML extraction for docx files.
Handles malformed namespace URIs and Mendeley/Zotero citation blobs.
"""

import hashlib
import re
import zipfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from configs import settings
from observability.logger import get_logger

logger = get_logger("ingestion.loader")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_docx_clean(file_path: Path) -> List[Document]:
    with zipfile.ZipFile(str(file_path), 'r') as z:
        if 'word/document.xml' not in z.namelist():
            raise RuntimeError("Not a valid docx: word/document.xml not found")
        xml = z.read('word/document.xml').decode('utf-8', errors='replace')
    text_nodes = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml, re.DOTALL)
    raw = ' '.join(t for t in text_nodes if t.strip())
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'ADDIN\s+\S+\s*\{.*?\}', ' ', raw, flags=re.DOTALL)
    raw = re.sub(r'\[\d+\]', '', raw)
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = raw.strip()
    if len(raw) < 50:
        raise RuntimeError(f"Document text too short ({len(raw)} chars).")
    logger.info(f"Extracted {len(raw)} chars from docx via direct XML method")
    return [Document(page_content=raw, metadata={"page": 0})]


def load_document(file_path, display_name=None):
    path = Path(file_path)
    ext  = path.suffix.lower()
    name = display_name or path.name
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")
    logger.info(f"Loading document: {name} ({ext})")
    try:
        if ext == ".pdf":
            docs = PyPDFLoader(str(path)).load()
        elif ext == ".txt":
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".md":
            docs = UnstructuredMarkdownLoader(str(path)).load()
        elif ext == ".docx":
            docs = _load_docx_clean(path)
        else:
            raise ValueError(f"Unhandled extension: {ext}")
    except Exception as exc:
        logger.error(f"Failed to load {name}: {exc}")
        raise RuntimeError(f"Could not parse '{name}': {exc}") from exc
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source":       name,
            "file_type":    ext.lstrip("."),
            "page":         doc.metadata.get("page", i),
            "content_hash": _content_hash(doc.page_content),
        })
        doc.metadata.pop("file_path", None)
        doc.metadata.pop("source_path", None)
    logger.info(f"Loaded {len(docs)} page(s) from '{name}'")
    return docs
