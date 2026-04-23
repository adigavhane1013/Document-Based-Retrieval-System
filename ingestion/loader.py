"""
ingestion/loader.py - FIXED VERSION
Uses pdfplumber instead of PyPDFLoader to preserve document structure.
"""

import hashlib
import re
import zipfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)

from configs.settings import settings
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


def _load_pdf_with_structure(file_path: Path) -> List[Document]:
    """
    Load PDF using pdfplumber to preserve document structure.
    
    pdfplumber maintains:
    - Paragraph breaks (\n\n)
    - Section spacing
    - Text layout information
    
    This produces better chunks than PyPDFLoader.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(file_path)).load()
    
    docs = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            logger.info(f"Extracting text from {len(pdf.pages)} page(s) with pdfplumber")
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if not text or len(text.strip()) < 50:
                    logger.warning(f"Page {page_num} produced minimal text ({len(text)} chars)")
                    continue
                
                # Enhance paragraph breaks: add \n before common section headers
                # Resume sections typically don't have explicit spacing in PDFs
                section_keywords = [
                    "Education", "Experience", "Projects", "Skills", "Certifications",
                    "Work Experience", "Technical Skills", "Internship", "Summary",
                    "Background", "Expertise", "Employment", "Accomplishments"
                ]
                
                for keyword in section_keywords:
                    # Convert "keyword\n" to "\n\nkeyword\n" to create paragraph breaks
                    text = re.sub(
                        rf'(\n)({keyword})',
                        r'\n\n\2',
                        text,
                        flags=re.IGNORECASE
                    )
                
                # Clean up excessive whitespace but preserve paragraph breaks
                text = re.sub(r'[ \t]{2,}', ' ', text)  # Multiple spaces → single space
                text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines → 2 newlines
                text = text.strip()
                
                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num}
                ))
                
                logger.info(f"Page {page_num}: Extracted {len(text)} chars, {len(text.split())} words")
    
    except Exception as exc:
        logger.error(f"pdfplumber extraction failed: {exc}")
        # Fallback to PyPDFLoader
        logger.warning("Falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(file_path)).load()
    
    if not docs:
        raise RuntimeError("No text could be extracted from PDF")
    
    logger.info(f"Successfully extracted {len(docs)} page(s) from PDF")
    return docs


def load_document(file_path, display_name=None):
    """
    Load a document from file.
    
    Supports: .pdf, .txt, .md, .docx
    
    Returns:
        List[Document]: Loaded documents with metadata
    """
    path = Path(file_path)
    ext  = path.suffix.lower()
    name = display_name or path.name
    
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")
    
    logger.info(f"Loading document: {name} ({ext})")
    
    try:
        if ext == ".pdf":
            # Use pdfplumber for better structure preservation
            docs = _load_pdf_with_structure(path)
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
    
    # Add consistent metadata
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source":       name,
            "file_type":    ext.lstrip("."),
            "page":         doc.metadata.get("page", i),
            "content_hash": _content_hash(doc.page_content),
        })
        # Clean up unnecessary metadata
        doc.metadata.pop("file_path", None)
        doc.metadata.pop("source_path", None)
    
    logger.info(f"Loaded {len(docs)} page(s) from '{name}'")
    return docs