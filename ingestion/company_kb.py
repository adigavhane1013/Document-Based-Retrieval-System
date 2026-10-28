"""
ingestion/company_kb.py

Dedicated ingestion and structure-aware hierarchical chunking module for the
Company Knowledge Base (Nexora Technologies policies).

Architecture:
  Six Markdown policies (company_documents/)
          ↓
  Markdown ingestion (load_document / UnstructuredMarkdownLoader)
          ↓
  Structure-aware / hierarchical chunking
          ↓
  Rich policy + section metadata
          ↓
  Dedicated company_kb Qdrant collection
"""

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from configs.settings import settings
from ingestion.loader import load_document
from observability.logger import get_logger
from vectorstore.vectordb import (
    COMPANY_KB_COLLECTION,
    company_kb_exists,
    create_company_kb_vectorstore,
)

logger = get_logger("ingestion.company_kb")

# Tiktoken encoder for token counting (model-agnostic cl100k_base)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """Return token count for a text string using cl100k_base."""
    return len(_ENCODER.encode(text))


def _deterministic_chunk_id(policy_id: str, chunk_index: int, content: str) -> str:
    """Generate a stable, deterministic 12-char chunk ID."""
    fingerprint = f"{policy_id}::{chunk_index}::{content[:64]}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


def _deterministic_point_id(policy_id: str, chunk_id: str) -> str:
    """Generate a deterministic UUID string for Qdrant vector store point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"company_kb::{policy_id}::{chunk_id}"))


def extract_policy_metadata(raw_markdown: str, filename: str) -> Dict[str, Any]:
    """
    Extract document-level metadata from the policy header table and headings.

    Extracts:
        - policy_name / document_name
        - policy_id / document_id
        - version / document_version
        - effective_date
        - next_scheduled_review
        - owner
        - approved_by
        - classification
        - applies_to
        - source (basename)

    Where the source document provides the information, extracts it directly.
    Does not invent metadata values.
    """
    header_part = raw_markdown.split("## 1.")[0]

    # Policy Title from '## <Title>'
    title_match = re.search(r"^##\s+(.+)$", header_part, re.MULTILINE)
    policy_name = title_match.group(1).strip() if title_match else Path(filename).stem

    # Header metadata table: | Field | Detail |
    table_lines = re.findall(
        r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", header_part, re.MULTILINE
    )
    meta_table: Dict[str, str] = {}
    for k, v in table_lines:
        k_clean = k.strip()
        v_clean = v.strip()
        if k_clean not in meta_table and k_clean not in ("Field", "---"):
            meta_table[k_clean] = v_clean

    policy_id = meta_table.get("Document ID", "")
    version = meta_table.get("Version", "")
    effective_date = meta_table.get("Effective Date", "")
    next_review = meta_table.get("Next Scheduled Review", "")
    owner = meta_table.get("Document Owner", "")
    approved_by = meta_table.get("Approved By", "")
    classification = meta_table.get("Classification", "Internal")
    applies_to = meta_table.get("Applies To", "")

    return {
        "policy_name": policy_name,
        "document_name": policy_name,
        "policy_id": policy_id,
        "document_id": policy_id,
        "version": version,
        "document_version": version,
        "effective_date": effective_date,
        "next_scheduled_review": next_review,
        "owner": owner,
        "approved_by": approved_by,
        "classification": classification,
        "applies_to": applies_to,
        "source": Path(filename).name,
    }


def chunk_company_document(
    file_path: Path,
    raw_markdown: Optional[str] = None,
    max_chunk_tokens: int = 600,
    min_chunk_chars: int = 100,
) -> List[Document]:
    """
    Split a Company Policy Markdown document using structure-aware hierarchical chunking.

    Hierarchy:
        document
        → section (## 1. Purpose, ## 2. Scope, etc.)
        → subsection (2.1, 2.2, etc.)
        → paragraph/table/list
        → recursive fallback when content is too large

    Preserves:
        - Top-level sections
        - Subsections such as 3.1, 3.2, etc.
        - Related paragraphs together where practical
        - Tables and lists associated with their surrounding section
        - Recursive token-aware fallback if content exceeds CHUNK_SIZE
        - Filter out tiny chunks (< min_chunk_chars)
    """
    path = Path(file_path)
    if raw_markdown is None:
        raw_markdown = path.read_text(encoding="utf-8")

    meta = extract_policy_metadata(raw_markdown, path.name)
    policy_name = meta["policy_name"]
    policy_id = meta["policy_id"]

    # Split document by top-level sections: '\n## <number>. '
    section_splits = re.split(r"\n(?=## \d+\. )", raw_markdown)
    raw_chunks: List[Tuple[str, str, str, str]] = []  # (content, section, subsection, heading_path)

    # 1. Header / Document Overview
    overview_text = section_splits[0].strip()
    if overview_text:
        overview_sec = "Document Overview"
        overview_path = f"{policy_name} > {overview_sec}"
        raw_chunks.append((overview_text, overview_sec, "", overview_path))

    # 2. Numbered sections
    for sec_raw in section_splits[1:]:
        sec_text = sec_raw.strip()
        if not sec_text:
            continue

        lines = [l for l in sec_text.splitlines() if l.strip()]
        sec_header = lines[0]  # e.g. "## 1. Purpose"
        sec_title = sec_header.replace("## ", "").strip()

        # Find subsection boundaries like '\n2.1 ', '\n3.2 '
        sub_matches = list(re.finditer(r"\n(?=(\d+\.\d+)\s+)", sec_text))

        sec_tokens = _token_len(sec_text)

        # Case A: No subsections OR the entire section fits comfortably within max_chunk_tokens
        if not sub_matches or sec_tokens <= max_chunk_tokens:
            sub_nums = re.findall(r"\n(\d+\.\d+)\s+", sec_text)
            if len(sub_nums) > 1:
                sub_label = f"{sub_nums[0]} - {sub_nums[-1]}"
            elif len(sub_nums) == 1:
                sub_label = sub_nums[0]
            else:
                sub_label = ""

            heading_path = (
                f"{policy_name} > {sec_title}"
                + (f" ({sub_label})" if sub_label else "")
            )
            raw_chunks.append((sec_text, sec_title, sub_label, heading_path))

        else:
            # Case B: Section exceeds max_chunk_tokens and contains subsections.
            # Group subsections cohesively up to max_chunk_tokens.
            first_sub_pos = sub_matches[0].start()
            intro = sec_text[:first_sub_pos].strip()

            current_block: List[str] = []
            if len(intro) > len(sec_header) and _token_len(intro) >= 30:
                current_block.append(intro)

            current_subs: List[Tuple[str, str]] = []
            sub_blocks = re.split(
                r"\n(?=\d+\.\d+\s+)", sec_text[first_sub_pos:]
            )

            for sb in sub_blocks:
                sb_str = sb.strip()
                if not sb_str:
                    continue

                m = re.match(r"^(\d+\.\d+)\s+(\*\*(.+?)\*\*\.?\s*)?", sb_str)
                sub_num = m.group(1) if m else ""
                sub_name = m.group(3).strip() if (m and m.group(3)) else ""

                candidate_content = "\n\n".join(current_block + [sb_str])
                candidate_tokens = _token_len(f"{sec_header}\n\n{candidate_content}")

                if candidate_tokens <= max_chunk_tokens or not current_block:
                    current_block.append(sb_str)
                    if sub_num:
                        current_subs.append((sub_num, sub_name))
                else:
                    # Flush current block
                    chunk_body = f"{sec_header}\n\n" + "\n\n".join(current_block)
                    if len(current_subs) > 1:
                        sub_lbl = f"{current_subs[0][0]} - {current_subs[-1][0]}"
                    elif len(current_subs) == 1:
                        sub_lbl = (
                            f"{current_subs[0][0]} {current_subs[0][1]}".strip()
                        )
                    else:
                        sub_lbl = ""

                    h_path = f"{policy_name} > {sec_title}" + (
                        f" > {sub_lbl}" if sub_lbl else ""
                    )
                    raw_chunks.append((chunk_body, sec_title, sub_lbl, h_path))

                    # Start next block
                    current_block = [sb_str]
                    current_subs = [(sub_num, sub_name)] if sub_num else []

            # Flush trailing block
            if current_block:
                chunk_body = f"{sec_header}\n\n" + "\n\n".join(current_block)
                if len(current_subs) > 1:
                    sub_lbl = f"{current_subs[0][0]} - {current_subs[-1][0]}"
                elif len(current_subs) == 1:
                    sub_lbl = (
                        f"{current_subs[0][0]} {current_subs[0][1]}".strip()
                    )
                else:
                    sub_lbl = ""

                h_path = f"{policy_name} > {sec_title}" + (
                    f" > {sub_lbl}" if sub_lbl else ""
                )
                raw_chunks.append((chunk_body, sec_title, sub_lbl, h_path))

    # 3. Recursive fallback for any chunk exceeding settings.CHUNK_SIZE
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=_token_len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
        is_separator_regex=False,
    )

    final_chunks: List[Tuple[str, str, str, str]] = []
    for content, sec, sub, h_path in raw_chunks:
        if _token_len(content) > settings.CHUNK_SIZE:
            sub_splits = fallback_splitter.split_text(content)
            for part_idx, part in enumerate(sub_splits):
                part_h_path = f"{h_path} (part {part_idx + 1})"
                final_chunks.append((part, sec, sub, part_h_path))
        else:
            final_chunks.append((content, sec, sub, h_path))

    # 4. Filter degenerate tiny chunks
    valid_chunks = [
        (c, sec, sub, h)
        for c, sec, sub, h in final_chunks
        if len(c.strip()) >= min_chunk_chars
    ]

    total_chunks = len(valid_chunks)
    documents: List[Document] = []

    for idx, (content, sec, sub, h_path) in enumerate(valid_chunks):
        chunk_id = _deterministic_chunk_id(policy_id, idx, content)
        token_count = _token_len(content)

        chunk_meta: Dict[str, Any] = {
            # Identification & indexing
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "chunk_total": total_chunks,
            "token_count": token_count,
            "source": meta["source"],
            # Policy-level metadata
            "policy_id": meta["policy_id"],
            "document_id": meta["document_id"],
            "policy_name": meta["policy_name"],
            "document_name": meta["document_name"],
            "version": meta["version"],
            "document_version": meta["document_version"],
            "effective_date": meta["effective_date"],
            "owner": meta["owner"],
            "classification": meta["classification"],
            # Section & hierarchy metadata
            "section": sec,
            "subsection": sub,
            "section_hierarchy": h_path,
            "heading_path": h_path,
        }

        # Sanitize metadata to primitives for Qdrant payload
        safe_meta: Dict[str, Any] = {}
        for k, v in chunk_meta.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            elif v is not None:
                safe_meta[k] = str(v)

        documents.append(
            Document(
                page_content=content,
                metadata=safe_meta,
            )
        )

    logger.info(
        f"Document '{meta['source']}' ({policy_id}) chunked into "
        f"{len(documents)} structure-aware chunks "
        f"(avg {_token_len(''.join(d.page_content for d in documents)) // max(len(documents), 1)} tokens/chunk)"
    )

    return documents


def load_and_chunk_company_documents(
    docs_dir: Optional[Path] = None,
) -> List[Document]:
    """
    Load all policy documents from company_documents/ and apply structure-aware chunking.

    Uses existing ingestion infrastructure (load_document / UnstructuredMarkdownLoader)
    to validate and inspect files, then structures and chunks them.

    Returns:
        List of enriched LangChain Document chunks across all company policies.
    """
    directory = docs_dir or settings.COMPANY_DOCS_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Company documents directory not found: {directory}")

    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown documents found in {directory}")

    all_chunks: List[Document] = []

    for file_path in md_files:
        logger.info(f"Processing company document: {file_path.name}")
        # Validate through existing loader infrastructure
        _ = load_document(file_path)

        # Structure-aware chunking
        raw_text = file_path.read_text(encoding="utf-8")
        chunks = chunk_company_document(file_path, raw_markdown=raw_text)
        all_chunks.extend(chunks)

    logger.info(
        f"Successfully loaded and chunked {len(md_files)} company policies "
        f"into {len(all_chunks)} total chunks"
    )
    return all_chunks


def ingest_company_kb(
    force_recreate: bool = False,
    docs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ingest all company documents into the dedicated company_kb Qdrant collection.

    Idempotent and repeatable:
        - When force_recreate is True, the collection is cleanly replaced.
        - When force_recreate is False, chunks are indexed with deterministic UUIDs
          so re-running never creates duplicate vectors.

    Returns:
        Summary dict containing ingestion statistics.
    """
    logger.info(
        f"Starting Company Knowledge Base ingestion (force_recreate={force_recreate})"
    )

    chunks = load_and_chunk_company_documents(docs_dir=docs_dir)

    # Generate deterministic point IDs for Qdrant
    point_ids = [
        _deterministic_point_id(
            c.metadata.get("policy_id", "kb"),
            c.metadata.get("chunk_id", str(i)),
        )
        for i, c in enumerate(chunks)
    ]

    create_company_kb_vectorstore(
        chunks=chunks,
        force_recreate=force_recreate,
        ids=point_ids,
    )

    # Compile policy summary
    policy_counts: Dict[str, int] = {}
    for c in chunks:
        pid = c.metadata.get("policy_id", "unknown")
        policy_counts[pid] = policy_counts.get(pid, 0) + 1

    summary = {
        "status": "success",
        "collection_name": COMPANY_KB_COLLECTION,
        "total_documents": len(set(c.metadata.get("source") for c in chunks)),
        "total_chunks": len(chunks),
        "policy_chunks": policy_counts,
        "avg_tokens_per_chunk": sum(c.metadata.get("token_count", 0) for c in chunks) // max(len(chunks), 1),
        "force_recreate": force_recreate,
    }

    logger.info(f"Company Knowledge Base ingestion completed: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ingest company documents into dedicated company_kb Qdrant collection"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate the company_kb collection in Qdrant",
    )
    args = parser.parse_args()

    result = ingest_company_kb(force_recreate=args.force)
    print("\n--- Company Knowledge Base Ingestion Summary ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
