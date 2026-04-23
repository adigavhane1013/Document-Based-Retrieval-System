"""
main.py

FastAPI entrypoint. Thin HTTP layer only — no business logic.
"""

import json
import math
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs.settings import settings
from ingestion.loader import load_document
from ingestion.chunking import chunk_documents
from vectorstore.vectordb import (
    add_to_vectorstore,
    create_vectorstore,
    delete_vectorstore,
    load_vectorstore,
)
from retrieval.retriever import HybridRetriever
from rag.pipeline import RAGSession, RAGResponse, run_pipeline
from observability.logger import get_logger

logger = get_logger("main")


def _sanitize_floats(obj: Any) -> Any:
    """
    Recursively replace NaN/Inf with None so JSON serialization never crashes.
    
    Args:
        obj: Object to sanitize (can be float, dict, list, or other)
    
    Returns:
        Sanitized object with no NaN/Inf values
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Production RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Persistence helpers ────────────────────────────────────────────────────────

SESSIONS_FILE     = settings.STORAGE_DIR / "sessions.json"
EVAL_HISTORY_FILE = settings.STORAGE_DIR / "eval_history.json"


def _load_sessions() -> Dict[str, Any]:
    """Load all chat sessions from disk."""
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8-sig")) or {}
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
        return {}


def _save_sessions(sessions: Dict[str, Any]) -> None:
    """Save all chat sessions to disk."""
    try:
        SESSIONS_FILE.write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")


def _load_eval_history() -> List[Dict[str, Any]]:
    """Load evaluation history from disk."""
    if not EVAL_HISTORY_FILE.exists():
        return []
    try:
        return json.loads(EVAL_HISTORY_FILE.read_text(encoding="utf-8-sig")) or []
    except Exception as e:
        logger.error(f"Failed to load eval history: {e}")
        return []


def _save_eval_history(history: List[Dict[str, Any]]) -> None:
    """Save evaluation history to disk."""
    try:
        EVAL_HISTORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save eval history: {e}")


# ── In-memory state ────────────────────────────────────────────────────────────

chat_sessions: Dict[str, Dict[str, Any]] = _load_sessions()
rag_sessions:  Dict[str, RAGSession]     = {}


def _rebuild_session(session_id: str) -> None:
    """
    Rebuild a RAG session from persisted vectorstore and chunks cache.
    
    Args:
        session_id: Session ID to rebuild
    """
    try:
        store = load_vectorstore(session_id)
        if store is None:
            logger.warning(f"No vectorstore for session {session_id}, skipping rebuild")
            return
        
        chunk_cache_path = settings.STORAGE_DIR / f"chunks_{session_id}.json"
        all_chunks = []
        
        if chunk_cache_path.exists():
            from langchain_core.documents import Document
            raw = json.loads(chunk_cache_path.read_text(encoding="utf-8"))
            all_chunks = [
                Document(page_content=c["content"], metadata=c["metadata"])
                for c in raw
            ]
        
        retriever = HybridRetriever(vectorstore=store, all_chunks=all_chunks)
        rag_sessions[session_id] = RAGSession(
            session_id=session_id,
            retriever=retriever,
            all_chunks=all_chunks,
        )
        logger.info(f"Rebuilt session {session_id}")
    except Exception as e:
        logger.error(f"Failed to rebuild session {session_id}: {e}")


def _cache_chunks(session_id: str, chunks: List) -> None:
    """
    Cache chunks to disk for session recovery.
    
    Args:
        session_id: Session ID
        chunks: List of Document chunks to cache
    """
    path = settings.STORAGE_DIR / f"chunks_{session_id}.json"
    existing = []
    
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load existing chunks cache: {e}")
            existing = []
    
    new_entries = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    
    try:
        path.write_text(
            json.dumps(existing + new_entries, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to cache chunks for session {session_id}: {e}")


logger.info("Rebuilding sessions on startup...")
for sid in chat_sessions:
    _rebuild_session(sid)
logger.info(f"Loaded {len(rag_sessions)} session(s)")


# ── Pydantic models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    session_id: str
    question:   str


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""
    chunk_id:   Optional[str] = None
    source:     Optional[str] = None
    page:       Optional[int] = None
    chunk_text: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat/question endpoint."""
    answer:          str
    is_grounded:     bool
    grounding_score: float
    refused:         bool
    sources_count:   int
    sources:         List[SourceInfo]
    trace_id:        str
    timestamp:       str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "sessions_loaded": len(rag_sessions)}


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Upload a document and add to RAG system.
    
    Supports PDF, TXT, MD, DOCX files.
    Can create new session or append to existing session.
    
    Args:
        file: Document file to upload
        session_id: Optional existing session ID to append to
    
    Returns:
        Dict with session_id, filename, pages, chunks, is_new_session
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'")

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large ({size_mb:.1f}MB > {settings.MAX_FILE_SIZE_MB}MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        documents = load_document(tmp_path, display_name=file.filename)
        chunks    = chunk_documents(documents)

        if not chunks:
            raise HTTPException(400, "Could not extract any text from the document.")

        is_new = not session_id or session_id not in chat_sessions
        sid    = session_id if not is_new else str(uuid.uuid4())

        store = create_vectorstore(sid, chunks) if is_new else add_to_vectorstore(sid, chunks)
        _cache_chunks(sid, chunks)

        existing_chunks = rag_sessions[sid].all_chunks if sid in rag_sessions else []
        all_chunks = existing_chunks + chunks
        retriever  = HybridRetriever(vectorstore=store, all_chunks=all_chunks)
        rag_sessions[sid] = RAGSession(session_id=sid, retriever=retriever, all_chunks=all_chunks)

        now = datetime.now().isoformat()
        if is_new:
            chat_sessions[sid] = {
                "documents": [file.filename],
                "documents_count": 1,
                "pages":     len(documents),
                "chunks":    len(chunks),
                "created_at":  now,
                "last_updated": now,
                "messages":    [],
            }
        else:
            # FIXED: Use .get() with default to avoid KeyError
            chat_sessions[sid]["documents"].append(file.filename)
            chat_sessions[sid]["documents_count"] = chat_sessions[sid].get("documents_count", 0) + 1
            chat_sessions[sid]["pages"]  = chat_sessions[sid].get("pages", 0) + len(documents)
            chat_sessions[sid]["chunks"] = chat_sessions[sid].get("chunks", 0) + len(chunks)
            chat_sessions[sid]["last_updated"] = now

        _save_sessions(chat_sessions)
        return {
            "session_id": sid,
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "is_new_session": is_new
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=ChatResponse)
def ask_question(request: QuestionRequest) -> ChatResponse:
    """
    Ask a question about uploaded documents.
    
    Args:
        request: QuestionRequest with session_id and question
    
    Returns:
        ChatResponse with answer, grounding_score, sources, trace_id
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if request.session_id not in rag_sessions:
        if request.session_id in chat_sessions:
            _rebuild_session(request.session_id)
        if request.session_id not in rag_sessions:
            raise HTTPException(404, "Session not found")

    try:
        session   = rag_sessions[request.session_id]
        result: RAGResponse = run_pipeline(session, request.question)
        timestamp = datetime.now().isoformat()

        # FIXED: Improved context extraction with better filtering
        contexts = []
        if not result.refused and result.sources:
            for s in result.sources:
                chunk_text = s.get("chunk_text", "").strip()
                source = s.get("source", "").strip()
                if chunk_text or source:
                    contexts.append(chunk_text if chunk_text else source)

        # FIXED: Ensure message list exists before appending
        if request.session_id not in chat_sessions:
            logger.warning(f"Session {request.session_id} disappeared during ask")
            raise HTTPException(404, "Session not found")
        
        if "messages" not in chat_sessions[request.session_id]:
            chat_sessions[request.session_id]["messages"] = []

        chat_sessions[request.session_id]["messages"].append({
            "question":        request.question,
            "answer":          result.answer,
            "grounding_score": result.grounding_score,
            "refused":         result.refused,
            "trace_id":        result.trace_id,
            "timestamp":       timestamp,
            "contexts":        contexts,
        })
        chat_sessions[request.session_id]["last_updated"] = timestamp
        _save_sessions(chat_sessions)

        return ChatResponse(
            answer=result.answer,
            is_grounded=result.is_grounded,
            grounding_score=result.grounding_score,
            refused=result.refused,
            sources_count=len(result.sources),
            sources=[SourceInfo(**s) for s in result.sources],
            trace_id=result.trace_id,
            timestamp=timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        logger.error(f"Ask error: {e}")
        
        # FIXED: Check for specific error patterns instead of just string matching
        if "401" in err or "Unauthorized" in err:
            raise HTTPException(401, "API key invalid")
        if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower():
            raise HTTPException(429, "Rate limit exceeded")
        if "timeout" in err.lower():
            raise HTTPException(504, "Request timeout")
        
        raise HTTPException(500, f"Error generating answer: {err}")


@app.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    """
    List all sessions with their metadata.
    
    Returns:
        Dict with "sessions" key containing list of session summaries
    """
    return {
        "sessions": [
            {
                "session_id":      sid,
                "documents":       data.get("documents", []),
                "documents_count": data.get("documents_count", 0),
                "created_at":      data.get("created_at"),
                "last_updated":    data.get("last_updated"),
                "message_count":   len(data.get("messages", [])),
                "pages":           data.get("pages", 0),
                "chunks":          data.get("chunks", 0),
            }
            for sid, data in chat_sessions.items()
        ]
    }


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific session.
    
    Args:
        session_id: Session ID to retrieve
    
    Returns:
        Full session data with all messages and metadata
    """
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    return chat_sessions[session_id]


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """
    Delete a session and all its data.
    
    Args:
        session_id: Session ID to delete
    
    Returns:
        Confirmation message
    """
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    try:
        chat_sessions.pop(session_id, None)
        rag_sessions.pop(session_id, None)
        delete_vectorstore(session_id)
        (settings.STORAGE_DIR / f"chunks_{session_id}.json").unlink(missing_ok=True)
        _save_sessions(chat_sessions)
        return {"message": "Session deleted"}
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(500, f"Delete failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)