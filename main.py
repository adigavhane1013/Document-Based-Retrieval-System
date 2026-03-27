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
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs import settings
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


def _sanitize_floats(obj):
    """Recursively replace NaN/Inf with None so JSON serialization never crashes."""
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


def _load_sessions() -> Dict:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8-sig")) or {}
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
        return {}


def _save_sessions(sessions: Dict) -> None:
    try:
        SESSIONS_FILE.write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")


def _load_eval_history() -> List:
    if not EVAL_HISTORY_FILE.exists():
        return []
    try:
        return json.loads(EVAL_HISTORY_FILE.read_text(encoding="utf-8-sig")) or []
    except Exception as e:
        logger.error(f"Failed to load eval history: {e}")
        return []


def _save_eval_history(history: List) -> None:
    try:
        EVAL_HISTORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save eval history: {e}")


def _append_eval_history(entry: Dict) -> None:
    history = _load_eval_history()
    history.append(entry)
    _save_eval_history(history)


# ── In-memory state ────────────────────────────────────────────────────────────

chat_sessions: Dict[str, Dict]       = _load_sessions()
rag_sessions:  Dict[str, RAGSession] = {}


def _rebuild_session(session_id: str) -> None:
    try:
        store = load_vectorstore(session_id)
        if store is None:
            logger.warning(f"No vectorstore for session {session_id}, skipping rebuild")
            return
        chunk_cache_path = settings.STORAGE_DIR / f"chunks_{session_id}.json"
        all_chunks = []
        if chunk_cache_path.exists():
            from langchain_core.documents import Document
            raw = json.loads(chunk_cache_path.read_text())
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


def _cache_chunks(session_id: str, chunks) -> None:
    path = settings.STORAGE_DIR / f"chunks_{session_id}.json"
    existing = []
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
    new_entries = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    path.write_text(json.dumps(existing + new_entries, ensure_ascii=False), encoding='utf-8')


logger.info("Rebuilding sessions on startup...")
for sid in chat_sessions:
    _rebuild_session(sid)
logger.info(f"Loaded {len(rag_sessions)} session(s)")


# ── Pydantic models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    session_id: str
    question:   str


class SourceInfo(BaseModel):
    chunk_id:   Optional[str] = None
    source:     Optional[str] = None
    page:       Optional[int] = None
    chunk_text: Optional[str] = None


class ChatResponse(BaseModel):
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
def health():
    return {"status": "ok", "sessions_loaded": len(rag_sessions)}


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
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
            chat_sessions[sid]["documents"].append(file.filename)
            chat_sessions[sid]["documents_count"] += 1
            chat_sessions[sid]["pages"]  += len(documents)
            chat_sessions[sid]["chunks"] += len(chunks)
            chat_sessions[sid]["last_updated"] = now

        _save_sessions(chat_sessions)
        return {"session_id": sid, "filename": file.filename,
                "pages": len(documents), "chunks": len(chunks), "is_new_session": is_new}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=ChatResponse)
def ask_question(request: QuestionRequest):
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

        chat_sessions[request.session_id]["messages"].append({
            "question":        request.question,
            "answer":          result.answer,
            "grounding_score": result.grounding_score,
            "refused":         result.refused,
            "trace_id":        result.trace_id,
            "timestamp":       timestamp,
            "contexts":        [s.get("chunk_text", s.get("source", "")) for s in result.sources if s.get("chunk_text") or s.get("source")] if not result.refused and result.sources else [],
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
        if "401" in err or "Unauthorized" in err:
            raise HTTPException(401, "API key invalid")
        if "rate" in err.lower() or "limit" in err.lower():
            raise HTTPException(429, "Rate limit exceeded")
        raise HTTPException(500, f"Error generating answer: {err}")


@app.get("/sessions")
def list_sessions():
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
def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    return chat_sessions[session_id]


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
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
        raise HTTPException(500, f"Delete failed: {e}")





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)