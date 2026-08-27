"""
main.py

FastAPI entrypoint. Thin HTTP layer only — no business logic.

UPDATED: Decision layer integration with on-demand RAGAS evaluation.
- /ask now accepts optional 'run_evaluation' parameter
- If run_evaluation=true, runs RAGAS on-demand and passes scores to pipeline
- Decision layer then accepts/retries/fallbacks based on quality scores
- Responses include decision_metadata showing why answer was accepted/rejected
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
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
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
from evaluation.ragas_eval import run_ragas_evaluation
from observability.logger import get_logger
from auth import (
    init_auth_db,
    get_current_user,
    CurrentUser,
)

logger = get_logger("main")

init_auth_db()


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


# ── Global session storage ─────────────────────────────────────────────────────
rag_sessions = {}
chat_sessions = _load_sessions()


def _rebuild_session(session_id: str) -> None:
    """Rebuild a RAGSession from stored chat data."""
    if session_id not in chat_sessions:
        return
    
    vectorstore = load_vectorstore(session_id)
    if vectorstore is None:
        logger.warning(f"Could not load vectorstore for session {session_id}")
        return
    
    # Note: HybridRetriever requires non-empty all_chunks for BM25 init
    # Skip retriever creation here - it will be created fresh on next /ask call
    logger.debug(f"Vectorstore loaded for session {session_id}")


# Load all sessions on startup
for sid in chat_sessions:
    _rebuild_session(sid)


# ── Request/Response Models ────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    session_id: str
    question:   str
    run_evaluation: bool = False  # NEW: Optional on-demand RAGAS evaluation


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""
    chunk_id:   Optional[str] = None
    source:     Optional[str] = None
    page:       Optional[int] = None
    chunk_text: Optional[str] = None


class DecisionInfo(BaseModel):
    """Decision layer metadata."""
    decision_type: str  # ACCEPT, RETRY, FALLBACK, REJECT
    attempt: int
    max_attempts: int
    reason: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None


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
    decision_metadata: Optional[DecisionInfo] = None  # NEW: Decision layer output


class EvaluateRequest(BaseModel):
    """Request model for the optional RAGAS evaluation endpoint."""
    session_id: str
    trace_id:   str


class EvaluateResponse(BaseModel):
    """Response model for the optional RAGAS evaluation endpoint."""
    trace_id:           str
    faithfulness:       Optional[float] = None
    answer_relevancy:   Optional[float] = None
    hallucination_rate: Optional[float] = None
    cached:             bool


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "sessions_loaded": len(rag_sessions)}


# ── Auth ─────────────────────────────────────────────────────────────────
# Registration and login happen client-side via the Firebase JS SDK
# (docmind_ui.html). The backend only verifies the Firebase ID token
# on every protected request — see auth.get_current_user.


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload a PDF or DOCX document.

    Creates a new session if session_id is not provided.
    """
    if not file:
        raise HTTPException(400, "No file provided")

    # ── Validate file type ──────────────────────────────────────────────────
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type {file_ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # ── Validate file size ──────────────────────────────────────────────────
    file_content = file.file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            413,
            f"File size {file_size_mb:.1f}MB exceeds limit {settings.MAX_FILE_SIZE_MB}MB"
        )

    # ── Generate session_id if not provided ─────────────────────────────────
    if not session_id:
        session_id = str(uuid.uuid4())
    is_new = session_id not in chat_sessions

    tmp_path = None
    try:
        # ── Write to temp file ──────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            suffix=file_ext, delete=False
        ) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # ── Load and chunk document ─────────────────────────────────────────
        docs = load_document(tmp_path)
        if not docs:
            raise ValueError("Document contains no extractable text")

        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError("No valid chunks after processing")

        logger.info(
            f"Loaded {len(docs)} doc(s), created {len(chunks)} chunk(s) "
            f"for session {session_id}"
        )

        # ── Create or add to vectorstore ─────────────────────────────────────
        # Use add_to_vectorstore for a session that already has documents,
        # so new files extend the knowledge base instead of wiping it —
        # create_vectorstore always recreates the collection from scratch.
        is_existing_session = session_id in chat_sessions
        if is_existing_session:
            vectorstore = add_to_vectorstore(session_id, chunks)
        else:
            vectorstore = create_vectorstore(session_id, chunks)
        
        # ── Create RAGSession ──────────────────────────────────────────────
        retriever = HybridRetriever(vectorstore=vectorstore, all_chunks=chunks)
        rag_sessions[session_id] = RAGSession(
            session_id=session_id,
            retriever=retriever,
            all_chunks=chunks,
        )

        # ── Initialize chat session ─────────────────────────────────────────
        timestamp = datetime.now().isoformat()
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "session_id": session_id,
                "user_id": current_user.user_id,
                "created_at": timestamp,
                "last_updated": timestamp,
                "filename": file.filename,
                "filenames": [file.filename],
                "file_size_mb": round(file_size_mb, 2),
                "document_count": len(docs),
                "chunk_count": len(chunks),
                "messages": [],
            }
        else:
            if chat_sessions[session_id].get("user_id") != current_user.user_id:
                raise HTTPException(403, "This session belongs to another user")
            # Accumulate counts — this file's chunks were ADDED, not replacing prior ones
            existing = chat_sessions[session_id]
            existing["document_count"] = existing.get("document_count", 0) + len(docs)
            existing["chunk_count"] = existing.get("chunk_count", 0) + len(chunks)
            existing.setdefault("filenames", [existing.get("filename")] if existing.get("filename") else [])
            existing["filenames"].append(file.filename)
            existing["last_updated"] = timestamp
        _save_sessions(chat_sessions)

        return {
            "session_id": session_id,
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "document_count": len(docs),
            "chunk_count": len(chunks),
            "created_at": timestamp,
            "is_new_session": is_new
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True) if tmp_path else None


@app.post("/ask", response_model=ChatResponse)
def ask_question(
    request: QuestionRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    """
    Ask a question about uploaded documents.
    
    NEW: Supports optional on-demand RAGAS evaluation.
    If run_evaluation=true, evaluates answer quality and passes scores
    to decision layer for Accept/Retry/Fallback/Reject logic.
    
    Args:
        request: QuestionRequest with session_id, question, and optional run_evaluation
    
    Returns:
        ChatResponse with answer, grounding_score, sources, trace_id, decision_metadata
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if request.session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    if chat_sessions[request.session_id].get("user_id") != current_user.user_id:
        raise HTTPException(403, "This session belongs to another user")

    if request.session_id not in rag_sessions:
        _rebuild_session(request.session_id)
        if request.session_id not in rag_sessions:
            raise HTTPException(404, "Session not found")

    try:
        session = rag_sessions[request.session_id]

        # One trace_id per question/answer turn — reused below if the
        # pipeline is re-run with eval_scores, so /evaluate can find this
        # exact turn later regardless of how many other questions this
        # session has. Must NOT be session_id (that repeats every turn).
        trace_id = str(uuid.uuid4())

        # ── Run pipeline (no evaluation yet) ────────────────────────────────
        result: RAGResponse = run_pipeline(session, request.question, trace_id=trace_id)
        timestamp = datetime.now().isoformat()

        # ── Optional: Run RAGAS evaluation and re-run decision layer ────────
        eval_scores = None
        if request.run_evaluation and not result.refused:
            logger.info(f"Running on-demand RAGAS evaluation for trace {result.trace_id}")
            
            # Extract contexts from sources
            contexts = []
            if result.sources:
                for s in result.sources:
                    chunk_text = s.get("chunk_text", "").strip()
                    if chunk_text:
                        contexts.append(chunk_text)
            
            if contexts:
                try:
                    test_case = {
                        "question": request.question,
                        "answer": result.answer,
                        "contexts": contexts,
                    }
                    
                    eval_result = run_ragas_evaluation([test_case])
                    
                    if eval_result.get("error") or eval_result.get("evaluated_cases", 0) == 0:
                        logger.warning(f"RAGAS evaluation had issues: {eval_result.get('error')}")
                    else:
                        # Extract scores for decision layer
                        eval_scores = {
                            "faithfulness": eval_result.get("faithfulness", 0.0),
                            "answer_relevancy": eval_result.get("answer_relevancy", 0.0),
                        }
                        
                        logger.info(
                            f"RAGAS scores: faithfulness={eval_scores['faithfulness']:.2f}, "
                            f"relevancy={eval_scores['answer_relevancy']:.2f}"
                        )
                        
                        # ── Re-run decision layer with evaluation scores ────────
                        result_with_decision: RAGResponse = run_pipeline(
                            session, 
                            request.question,
                            eval_scores=eval_scores,
                            trace_id=trace_id,
                        )
                        
                        # Use decision-enhanced result
                        result = result_with_decision
                        
                except Exception as e:
                    logger.error(f"RAGAS evaluation failed: {e}")
                    logger.info("Continuing without evaluation")
            else:
                logger.warning("No contexts available for RAGAS evaluation")

        # ── Extract contexts for storage ────────────────────────────────────
        contexts = []
        if not result.refused and result.sources:
            for s in result.sources:
                chunk_text = s.get("chunk_text", "").strip()
                source = s.get("source", "").strip()
                if chunk_text or source:
                    contexts.append(chunk_text if chunk_text else source)

        # ── Ensure message list exists before appending ─────────────────────
        if request.session_id not in chat_sessions:
            logger.warning(f"Session {request.session_id} disappeared during ask")
            raise HTTPException(404, "Session not found")
        
        if "messages" not in chat_sessions[request.session_id]:
            chat_sessions[request.session_id]["messages"] = []

        # ── Store message in history ────────────────────────────────────────
        message_entry = {
            "question":        request.question,
            "answer":          result.answer,
            "grounding_score": result.grounding_score,
            "refused":         result.refused,
            "trace_id":        result.trace_id,
            "timestamp":       timestamp,
            "contexts":        contexts,
        }
        
        # Add evaluation scores to history if available
        if eval_scores:
            message_entry["eval_scores"] = eval_scores
        
        # Add decision metadata to history if available
        if result.decision_metadata:
            message_entry["decision_metadata"] = result.decision_metadata
        
        chat_sessions[request.session_id]["messages"].append(message_entry)
        chat_sessions[request.session_id]["last_updated"] = timestamp
        _save_sessions(chat_sessions)

        # ── Build response with optional decision metadata ─────────────────
        decision_info = None
        if result.decision_metadata:
            decision_info = DecisionInfo(
                decision_type=result.decision_metadata.get("decision_type"),
                attempt=result.decision_metadata.get("attempt", 1),
                max_attempts=result.decision_metadata.get("max_attempts", 1),
                reason=result.decision_metadata.get("reason", ""),
                faithfulness=result.decision_metadata.get("scores", {}).get("faithfulness"),
                answer_relevancy=result.decision_metadata.get("scores", {}).get("answer_relevancy"),
                thresholds=result.decision_metadata.get("thresholds"),
            )

        return ChatResponse(
            answer=result.answer,
            is_grounded=result.is_grounded,
            grounding_score=result.grounding_score,
            refused=result.refused,
            sources_count=len(result.sources),
            sources=[SourceInfo(**s) for s in result.sources],
            trace_id=result.trace_id,
            timestamp=timestamp,
            decision_metadata=decision_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        logger.error(f"Ask error: {e}")
        
        if "401" in err or "Unauthorized" in err:
            raise HTTPException(401, "API key invalid")
        if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower():
            raise HTTPException(429, "Rate limit exceeded")
        if "timeout" in err.lower():
            raise HTTPException(504, "Request timeout")
        
        raise HTTPException(500, f"Error generating answer: {err}")


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_answer(
    request: EvaluateRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> EvaluateResponse:
    """
    Run an optional, on-demand RAGAS evaluation for a previously generated answer.

    This is intentionally NOT called from /ask by default. It only runs when explicitly
    triggered (e.g. by the user clicking "Evaluate" in the UI), since RAGAS
    faithfulness/relevancy scoring makes its own LLM calls and is too slow/
    costly to run automatically on every question.

    Looks up the original question/answer/contexts from the session's stored
    message history by trace_id (no need for the frontend to resend chunk text).
    Results are cached in eval_history.json keyed by trace_id, so re-clicking
    Evaluate on an already-evaluated answer returns the cached result instead
    of re-running RAGAS.

    Args:
        request: EvaluateRequest with session_id and trace_id

    Returns:
        EvaluateResponse with faithfulness, answer_relevancy, hallucination_rate
    """
    if request.session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    if chat_sessions[request.session_id].get("user_id") != current_user.user_id:
        raise HTTPException(403, "This session belongs to another user")

    # ── Check cache first — never re-run for an already-evaluated trace ────────
    eval_history = _load_eval_history()
    cached = next((e for e in eval_history if e.get("trace_id") == request.trace_id), None)
    if cached:
        return EvaluateResponse(
            trace_id=request.trace_id,
            faithfulness=cached["faithfulness"],
            answer_relevancy=cached["answer_relevancy"],
            hallucination_rate=cached["hallucination_rate"],
            cached=True,
        )

    # ── Locate the original message by trace_id ─────────────────────────────────
    messages = chat_sessions[request.session_id].get("messages", [])
    message = next((m for m in messages if m.get("trace_id") == request.trace_id), None)
    if message is None:
        raise HTTPException(404, "No stored answer found for this trace_id")

    if message.get("refused"):
        raise HTTPException(400, "Cannot evaluate a refused/ungrounded answer — no answer was generated")

    contexts = message.get("contexts") or []
    if not contexts:
        raise HTTPException(400, "No retrieved context available to evaluate against")

    # ── Run RAGAS (reuses existing implementation, single test case) ────────────
    test_case = {
        "question": message["question"],
        "answer":   message["answer"],
        "contexts": contexts,
    }

    try:
        result = run_ragas_evaluation([test_case])
    except Exception as e:
        logger.error(f"RAGAS evaluation failed for trace {request.trace_id}: {e}")
        raise HTTPException(500, f"Evaluation failed: {e}")

    if result.get("error") or result.get("evaluated_cases", 0) == 0:
        raise HTTPException(500, f"Evaluation failed: {result.get('error', 'no valid cases evaluated')}")

    result = _sanitize_floats(result)

    response = EvaluateResponse(
        trace_id=request.trace_id,
        faithfulness=result["faithfulness"],
        answer_relevancy=result["answer_relevancy"],
        hallucination_rate=result["hallucination_rate"],
        cached=False,
    )

    # ── Persist to eval history so future clicks hit the cache ──────────────────
    eval_history.append({
        "trace_id":           request.trace_id,
        "session_id":         request.session_id,
        "faithfulness":       response.faithfulness,
        "answer_relevancy":   response.answer_relevancy,
        "hallucination_rate": response.hallucination_rate,
        "evaluated_at":       datetime.now().isoformat(),
    })
    _save_eval_history(eval_history)

    return response


@app.get("/sessions")
def list_sessions(current_user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """List sessions belonging to the current user."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": sess.get("created_at"),
                "last_updated": sess.get("last_updated"),
                "filename": sess.get("filename"),
                "message_count": len(sess.get("messages", [])),
                "document_count": sess.get("document_count"),
                "chunk_count": sess.get("chunk_count"),
            }
            for sid, sess in chat_sessions.items()
            if sess.get("user_id") == current_user.user_id
        ]
    }


@app.get("/session/{session_id}")
def get_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get session details and full chat history."""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    if chat_sessions[session_id].get("user_id") != current_user.user_id:
        raise HTTPException(403, "This session belongs to another user")

    session = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session.get("created_at"),
        "last_updated": session.get("last_updated"),
        "filename": session.get("filename"),
        "document_count": session.get("document_count"),
        "chunk_count": session.get("chunk_count"),
        "messages": session.get("messages", []),
    }


@app.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> Dict[str, str]:
    """Delete a session and its associated vectorstore."""
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    if chat_sessions[session_id].get("user_id") != current_user.user_id:
        raise HTTPException(403, "This session belongs to another user")

    try:
        delete_vectorstore(session_id)
        del chat_sessions[session_id]
        if session_id in rag_sessions:
            del rag_sessions[session_id]
        _save_sessions(chat_sessions)
        return {"message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(500, f"Failed to delete session: {e}")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )