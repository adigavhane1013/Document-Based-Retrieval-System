import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers import LangChainTracer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from pydantic import BaseModel

load_dotenv()

# ---------- Configuration ----------


class Config:
    # LLM Settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mixtral-8x7b-instruct")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

    # RAG Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    # File Upload Limits
    MAX_FILE_SIZE_MB = 50

    # Embedding Model
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"

    # LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-mixtral-project")


# ---------- LangSmith Setup ----------

langsmith_client = None
langchain_tracer = None

if Config.LANGSMITH_API_KEY and Config.LANGCHAIN_TRACING_V2:
    try:
        langsmith_client = Client(api_key=Config.LANGSMITH_API_KEY)
        langchain_tracer = LangChainTracer(project_name=Config.LANGCHAIN_PROJECT)
        print(f"✅ LangSmith tracing enabled for project: {Config.LANGCHAIN_PROJECT}")
    except Exception as e:
        print(f"⚠️ LangSmith setup failed: {e}")
        langchain_tracer = None
else:
    print("ℹ️ LangSmith tracing disabled")


# ---------- Paths & Persistence Helpers ----------

STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_FILE = STORAGE_DIR / "sessions.json"


def load_sessions_from_disk() -> Dict:
    """Load sessions metadata from disk."""
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as e:
        print(f"Error loading sessions.json: {e}")
        return {}


def save_sessions_to_disk(sessions: Dict):
    """Save sessions metadata to disk."""
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving sessions.json: {e}")


# ---------- FastAPI App ----------

app = FastAPI(title="RAG Backend API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-Memory State ----------

chat_sessions: Dict[str, Dict] = load_sessions_from_disk()
rag_chains: Dict[str, RetrievalQA] = {}

# ---------- Prompt Template ----------

custom_prompt_template = """You are a highly precise legal/technical assistant. Your goal is to answer questions STRICTLY based on the provided context.

RULES:
1. ONLY use information from the Context section below. Do NOT use outside knowledge.
2. If the answer is not contained within the Context, explicitly state "I don't know based on the provided documents."
3. Format your answer using Markdown:
   - Use **bold** for key terms, dates, or project names.
   - Use bullet points for lists.
4. If asked to "list" something, provide ONLY the list of items found in the context (e.g., project names, skills) without additional descriptions unless explicitly asked.
5. Avoid mentioning headings or personal info from the context unless it is directly part of the answer.

Context: {context}

Question: {query}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)


def get_embeddings():
    """Get embeddings model instance."""
    return OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENROUTER_API_KEY,
        openai_api_base=Config.EMBEDDING_BASE_URL,
    )


def get_llm():
    """Get LLM instance."""
    return ChatOpenAI(
        model=Config.OPENROUTER_MODEL,
        openai_api_key=Config.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=Config.TEMPERATURE,
    )


def rebuild_rag_chain_for_session(session_id: str):
    """Rebuild RAG chain for a session from its persisted Chroma store."""
    try:
        persist_dir = f"./vectorstore/session_{session_id}"
        if not os.path.exists(persist_dir):
            print(f"No vectorstore found for session {session_id}")
            return

        embeddings = get_embeddings()
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        llm = get_llm()

        # Create callbacks list for LangSmith
        callbacks = [langchain_tracer] if langchain_tracer else []

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": Config.TOP_K_RESULTS, "score_threshold": 0.4},
            ),
            return_source_documents=True,
            callbacks=callbacks,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        rag_chains[session_id] = rag_chain
        print(f"✅ Rebuilt RAG chain for session {session_id}")
    except Exception as e:
        print(f"❌ Failed to rebuild RAG chain for {session_id}: {e}")


# Rebuild chains for all existing sessions on startup
print("\n🔄 Rebuilding RAG chains for existing sessions...")
for sid in chat_sessions.keys():
    rebuild_rag_chain_for_session(sid)
print(f"✅ Loaded {len(rag_chains)} session(s)\n")


# ---------- Models ----------


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class SourceInfo(BaseModel):
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources_count: int
    sources: List[SourceInfo]
    timestamp: str
    trace_url: Optional[str] = None


# ---------- Routes ----------


@app.get("/")
def read_root():
    return {
        "message": "RAG Backend API is running",
        "version": "1.0",
        "sessions_loaded": len(chat_sessions),
        "langsmith_enabled": langchain_tracer is not None,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """
    Upload and process PDF document.
    - If session_id is provided: Add document to existing session.
    - If session_id is None: Create new session.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

        if not documents:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="PDF appears to be empty")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(documents)

        # Add chunk_id to metadata
        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid.uuid4())

        embeddings = get_embeddings()

        if session_id and session_id in chat_sessions:
            # Add to existing session
            persist_dir = f"./vectorstore/session_{session_id}"

            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            vectorstore.add_documents(chunks)

            chat_sessions[session_id]["documents"].append(file.filename)
            chat_sessions[session_id]["documents_count"] = len(chat_sessions[session_id]["documents"])
            chat_sessions[session_id]["pages"] += len(documents)
            chat_sessions[session_id]["chunks"] += len(chunks)
            chat_sessions[session_id]["last_updated"] = datetime.now().isoformat()

            llm = get_llm()
            callbacks = [langchain_tracer] if langchain_tracer else []

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": Config.TOP_K_RESULTS, "score_threshold": 0.4},
                ),
                return_source_documents=True,
                callbacks=callbacks,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )
            rag_chains[session_id] = rag_chain

            save_sessions_to_disk(chat_sessions)
            os.unlink(tmp_path)

            return {
                "session_id": session_id,
                "filename": file.filename,
                "pages": len(documents),
                "chunks": len(chunks),
                "total_documents": chat_sessions[session_id]["documents_count"],
                "total_pages": chat_sessions[session_id]["pages"],
                "total_chunks": chat_sessions[session_id]["chunks"],
                "message": f"Added document to existing session. Total: {chat_sessions[session_id]['documents_count']} documents",
            }

        else:
            # Create new session
            new_session_id = str(uuid.uuid4())
            persist_dir = f"./vectorstore/session_{new_session_id}"

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir,
            )

            llm = get_llm()
            callbacks = [langchain_tracer] if langchain_tracer else []

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": Config.TOP_K_RESULTS, "score_threshold": 0.4},
                ),
                return_source_documents=True,
                callbacks=callbacks,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

            rag_chains[new_session_id] = rag_chain

            chat_sessions[new_session_id] = {
                "filename": file.filename,
                "documents": [file.filename],
                "documents_count": 1,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "messages": [],
                "pages": len(documents),
                "chunks": len(chunks),
            }

            save_sessions_to_disk(chat_sessions)
            os.unlink(tmp_path)

            return {
                "session_id": new_session_id,
                "filename": file.filename,
                "pages": len(documents),
                "chunks": len(chunks),
                "message": "New session created successfully",
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the document(s) in a session with LangSmith tracing."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.session_id not in rag_chains:
        if request.session_id in chat_sessions:
            rebuild_rag_chain_for_session(request.session_id)
        if request.session_id not in rag_chains:
            raise HTTPException(status_code=404, detail="Session not found or invalid")

    try:
        rag_chain = rag_chains[request.session_id]

        # Pre-generation check: Get relevant documents first
        # Extract the retriever from the chain
        retriever = rag_chain.retriever
        source_docs = retriever.invoke(request.question)

        if not source_docs:
            timestamp = datetime.now().isoformat()
            fallback_answer = "I don't know based on the provided documents."

            # Store message
            chat_sessions[request.session_id]["messages"].append(
                {
                    "question": request.question,
                    "answer": fallback_answer,
                    "sources": 0,
                    "timestamp": timestamp,
                }
            )
            chat_sessions[request.session_id]["last_updated"] = timestamp
            save_sessions_to_disk(chat_sessions)

            return ChatResponse(
                answer=fallback_answer,
                sources_count=0,
                sources=[],
                timestamp=timestamp,
                trace_url=None,
            )

        # Invoke with LangChain tracing
        callbacks = [langchain_tracer] if langchain_tracer else []
        # Since we use RetrievalQA, it expects 'query' in input for RetrievalQA
        # but the prompt template now uses 'query' instead of 'question'
        response = rag_chain.invoke({"query": request.question}, config={"callbacks": callbacks})

        answer = response["result"]
        source_docs = response["source_documents"]

        # Extract source info
        sources_info = []
        for doc in source_docs:
            page = doc.metadata.get("page")
            chunk_id = doc.metadata.get("chunk_id")
            sources_info.append(SourceInfo(page=page if page is not None else None, chunk_id=chunk_id))

        timestamp = datetime.now().isoformat()

        # Store message
        chat_sessions[request.session_id]["messages"].append(
            {
                "question": request.question,
                "answer": answer,
                "sources": len(source_docs),
                "timestamp": timestamp,
            }
        )
        chat_sessions[request.session_id]["last_updated"] = timestamp

        save_sessions_to_disk(chat_sessions)

        # Get trace URL if LangSmith is enabled
        trace_url = None
        if langsmith_client:
            trace_url = f"https://smith.langchain.com/o/default/projects/p/{Config.LANGCHAIN_PROJECT}"

        return ChatResponse(
            answer=answer,
            sources_count=len(source_docs),
            sources=sources_info,
            timestamp=timestamp,
            trace_url=trace_url,
        )

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(status_code=401, detail="OpenRouter API key invalid or unauthorized")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later")
        else:
            raise HTTPException(status_code=500, detail=f"Error generating answer: {error_msg}")


@app.get("/sessions")
def get_sessions():
    """Get all chat sessions with metadata."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename": data.get("filename"),
                "documents": data.get("documents", [data.get("filename")]),
                "documents_count": data.get("documents_count", 1),
                "created_at": data.get("created_at"),
                "last_updated": data.get("last_updated", data.get("created_at")),
                "message_count": len(data.get("messages", [])),
                "pages": data.get("pages", 0),
                "chunks": data.get("chunks", 0),
            }
            for sid, data in chat_sessions.items()
        ]
    }


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get a specific session with all messages and metadata."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_sessions[session_id]


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and its vector store."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        del chat_sessions[session_id]
        if session_id in rag_chains:
            del rag_chains[session_id]

        shutil.rmtree(f"./vectorstore/session_{session_id}", ignore_errors=True)
        save_sessions_to_disk(chat_sessions)

        return {"message": "Session deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
