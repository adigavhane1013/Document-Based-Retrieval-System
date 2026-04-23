# 🤖 DocMind — Production-Grade Document Retrieval System

A production-grade RAG system that lets you upload documents and ask questions about them. Built with **FastAPI**, **ChromaDB**, **Groq (Llama-3.3-70b)**, and **Ollama local embeddings** with hybrid dense + BM25 retrieval, cross-encoder reranking, and RAGAS-based quality guardrails.

![CI](https://github.com/adigavhane1013/Document-Based-Retrieval-System/actions/workflows/ci.yml/badge.svg)

---

## ✨ Features

- 📄 **Multi-Format Upload** — Upload PDF and DOCX documents
- 💬 **Context-Aware Q&A** — Answers strictly grounded in your uploaded documents
- 🔍 **Hybrid Retrieval** — Dense (ChromaDB) + Sparse (BM25) retrieval with cross-encoder reranking
- 🧠 **Query Rewriting** — Automatic ambiguity detection and LLM-based query optimization
- ✅ **Quality Guardrails** — RAGAS Decision Layer with Accept / Retry / Fallback / Reject logic
- 📊 **RAGAS Evaluation** — Automated Faithfulness and Answer Relevancy scoring per query
- 🔒 **Hallucination Filtering** — Guardrails to detect and filter hallucinated content
- 🗂️ **Multi-Session Support** — Each session has its own isolated vector store
- 🖥️ **Web UI** — Browser-based interface via `docmind_ui.html`

---

## 🏗️ Project Structure

```
rag_production/
│
├── configs/
│   ├── __init__.py
│   └── settings.py                      # Central settings (thresholds, models, paths)
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py               # Ollama nomic-embed-text wrapper
│
├── evaluation/
│   ├── __init__.py
│   ├── cli_eval.py                      # CLI tool for running evaluations
│   ├── deepeval_tests.py                # DeepEval integration tests
│   └── ragas_eval.py                    # RAGAS faithfulness + relevancy scoring
│
├── guardrails/
│   ├── __init__.py
│   └── hallucination_filter.py          # Detect and filter hallucinated content
│
├── ingestion/
│   ├── __init__.py
│   ├── chunking.py                      # Document chunking (chunk_size=1024, overlap=256)
│   └── loader.py                        # PDF/DOCX document loader
│
├── logs/
│   └── rag.log                          # RAG system log file
│
├── observability/
│   ├── __init__.py
│   └── logger.py                        # Custom structured logging setup
│
├── rag/
│   ├── __init__.py
│   ├── decision_layer.py                # RAGAS-based Accept/Retry/Fallback/Reject logic
│   ├── pipeline.py                      # Main RAG orchestration pipeline
│   ├── prompt.py                        # LLM prompt templates
│   └── query_rewriter.py                # Query ambiguity detection + LLM rewriting
│
├── retrieval/
│   ├── __init__.py
│   ├── reranker.py                      # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   └── retriever.py                     # Dense (ChromaDB) + Sparse (BM25) hybrid retrieval
│
├── storage/
│   └── (ChromaDB database files)
│
├── tests/
│   ├── __init__.py
│   ├── test_decision_layer.py           # 28 tests — Decision layer ✅
│   └── test_query_rewriter.py           # 34 tests — Query rewriting ✅
│
├── vectorstore/
│   ├── session_*.db                     # ChromaDB session persistence files
│   └── vectordb.py                      # ChromaDB initialization + wrapper
│
├── .env
├── docmind_ui.html                      # Web UI interface
├── main.py                              # FastAPI application + REST endpoints
├── ollama_models.txt                    # Available Ollama models reference
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | HTML/CSS/JS (`docmind_ui.html`) |
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | Ollama — `nomic-embed-text` (local) |
| **Vector Store** | ChromaDB (persisted per session) |
| **Sparse Retrieval** | BM25 |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Evaluation** | RAGAS (Faithfulness + Answer Relevancy) |
| **Observability** | Custom structured logger |
| **CI** | GitHub Actions |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled
- A [Groq](https://console.groq.com) API key (free tier available)

---

### 1. Clone the repo

```bash
git clone https://github.com/adigavhane1013/Document-Based-Retrieval-System.git
cd rag_production
```

### 2. Set up the environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 4. Configure environment variables

Create a `.env` file at the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# RAG Settings (optional — these are the defaults)
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
TOP_K_DENSE=20
TOP_K_SPARSE=20
TOP_K_RERANK=5
RETRIEVAL_SCORE_THRESHOLD=0.45
HYBRID_ALPHA=0.7

# RAGAS Decision Layer
RAGAS_FAITHFULNESS_THRESHOLD=0.70
RAGAS_RELEVANCE_THRESHOLD=0.65
RAGAS_MAX_RETRY_ATTEMPTS=2
RAGAS_FALLBACK_ENABLED=true
```

### 5. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

Backend will be live at: `http://localhost:8000`  
API docs at: `http://localhost:8000/docs`

### 6. Open the UI

Open `docmind_ui.html` directly in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF/DOCX document |
| `POST` | `/ask` | Ask a question against uploaded documents |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/session/{id}` | Get session + chat history |
| `DELETE` | `/session/{id}` | Delete a session and its vector store |

### Example: Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "question": "What are the key findings?"}'
```

---

## ⚙️ How It Works

```
User uploads PDF / DOCX
        │
        ▼
Loader parses document (pdfplumber / python-docx)
        │
        ▼
RecursiveCharacterTextSplitter chunks text
(chunk_size=1024, overlap=256)
        │
        ▼
Ollama nomic-embed-text generates embeddings (local)
        │
        ▼
ChromaDB stores vectors (persisted to ./vectorstore/session_{id}/)
        │
        ▼
User asks a question
        │
        ▼
Query Rewriting Layer — detects ambiguity, rewrites query via LLM
        │
        ▼
Hybrid Retrieval — Dense (ChromaDB) + Sparse (BM25) merged (alpha=0.7)
        │
        ▼
Cross-Encoder Reranker — rescores top candidates (threshold=0.45, top-k=5)
        │
        ▼
Groq llama-3.3-70b generates answer from retrieved context
        │
        ▼
RAGAS Decision Layer evaluates answer quality
  ├── ACCEPT  → return answer (faithfulness ≥ 0.70, relevancy ≥ 0.65)
  ├── RETRY   → fetch more context and regenerate
  ├── FALLBACK → use better model if available
  └── REJECT  → max retries exceeded, return graceful degradation
        │
        ▼
Answer + decision metadata returned
```

---

## 🧪 Running Tests

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Clear cache
Remove-Item -r rag/__pycache__ -ErrorAction SilentlyContinue
Remove-Item -r .pytest_cache -ErrorAction SilentlyContinue

# Run all tests
python -m pytest tests/ -v

# Run individually
python -m pytest tests/test_query_rewriter.py -v   # 34 tests
python -m pytest tests/test_decision_layer.py -v   # 28 tests
```

**Current Status: 62/62 tests passing ✅**

---

## 🔄 CI Pipeline

Every push and pull request automatically runs:

| Job | What it checks |
|---|---|
| 🔍 Backend Lint | Black, isort, Flake8 |
| 🧪 Backend Tests | pytest — 62 tests with coverage |
| 🔒 Security Scan | Bandit (code) + Safety (dependencies) |

---

## 🔒 Environment & Security

The following are excluded from the repository via `.gitignore`:

- `.env` — API keys
- `.venv/` — virtual environment
- `vectorstore/session_*` — generated vector stores
- `logs/` — runtime logs

Never commit your `.env` file. Use [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request against `main`

Please run `black .` and `isort .` before submitting a PR.

---
