# 🤖 DocMind — Production-Grade Retrieval-Augmented Generation (RAG) System

A production-grade RAG system that lets you upload documents and ask grounded questions about them. Built with **FastAPI**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings (BAAI/bge-small-en-v1.5)**, **Groq (Llama-3.3-70b-versatile)**, **Hybrid Dense + BM25 Retrieval**, **Cross-Encoder Reranking**, **Grounding Verification**, and **On-Demand RAGAS Evaluation**.

![CI](https://github.com/adigavhane1013/Document-Based-Retrieval-System/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E)
![Groq](https://img.shields.io/badge/Groq-F55036)

---

## 📖 Project Overview

DocMind lets you upload PDF and DOCX documents and ask natural-language questions about their content. Answers are grounded strictly in the retrieved document chunks, returned alongside the source chunks used and a grounding score. Answer quality can optionally be evaluated on demand using RAGAS metrics (Faithfulness, Answer Relevancy, Hallucination Rate).

---

## ✨ Features

- 📄 **Multi-Format Upload** — Upload PDF and DOCX documents
- 💬 **Context-Aware Q&A** — Answers strictly grounded in your uploaded documents
- 🔍 **Hybrid Retrieval** — Dense (ChromaDB) + Sparse (BM25) retrieval with cross-encoder reranking
- 🧠 **Query Rewriting** — Automatic ambiguity detection and LLM-based query optimization
- 🗂️ **Session Isolation** — Each uploaded document gets its own isolated vector store
- 📌 **Source Citations** — Answers reference the retrieved chunks they came from
- 📈 **Grounding Score** — Each answer is returned with a grounding score
- 🧪 **Optional RAGAS Evaluation** — On-demand Faithfulness, Answer Relevancy, and Hallucination Rate scoring
- ⚡ **Evaluation Caching** — Avoids redundant evaluation calls for repeated queries
- 🪵 **Structured Logging** — Custom structured logger for observability
- 🧩 **Decision Layer** — Automatically determines whether to Accept, Retry, Fallback, or Reject an answer based on quality thresholds
- 🖥️ **Web UI** — Browser-based interface via `docmind_ui.html`

---

## 🌟 Key Highlights

- Hybrid Dense + Sparse Retrieval
- Cross-Encoder Reranking
- Automatic Query Rewriting
- Session-Isolated Vector Stores
- Grounding Verification
- Optional RAGAS Evaluation
- Evaluation Caching
- 62 Unit Tests

---

## 🏗️ Architecture

```
User uploads PDF / DOCX
        │
        ▼
Loader parses document (pdfplumber / python-docx)
        │
        ▼
Chunking (chunk_size=1024, overlap=256)
        │
        ▼
HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
        │
        ▼
ChromaDB stores vectors (persisted per session)
        │
        ▼
User asks a question
        │
        ▼
Query Rewriter — detects ambiguity, rewrites query via LLM
        │
        ▼
Hybrid Retrieval — Dense (ChromaDB) + Sparse (BM25) merged
        │
        ▼
Cross-Encoder Reranker — rescores top candidates
        │
        ▼
Groq llama-3.3-70b generates grounded answer + citations
        │
        ▼
Grounding Analysis
        │
        ▼
Grounding Score
        │
        ▼
(Optional) RAGAS Evaluation
  ├── Faithfulness
  ├── Answer Relevancy
  └── Hallucination Rate
        │
        ▼
Answer + sources + grounding score returned
```

---

## 📸 Screenshots

> _Add screenshots of the UI here._

| Upload | Q&A | Evaluation |
|---|---|---|
| _placeholder_ | _placeholder_ | _placeholder_ |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | HTML/CSS/JS (`docmind_ui.html`) |
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | HuggingFace — `BAAI/bge-small-en-v1.5` |
| **Orchestration** | LangChain |
| **Vector Store** | ChromaDB (persisted per session) |
| **Sparse Retrieval** | BM25 |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevancy, Hallucination Rate) |
| **Observability** | Custom structured logger |
| **CI** | GitHub Actions |

---

## 🏗️ Folder Structure

```
rag_production/
│
├── configs/
│   ├── __init__.py
│   └── settings.py                      # Central settings (thresholds, models, paths)
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py               # HuggingFace BAAI/bge-small-en-v1.5 wrapper
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
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key (free tier available)

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

> HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) are downloaded automatically on first run — no separate pull step is required.

---

## ⚙️ Configuration

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

---

## ▶️ Running the Project

### Start the backend

```bash
uvicorn main:app --reload --port 8000
```

Backend will be live at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Open the UI

Open `docmind_ui.html` directly in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF/DOCX document |
| `POST` | `/ask` | Ask a question against uploaded documents |
| `POST` | `/evaluate` | Run on-demand RAGAS evaluation on a previously generated answer (by `trace_id`) |
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

### Example: Evaluate an answer

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "trace_id": "trace-id-from-ask-response"}'
```

---

## 🔄 End-to-End Workflow

```
Upload
  ↓
Chunking
  ↓
HuggingFace Embeddings
  ↓
ChromaDB
  ↓
Query Rewriter
  ↓
Hybrid Retrieval
  ↓
Cross-Encoder Reranker
  ↓
LLM
  ↓
Grounding Analysis
  ↓
Grounding Score
  ↓
(Optional Evaluation)
  ↓
Faithfulness · Answer Relevancy · Hallucination Rate
```

---

## 🧪 Evaluation Pipeline

Evaluation is **on-demand**, triggered via `POST /evaluate` against a previously generated answer (identified by `session_id` and `trace_id`), and is not run automatically on every query.

| Metric | Description |
|---|---|
| **Faithfulness** | Measures whether the answer is grounded in the retrieved context |
| **Answer Relevancy** | Measures whether the answer addresses the question asked |
| **Hallucination Rate** | Measures the proportion of unsupported claims in the answer |

Evaluation results are cached to avoid redundant scoring of repeated question/answer pairs.

---

## 🧪 Testing

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

### CI Pipeline

Every push and pull request automatically runs:

| Job | What it checks |
|---|---|
| 🔍 Backend Lint | Black, isort, Flake8 |
| 🧪 Backend Tests | pytest — 62 tests with coverage |
| 🔒 Security Scan | Bandit (code) + Safety (dependencies) |

---

## 🪵 Logging

Structured logging is handled by `observability/logger.py` and written to `logs/rag.log`. Logs capture pipeline stage events (chunking, retrieval, reranking, generation, evaluation) for debugging and observability.

---

## 🔒 Environment & Security

The following are excluded from the repository via `.gitignore`:

- `.env` — API keys
- `.venv/` — virtual environment
- `vectorstore/session_*` — generated vector stores
- `logs/` — runtime logs

Never commit your `.env` file. Use [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI.

---

## 🛠️ Future Improvements

- Authentication for multi-user usage
- Streaming responses in the UI
- Additional document format support
- Benchmark dataset for retrieval/answer quality
- Evaluation dashboard for RAGAS metrics
- Multi-document retrieval (cross-session querying)
- Production deployment (Docker, cloud hosting)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request against `main`

Please run `black .` and `isort .` before submitting a PR.

---

## 📄 License

This project is licensed under the MIT License.