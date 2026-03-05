# 🤖 RAG Chatbot — Document-Based Retrieval System

A full-stack AI chatbot that lets you upload PDF documents and ask questions about them. Built with **FastAPI**, **React**, **LangChain**, **ChromaDB**, and powered by **Mixtral-8x7B** via OpenRouter.

![CI](https://github.com/adigavhane1013/Document-Based-Retrieval-System/actions/workflows/ci.yml/badge.svg)

---
---

## ✨ Features

- 📄 **PDF Upload** — Upload one or multiple PDFs per chat session
- 💬 **Context-Aware Q&A** — Answers strictly grounded in your uploaded documents
- 🗂️ **Multi-Session Support** — Each chat is an isolated session with its own vector store
- 💾 **Session Persistence** — Sessions and chat history survive server restarts
- 🔍 **Source Attribution** — Every answer includes the page numbers it was drawn from
- 📊 **LangSmith Tracing** — Optional observability for every LLM call
- 🧠 **Configurable RAG** — Chunk size, overlap, top-k, temperature all via `.env`

---

## 🏗️ Project Structure

```
Document-Based-Retrieval-System/
│
├── backend/
│   ├── app.py                  # FastAPI app — all endpoints & RAG logic
│   ├── storage/
│   │   └── sessions.json       # Persisted session metadata (auto-generated)
│   └── vectorstore/            # Per-session Chroma vector stores (auto-generated)
│
├── frontend/
│   ├── src/                    # React source code
│   └── package.json
│
├── src/                        # Shared utilities / helpers
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, Axios |
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Mixtral-8x7B via OpenRouter |
| **Embeddings** | `text-embedding-3-small` via OpenRouter |
| **Vector Store** | ChromaDB (persisted per session) |
| **RAG Framework** | LangChain (`RetrievalQA`) |
| **PDF Parsing** | PyPDFLoader |
| **Observability** | LangSmith (optional) |
| **CI** | GitHub Actions |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An [OpenRouter](https://openrouter.ai) API key (free tier available)
- Optional: A [LangSmith](https://smith.langchain.com) API key for tracing

---

### 1. Clone the repo

```bash
git clone https://github.com/adigavhane1013/Document-Based-Retrieval-System.git
cd Document-Based-Retrieval-System
```

### 2. Set up the backend

```bash
# Create and activate virtual environment
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt
```

### 3. Configure environment variables

Create a `.env` file inside the `backend/` folder:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mixtral-8x7b-instruct

# RAG Settings (optional — these are the defaults)
TEMPERATURE=0.2
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
TOP_K_RESULTS=5

# LangSmith Tracing (optional)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-mixtral-project
```

### 4. Start the backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Backend will be live at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 5. Set up and start the frontend

```bash
cd frontend
npm install
npm start
```

Frontend will be live at: `http://localhost:3000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns API status |
| `POST` | `/upload` | Upload a PDF and create/extend a session |
| `POST` | `/ask` | Ask a question in a session |
| `GET` | `/sessions` | List all sessions with metadata |
| `GET` | `/session/{id}` | Get a specific session + full chat history |
| `DELETE` | `/session/{id}` | Delete a session and its vector store |

### Example: Upload a PDF

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
User uploads PDF
      │
      ▼
PyPDFLoader parses pages
      │
      ▼
RecursiveCharacterTextSplitter chunks text
(chunk_size=1024, overlap=100)
      │
      ▼
OpenAI Embeddings (text-embedding-3-small via OpenRouter)
      │
      ▼
ChromaDB stores vectors (persisted to ./vectorstore/session_{id}/)
      │
      ▼
User asks a question
      │
      ▼
Similarity search (top-5, score threshold 0.4)
      │
      ▼
RetrievalQA chain → Mixtral-8x7B generates answer
      │
      ▼
Answer + source pages returned to frontend
```

---

## 🔄 CI Pipeline

Every push and pull request automatically runs:

| Job | What it checks |
|---|---|
| 🔍 Backend Lint | Black, isort, Flake8 on `backend/` |
| 🧪 Backend Tests | pytest with coverage on `tests/backend/` |
| 🔒 Security Scan | Bandit (code) + Safety (dependencies) |
| 🔍 Frontend Lint | ESLint on `frontend/src/` |
| 🏗️ Frontend Build | `npm run build` — catches build errors |
| 🧪 Frontend Tests | Jest with coverage |
| 🐳 Docker Build | Verifies multi-stage image builds cleanly |

---

## 🔒 Environment & Security

The following are excluded from the repository via `.gitignore`:

- `.env` — API keys
- `backend/venv/` — virtual environment
- `backend/vectorstore/` — generated vector stores
- `backend/storage/sessions.json` — runtime session data

Never commit your `.env` file. Use [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request against `main`

Please run `black backend/` and `isort backend/` before submitting a PR.

---

## 📄 License

This project is licensed under the MIT License.
