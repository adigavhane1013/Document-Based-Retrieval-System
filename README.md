# DocMind — Retrieval-Augmented Generation (RAG) System

Upload PDF, DOCX, TXT, or Markdown documents and ask grounded questions about them. Answers are restricted to retrieved document content, returned with source citations and a grounding score. Answer quality can optionally be checked on demand via RAGAS metrics (Faithfulness, Answer Relevancy, Hallucination Rate). Each user's documents and chat history are private to their account.

Built with **FastAPI**, **LangChain**, **Qdrant Cloud**, **HuggingFace Embeddings (BAAI/bge-small-en-v1.5)**, **Groq (llama-3.3-70b-versatile)**, and **Firebase Authentication**.

---

## Features

- **Authentication** — email/password sign-up and login via Firebase; every session, upload, and answer is scoped to the logged-in user (403 if you try to access another user's session)
- **Multi-format upload** — PDF, TXT, MD, DOCX
- **Column-aware PDF extraction** — detects multi-column layouts (e.g. resumes/CVs) and reads each column top-to-bottom instead of merging unrelated text across columns
- **Context-aware Q&A** — answers grounded strictly in retrieved chunks
- **Hybrid retrieval** — dense (Qdrant) + sparse (BM25), merged and cross-encoder reranked
- **Query rewriting** — detects ambiguous queries and rewrites them via LLM before retrieval
- **Session isolation** — each uploaded document gets its own vector collection, scoped to its owner
- **Source citations** — every answer cites the chunk IDs it used
- **Grounding score** — every answer includes a grounding score from the hallucination filter
- **On-demand RAGAS evaluation** — Faithfulness, Answer Relevancy, and Hallucination Rate, triggered manually per answer (not run automatically on every query)
- **Evaluation caching** — repeated evaluation requests for the same answer return the cached result instead of re-running RAGAS
- **Decision layer** — when evaluation scores are supplied, classifies an answer as ACCEPT, RETRY, FALLBACK, or REJECT based on faithfulness/relevance thresholds
  - ACCEPT and REJECT are fully wired
  - RETRY and FALLBACK are currently logged but not yet acted on (no automatic re-retrieval or model-switching loop implemented yet)
  - Decision-layer errors are caught — a failure here never crashes the pipeline; the answer is still returned
- **Structured logging** — JSON logs written to `logs/rag.log`
- **Web UI** — browser-based interface via `docmind_ui.html`, including login/signup, show/hide password, and clear error messages for invalid credentials

---

## Architecture

```
Login (Firebase email/password)
        │
        ▼
Upload (PDF/TXT/MD/DOCX)
        │
        ▼
Loader (pdfplumber, column-aware for PDFs)
        │
        ▼
Chunking (chunk_size=1024, overlap=256)
        │
        ▼
HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
        │
        ▼
Qdrant Cloud (collection per session)
        │
        ▼
Question asked (Firebase ID token verified on every request)
        │
        ▼
Query Rewriter (ambiguity check → LLM rewrite if needed)
        │
        ▼
Hybrid Retrieval (Dense + BM25)
        │
        ▼
Cross-Encoder Reranker
        │
        ▼
Groq llama-3.3-70b-versatile → grounded answer + citations
        │
        ▼
Hallucination filter → grounding score
        │
        ▼
(Optional, on request) RAGAS evaluation → Decision layer
        │
        ▼
Answer + sources + grounding score (+ evaluation, if requested) returned
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | HTML/CSS/JS (`docmind_ui.html`) |
| Auth | Firebase Authentication (email/password) |
| Backend | FastAPI, Uvicorn |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `BAAI/bge-small-en-v1.5` |
| Orchestration | LangChain |
| Vector Store | Qdrant Cloud (collection per session) |
| Sparse Retrieval | BM25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy, Hallucination Rate) |
| Observability | Custom structured JSON logger |

---

## Folder Structure

```
rag_production/
│
├── configs/
│   └── settings.py                 # Central settings (thresholds, models, paths, Qdrant/Groq/Firebase config)
│
├── embeddings/
│   └── embedding_model.py          # HuggingFace BAAI/bge-small-en-v1.5 wrapper
│
├── evaluation/
│   └── ragas_eval.py               # RAGAS faithfulness + relevancy scoring, with retry/backoff
│
├── guardrails/
│   └── hallucination_filter.py     # Grounding / citation verification
│
├── ingestion/
│   ├── chunking.py                 # Document chunking
│   └── loader.py                   # PDF/TXT/MD/DOCX loader (column-aware PDF extraction)
│
├── logs/
│   └── rag.log
│
├── observability/
│   └── logger.py                   # Structured JSON logging
│
├── rag/
│   ├── decision_layer.py           # ACCEPT/RETRY/FALLBACK/REJECT logic
│   ├── pipeline.py                 # Main RAG orchestration
│   ├── prompt.py                   # LLM prompt templates
│   └── query_rewriter.py           # Ambiguity detection + LLM rewriting
│
├── retrieval/
│   ├── reranker.py                 # Cross-encoder reranking
│   └── retriever.py                # Hybrid dense (Qdrant) + BM25 retrieval
│
├── storage/                        # Session metadata, eval history
│
├── tests/
│   ├── test_decision_layer.py
│   ├── test_query_rewriter.py
│   ├── test_fixed_modules.py
│   └── test_integration_fixes.py
│
├── vectorstore/
│   └── vectordb.py                 # Qdrant Cloud init + wrapper
│
├── auth.py                         # Firebase ID token verification (FastAPI dependency)
├── firebase-service-account.json   # Firebase Admin SDK credentials (not committed)
├── .env
├── docmind_ui.html
├── main.py                         # FastAPI app + REST endpoints
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier works)
- A [Firebase](https://console.firebase.google.com) project with Email/Password sign-in enabled

### Setup

```bash
git clone https://github.com/adigavhane1013/Document-Based-Retrieval-System.git
cd rag_production

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) download automatically on first run.

### Firebase setup

1. Firebase Console → Authentication → enable **Email/Password** sign-in
2. Project Settings → General → Add app → Web (`</>`) → copy the `firebaseConfig` into the `<script type="module">` block in `docmind_ui.html`
3. Project Settings → Service Accounts → Generate new private key → save the downloaded file as `firebase-service-account.json` in the project root

---

## Configuration

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

Other settings (chunk size, retrieval top-k, RAGAS thresholds, etc.) have defaults in `configs/settings.py` and can be overridden there or via environment variables.

---

## Running

```bash
python main.py
```

- API: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- UI: open `docmind_ui.html` directly in your browser, sign up / log in, then upload documents

---

## API Endpoints

All endpoints below (except `/health`) require a Firebase ID token: `Authorization: Bearer <token>`. `docmind_ui.html` attaches this automatically once you're logged in.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (no auth) |
| `POST` | `/upload` | Upload a PDF/TXT/MD/DOCX document |
| `POST` | `/ask` | Ask a question against a session's documents (`run_evaluation: true` to also run RAGAS + decision layer) |
| `POST` | `/evaluate` | Run on-demand RAGAS evaluation on a previously generated answer, by `trace_id` (cached on repeat) |
| `GET` | `/sessions` | List sessions belonging to the current user |
| `GET` | `/session/{id}` | Get session details + chat history (owner only) |
| `DELETE` | `/session/{id}` | Delete a session and its vector store (owner only) |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <firebase-id-token>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "question": "What are the key findings?"}'
```

### Example: Evaluate an answer

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Authorization: Bearer <firebase-id-token>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "trace_id": "trace-id-from-ask-response"}'
```

---

## Evaluation

Evaluation is on-demand only — triggered via `POST /evaluate` (or `POST /ask` with `run_evaluation: true`) against a specific answer, identified by `trace_id`. It is not run automatically on every query, since RAGAS scoring makes its own LLM calls and adds several seconds of latency.

| Metric | Description |
|---|---|
| Faithfulness | Whether the answer's claims are supported by the retrieved context |
| Answer Relevancy | Whether the answer addresses the question asked |
| Hallucination Rate | `1 - Faithfulness` |

Results are cached by `trace_id`, so re-evaluating the same answer returns the cached score instead of re-running RAGAS. JSON parsing failures from the judge LLM (common with Llama-via-Groq) are retried with exponential backoff before falling back to a partial (per-metric) result.

---

## Testing

```bash
.venv\Scripts\activate
python -m pytest tests/ -v
```

Test suite covers query rewriting, decision layer logic, RAGAS error handling/backoff, and pipeline integration (including backwards compatibility).

---

## Logging

Structured JSON logs are written to `logs/rag.log`, covering ingestion, retrieval, reranking, generation, and evaluation stages.

---

## Environment & Security

Excluded from the repository via `.gitignore`:

- `.env` — API keys (Groq, Qdrant)
- `firebase-service-account.json` — Firebase Admin SDK credentials
- `.venv/`
- `storage/users.db` (legacy, unused — see Known Limitations)
- `logs/`

Never commit `.env` or `firebase-service-account.json`.

---

## Known Limitations

- RETRY and FALLBACK decision-layer outcomes are logged but not yet acted upon — no automatic re-retrieval or model-switching loop yet
- `FALLBACK_LLM_MODEL` is not configured by default; fallback decisions currently fall through to ACCEPT
- Evaluation is single-turn only (no multi-turn conversation evaluation)
- Not yet deployed — designed to run with the FastAPI backend on a persistent host (Render/Railway/Fly.io) and `docmind_ui.html` on Vercel; not suited to serverless deployment as-is due to in-memory ML model loading (sentence-transformers reranker, BM25 index) at startup

---

## License

MIT License.