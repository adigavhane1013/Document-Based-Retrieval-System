# DocMind — Retrieval-Augmented Generation (RAG) System

Upload PDF, DOCX, TXT, or Markdown documents and ask grounded questions about them. Answers are restricted to retrieved document content, returned with source citations and a grounding score. Answer quality can optionally be checked on demand via RAGAS metrics (Faithfulness, Answer Relevancy, Hallucination Rate). Each user's documents and chat history are private to their account.

Built with **FastAPI**, **LangChain**, **Qdrant Cloud**, **HuggingFace Embeddings (BAAI/bge-small-en-v1.5)**, **Groq (llama-3.3-70b-versatile)**, and **Firebase Authentication**.

**Live:**
- Frontend: https://document-based-retrieval-system.vercel.app/
- Backend API: https://document-based-retrieval-system-production.up.railway.app

---

## Features

- **Authentication** — email/password sign-up and login via Firebase; every session, upload, and answer is scoped to the logged-in user (403 if you try to access another user's session)
- **Multi-format upload** — PDF, TXT, MD, DOCX
- **Multi-document sessions** — add additional files to an already-open session via the attach (`+`) button in the chat composer; they extend the same knowledge base instead of replacing it
- **Column-aware PDF extraction** — detects multi-column layouts (e.g. resumes/CVs) and reads each column top-to-bottom instead of merging unrelated text across columns
- **Context-aware Q&A** — answers grounded strictly in retrieved chunks
- **Hybrid retrieval** — dense (Qdrant) + sparse (BM25), merged and cross-encoder reranked
- **Query rewriting** — detects ambiguous queries and rewrites them via LLM before retrieval
- **Session isolation** — each uploaded document set gets its own vector collection, scoped to its owner
- **Source citations** — every answer cites the chunk IDs it used
- **Grounding score** — every answer includes a grounding score from the hallucination filter
- **On-demand RAGAS evaluation** — Faithfulness, Answer Relevancy, and Hallucination Rate, triggered manually per answer (not run automatically on every query)
- **Evaluation caching** — repeated evaluation requests for the same answer return the cached result instead of re-running RAGAS
- **Decision layer** — when evaluation scores are supplied, classifies an answer as ACCEPT, RETRY, FALLBACK, or REJECT based on faithfulness/relevance thresholds
  - ACCEPT and REJECT are fully wired
  - RETRY and FALLBACK are currently logged but not yet acted on (no automatic re-retrieval or model-switching loop implemented yet)
  - Decision-layer errors are caught — a failure here never crashes the pipeline; the answer is still returned
- **Structured logging** — JSON logs written to `logs/rag.log`
- **Web UI** — browser-based interface (`index.html`), including login/signup, show/hide password, and clear error messages for invalid credentials

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
openai/gpt-oss-120b → grounded answer + citations
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
| UI | HTML/CSS/JS (`index.html`) — deployed on Vercel |
| Backend | FastAPI, Uvicorn — deployed on Railway |
| Auth | Firebase Authentication (email/password) |
| LLM | Groq — `openai/gpt-oss-120b` |
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
│   └── settings.py                 # Central settings (thresholds, models, paths, Qdrant/Groq config)
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
├── storage/                        # Session metadata, eval history (created at runtime)
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
├── .env                             # local secrets (not committed)
├── index.html                      # Web UI — entry point
├── main.py                         # FastAPI app + REST endpoints
└── requirements.txt
```

`firebase-service-account.json` is required locally (project root) but never committed — see Deployment below for how it's supplied in production.

---

## Local Setup

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier works)
- A [Firebase](https://console.firebase.google.com) project with Email/Password sign-in enabled

### Install

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
2. Project Settings → General → Add app → Web (`</>`) → copy the `firebaseConfig` into the `<script type="module">` block in `index.html`
3. Project Settings → Service Accounts → Generate new private key → save the downloaded file as `firebase-service-account.json` in the project root (local dev only)

### Configuration

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

Other settings (chunk size, retrieval top-k, RAGAS thresholds, etc.) have defaults in `configs/settings.py` and can be overridden there or via environment variables.

### Run locally

```bash
python main.py
```

- API: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- UI: serve `index.html` over local HTTP (e.g. `python -m http.server 5500`) and open it — Firebase's SDK requires `http://`, not `file://`

---

## Deployment

- **Backend** runs on **Railway** (persistent container — needed because ML models, i.e. the sentence-transformers reranker and BM25 index, load into memory at startup and stay warm; a serverless platform would reload them on every cold start).
  - Port is read from the `PORT` environment variable Railway injects.
  - Firebase credentials are supplied via a `FIREBASE_SERVICE_ACCOUNT_JSON` environment variable (the full service-account JSON as one value) instead of a committed file.
  - Required Railway env vars: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_JSON`.
- **Frontend** (`index.html`) is deployed as a static site on **Vercel**, pointed at the Railway backend URL via the `API` constant in the file.

---

## API Endpoints

All endpoints below (except `/health`) require a Firebase ID token: `Authorization: Bearer <token>`. `index.html` attaches this automatically once you're logged in.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (no auth) |
| `POST` | `/upload` | Upload a PDF/TXT/MD/DOCX document. Pass `session_id` to add to an existing session instead of creating a new one |
| `POST` | `/ask` | Ask a question against a session's documents (`run_evaluation: true` to also run RAGAS + decision layer) |
| `POST` | `/evaluate` | Run on-demand RAGAS evaluation on a previously generated answer, by `trace_id` (cached on repeat) |
| `GET` | `/sessions` | List sessions belonging to the current user |
| `GET` | `/session/{id}` | Get session details + chat history (owner only) |
| `DELETE` | `/session/{id}` | Delete a session and its vector store (owner only) |

### Example: Ask a question

```bash
curl -X POST https://document-based-retrieval-system-production.up.railway.app/ask \
  -H "Authorization: Bearer <firebase-id-token>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "question": "What are the key findings?"}'
```

### Example: Evaluate an answer

```bash
curl -X POST https://document-based-retrieval-system-production.up.railway.app/evaluate \
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

Structured JSON logs are written to `logs/rag.log` locally, and to stdout (captured by Railway) in production — covering ingestion, retrieval, reranking, generation, and evaluation stages.

---

## Environment & Security

Excluded from the repository via `.gitignore`:

- `.env` — API keys (Groq, Qdrant)
- `firebase-service-account.json` — Firebase Admin SDK credentials (local dev only; production uses the `FIREBASE_SERVICE_ACCOUNT_JSON` env var)
- `.venv/`
- `storage/`, `logs/` — runtime-generated, not source

Never commit `.env` or `firebase-service-account.json`.

---

## Known Limitations

- RETRY and FALLBACK decision-layer outcomes are logged but not yet acted upon — no automatic re-retrieval or model-switching loop yet
- `FALLBACK_LLM_MODEL` is not configured by default; fallback decisions currently fall through to ACCEPT
- Evaluation is single-turn only (no multi-turn conversation evaluation)
- No automated CI pipeline currently runs the test suite on push — tests must be run manually

---

## License

MIT License.