# DocMind

DocMind is a FastAPI-based Retrieval-Augmented Generation (RAG) application for grounded question answering over user-uploaded documents and a shared, pre-indexed Company Knowledge Base. It combines hybrid retrieval, cross-encoder reranking, query rewriting, citation-based grounding checks, optional RAGAS evaluation, a decision layer, Firebase authentication, and performance instrumentation.


## Features

- Firebase Authentication with backend token verification.
- Upload and query PDF, DOCX, TXT, and Markdown documents.
- Multiple documents per user session.
- Dense Qdrant retrieval combined with BM25 sparse retrieval.
- Reciprocal Rank Fusion (RRF), deduplication, and cross-encoder reranking.
- Query rewriting when a question is sufficiently ambiguous.
- Token-budgeted context construction that preserves complete chunks.
- Grounded generation with `[SOURCE:chunk_id]` citations.
- Citation-based grounding score and source metadata.
- Optional RAGAS faithfulness and answer-relevancy evaluation.
- Decision outcomes: `ACCEPT`, `RETRY`, `FALLBACK`, and `REJECT`.
- Trace-level evaluation caching and structured logging.
- Performance dashboard at `performance.html`.
- Shared Company Knowledge Base available to authenticated users without uploading documents.

## Architecture

### Request flow

```text
Firebase ID-token verification
          |
          v
Resolve user-owned session or shared company_kb session
          |
          v
Load or reconstruct session from Qdrant
          |
          v
Query rewriting when ambiguity exceeds threshold
          |
          v
Hybrid retrieval
  ├── Dense Qdrant search
  └── Sparse BM25 search
          |
          v
Reciprocal Rank Fusion + deduplication
          |
          v
Cross-encoder reranking
          |
          v
Build context within the approximately 4096-token budget
          |
          v
Generate grounded answer with Groq
          |
          v
Parse citations and calculate grounding score
          |
          v
Optional RAGAS evaluation and decision layer
          |
          v
Return answer, sources, decision, trace and performance metrics
```

### Retrieval and context construction

- Dense retrieval uses Qdrant with `TOP_K_DENSE = 25` and a default similarity threshold of `0.35`.
- Sparse retrieval uses BM25 with `TOP_K_SPARSE = 25`.
- Reciprocal Rank Fusion combines and deduplicates the candidate results.
- The cross-encoder reranker reduces the candidates to `TOP_K_RERANK = 12`.
- Context construction adds complete chunks in reranked order up to the configured context budget. Chunks are not truncated mid-block.
- Query rewriting is enabled by default and uses Groq with `openai/gpt-oss-20b`.

### Grounding and decision layer

The grounding guardrail extracts `[SOURCE:chunk_id]` citations and calculates a score between `0.0` and `1.0`. When evaluation scores are supplied, the decision layer can apply these outcomes:

- **`ACCEPT`**: Return the answer when it meets the configured checks.
- **`RETRY`**: Increase retrieval breadth, lower the retrieval threshold, rerank, generate again, and re-evaluate. Retries are limited by `RAGAS_MAX_RETRY_ATTEMPTS`.
- **`FALLBACK`**: Generate using `FALLBACK_LLM_MODEL` with the existing retrieved context. No fallback model is configured by default; if the path is selected without one, the pipeline returns a refusal response.
- **`REJECT`**: Return a refusal response. The decision layer applies a grounding-score penalty.

The decision layer is used when evaluation scores are passed into the pipeline, such as with `run_evaluation: true` on `/ask`. Evaluation does not run automatically for every `/ask` request.

### Company Knowledge Base

The shared Company Knowledge Base uses the reserved session ID `company_kb` and a dedicated Qdrant collection named `company_kb`. It follows the same RAG pipeline as user sessions rather than a separate query pipeline.

The indexed corpus contains six Markdown policy documents and **118 chunks**:

1. `01_HR_Policy_Manual.md` — 20 chunks
2. `02_Remote_Work_Policy.md` — 19 chunks
3. `03_Travel_and_Expense_Policy.md` — 19 chunks
4. `04_IT_Security_Policy.md` — 21 chunks
5. `05_Engineering_Guidelines.md` — 20 chunks
6. `06_Code_of_Conduct_and_Ethics.md` — 19 chunks

Ingestion uses structure-aware hierarchical chunking and deterministic point IDs to avoid duplicate points during re-ingestion. Company KB sources include policy and section metadata; user-uploaded document sources retain their filename and page metadata. The shared session is reconstructed from Qdrant on startup and is not stored in `sessions.json`.

The prompt is designed to keep policy answers scoped to explicit evidence: distinguish defaults from exceptions, preserve all mandatory conditions, avoid transferring rules across scenarios, and state when the provided policies do not specify an answer. Factual claims must be supported by source citations.

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI, Uvicorn |
| Frontend | HTML5, CSS3, JavaScript (`index.html`, `performance.html`) |
| Authentication | Firebase Authentication, `firebase-admin` |
| Primary LLM | Groq API, `qwen/qwen3.8-27b` |
| Query-rewriting model | Groq API, `openai/gpt-oss-20b` |
| RAGAS evaluator model | Groq API, `llama-3.3-70b-versatile` |
| Embeddings | SentenceTransformers, `BAAI/bge-small-en-v1.5` |
| Vector database | Qdrant |
| Sparse retrieval | BM25 via `rank_bm25` |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | SentenceTransformers, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | RAGAS |
| Observability | `time.perf_counter_ns()` and structured JSON logging |
| Frontend hosting | Vercel |
| Backend hosting | Railway |

## Configuration

Settings are defined centrally in `configs/settings.py`. The following are the documented defaults:

| Setting | Default |
|---|---|
| `LLM_MODEL` | `qwen/qwen3.8-27b` |
| `LLM_TEMPERATURE` | `0.0` |
| `LLM_MAX_TOKENS` | `1536` |
| `LLM_CONTEXT_MAX_TOKENS` | `4096` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_BATCH_SIZE` | `64` |
| `EMBEDDING_DEVICE` | `cpu` |
| `COMPANY_KB_COLLECTION` | `company_kb` |
| `TOP_K_DENSE` | `25` |
| `TOP_K_SPARSE` | `25` |
| `TOP_K_RERANK` | `12` |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` |
| `HYBRID_ALPHA` | `0.7` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANKER_ENABLED` | `true` |
| `RAGAS_FAITHFULNESS_THRESHOLD` | `0.7` |
| `RAGAS_RELEVANCE_THRESHOLD` | `0.65` |
| `RAGAS_MAX_RETRY_ATTEMPTS` | `2` |
| `RETRY_RETRIEVAL_TOP_K_INCREASE` | `10` |
| `RETRY_LOWERED_THRESHOLD` | `0.3` |
| `ENABLE_QUERY_REWRITING` | `true` |
| `QUERY_AMBIGUITY_THRESHOLD` | `0.4` |
| `QUERY_REWRITE_MODEL` | `openai/gpt-oss-20b` |

RAGAS contexts are limited by `RAGAS_MAX_CONTEXTS` and `RAGAS_CONTEXT_MAX_CHARS`. Missing or invalid metric values remain unavailable rather than being fabricated as numeric scores.

## Project Structure

```text
.
├── configs/
│   └── settings.py
├── embeddings/
│   └── embedding_model.py
├── evaluation/
│   └── ragas_eval.py
├── guardrails/
│   └── hallucination_filter.py
├── ingestion/
│   ├── chunking.py
│   ├── company_kb.py
│   └── loader.py
├── observability/
│   └── logger.py
├── rag/
│   ├── decision_layer.py
│   ├── pipeline.py
│   ├── prompt.py
│   └── query_rewriter.py
├── retrieval/
│   ├── reranker.py
│   └── retriever.py
├── storage/                 # Runtime session and evaluation data
├── vectorstore/
│   └── vectordb.py
├── company_documents/       # Company policy Markdown files
├── auth.py
├── main.py
├── index.html               # Main chat interface
├── performance.html         # Performance dashboard
├── requirements.txt
└── .env                     # Local configuration; do not commit
```

Runtime-generated files and directories may not exist in a fresh clone. Refer to the repository for the current complete file layout.

## Setup

### Prerequisites

- Python 3.8.10 or later.
- A Groq API key.
- A Qdrant instance and its URL/API key.
- A Firebase project with Email/Password sign-in enabled.

### Install

```bash
git clone https://github.com/adigavhane1013/Document-Based-Retrieval-System.git
cd Document-Based-Retrieval-System

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The embedding and reranker models are loaded through SentenceTransformers and may be downloaded on first use. The configured embedding and reranker devices default to CPU.

### Firebase setup

1. Enable **Authentication → Sign-in method → Email/Password** in Firebase.
2. Configure the Firebase web app settings used by `index.html`.
3. For local backend use, provide Firebase Admin credentials using the configuration expected by the application, such as `firebase-service-account.json` or `FIREBASE_SERVICE_ACCOUNT_JSON`.
4. Never commit service-account credentials.

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

The application also supports configuration through environment variables. Set `FALLBACK_LLM_MODEL` if you want to enable the fallback generation path. Do not commit `.env`, API keys, Firebase tokens, or service-account credentials.

### Run the backend

```bash
python main.py
```

The API normally runs at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

Serve the frontend over HTTP rather than opening the HTML file directly with `file://`. For example:

```bash
python -m http.server 5500
```

Open `http://localhost:5500/` and ensure the frontend's API base URL points to the backend being used.

## API

All endpoints except `GET /health` require a Firebase ID token in the request header:

```http
Authorization: Bearer <firebase-id-token>
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check; does not require authentication |
| `POST` | `/upload` | Upload a supported document and optionally add it to an existing session |
| `POST` | `/ask` | Ask a question in a user-owned session or the shared `company_kb` session |
| `POST` | `/evaluate` | Evaluate a previously generated answer using `session_id` and `trace_id` |
| `GET` | `/sessions` | List accessible sessions, including the shared Company KB |
| `GET` | `/session/{session_id}` | Retrieve an accessible session and its chat history |
| `DELETE` | `/session/{session_id}` | Delete an accessible user-owned session |

### Ask a question

```bash
curl -X POST "https://document-based-retrieval-system-production.up.railway.app/ask" \
  -H "Authorization: Bearer <firebase-id-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "question": "What are the key findings?",
    "run_evaluation": false
  }'
```

Set `run_evaluation` to `true` to run RAGAS during the request and pass its scores to the decision layer. The shared Company KB session ID is `company_kb`.

### Evaluate an existing answer

```bash
curl -X POST "https://document-based-retrieval-system-production.up.railway.app/evaluate" \
  -H "Authorization: Bearer <firebase-id-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "trace_id": "trace-id-from-ask-response"
  }'
```

The backend verifies ownership of the referenced message before using the evaluation cache.

## Evaluation

The configured RAGAS metrics are:

| Metric | Meaning |
|---|---|
| Faithfulness | Measures whether the answer is supported by the retrieved context |
| Answer Relevancy | Measures how well the answer addresses the question |
| Hallucination rate | Derived as `1 - faithfulness` when a valid faithfulness score is available |

Evaluation can be triggered during `/ask` with `run_evaluation: true`, or after generation through `/evaluate` with `session_id` and `trace_id`. Evaluation includes retry/backoff handling, and results are cached by trace ID. Because contexts are limited before evaluation, the evaluated context may be a subset of the original retrieved material. If a metric fails or returns an invalid value, its result may be unavailable.

## Performance Dashboard

Open `performance.html` through a local HTTP server or the deployed frontend setup. The dashboard sends authenticated test queries to `/ask` with `run_evaluation: false` and displays API-provided information including:

- Total request latency.
- Retrieval, reranking, and LLM-generation latency.
- Input, output, and total token counts when available.
- Grounding score and decision status.
- Sources, trace ID, and timestamp.

Performance measurements use `time.perf_counter_ns()`. Latency and token counters accumulate across initial, retry, and fallback attempts. Token values that are unavailable remain `null`/unavailable and are not reported as zero.

The dashboard derives its overhead value as:

```text
max(0, total latency - retrieval latency - reranking latency - LLM latency)
```

This overhead is a calculated remainder, not an independently timed pipeline stage. A single dashboard run is one request, not a multi-run benchmark.

The main chat interface does not display every detailed performance field inline; the dedicated dashboard provides the more detailed view.

## Authentication and Data Access

- Firebase ID tokens are sent as Bearer tokens and verified by Firebase Admin on the backend.
- User sessions use IDs such as `session_<uuid>` and are scoped to their authenticated owner.
- Upload, query, and evaluation endpoints enforce ownership checks for user sessions.
- The reserved `company_kb` session is intentionally shared among authenticated users.
- The Company KB collection is separate from user-session collections.

## Verification Status and Limitations

The project context records the Company KB ingestion, pipeline integration, metadata flow, end-to-end flow, and performance instrumentation as verified. It records a manual smoke test for a remote-work device policy question with a 100% grounding score and 12 cited sources. These are documented project results, not a claim that tests were executed while updating this README.

Known limitations:

- The cross-encoder reranker runs on CPU in development and can contribute substantially to warm-query latency.
- The primary chat UI does not display all detailed performance and decision fields inline.
- Broader formal regression/evaluation suites and deeper latency profiling are listed as remaining work.

## Testing

Execute tests manually when needed. Do not treat a test as passed unless it has actually been run and its output verified. The project context lists broader formal regression and evaluation runs as remaining work.

## License

See the repository for the applicable license.
