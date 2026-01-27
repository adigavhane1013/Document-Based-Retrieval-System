# RAG PDF Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot backend built with FastAPI and LangChain that allows multi-document question answering with persistent session management.
***

## What This Project Does

1. Accepts PDF document uploads
2. Splits documents into chunks and creates embeddings
3. Stores vectors in ChromaDB with session persistence
4. Answers questions strictly based on document context
5. Manages multiple chat sessions with automatic cleanup
6. Tracks LLM calls with LangSmith integration (optional)

***

## Tech Stack

- **Backend**: FastAPI
- **LLM**: OpenRouter (Mixtral-8x7B-Instruct)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB
- **RAG Framework**: LangChain
- **Monitoring**: LangSmith (optional)

***

## Key Features

- **Multi-document support** – Add multiple PDFs to a single session
- **Persistent sessions** – Sessions and vector stores survive server restarts 
- **Hallucination prevention** – Strict prompt template restricts answers to document context 
- **Automatic cleanup** – LRU-based session removal when memory limits are exceeded 
- **Source tracking** – Returns page numbers and chunk references for each answer 

***

## How to Use

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file:
```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=mistralai/mixtral-8x7b-instruct
TEMPERATURE=0.2
```

3. Run the application:
```bash
python app_fixed.py
```

4. API will be available at `http://localhost:8000` 

***

## Configuration

Key settings in `.env`:

- `CHUNK_SIZE` – Text chunk size 
- `CHUNK_OVERLAP` – Overlap between chunks 
- `TOP_K_RESULTS` – Number of chunks to retrieve
- `MAX_SESSIONS_IN_MEMORY` – Session limit before cleanup

***

## Project Structure

```
├── app.py          # Main FastAPI application
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── storage/              # Session metadata
│   └── sessions.json
└── vectorstore/          # ChromaDB vector stores
    └── session_{id}/
```

***

## Notes

- PDF files only (max 50MB) 
- Sessions are automatically rebuilt on server restart
- Vector stores are persisted to disk after each upload
- Designed for single-user configuration per session

***

## Future Improvements

- Multi-format document support (DOCX, TXT, HTML)
- User authentication and multi-tenancy
- Conversation memory and context tracking
- Real-time streaming responses
- OCR integration for scanned documents

***

## Author

**Aditya Gavhane**  
Final-year B.Tech student (AI & Analytics)  

