import asyncio
from ingestion.loader import load_document
from ingestion.chunking import chunk_documents
from vectorstore.vectordb import create_vectorstore
import tempfile
import uuid
import os

def test_pipeline():
    # Create a dummy docx
    test_file = "test_doc.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document. It contains some text to be chunked.\n" * 10)
    
    print("Loading document...")
    try:
        docs = load_document(test_file, "test_doc.txt")
        print(f"Loaded {len(docs)} documents.")
        
        print("Chunking documents...")
        chunks = chunk_documents(docs)
        print(f"Chunked into {len(chunks)} chunks.")
        
        print("Creating vectorstore...")
        sid = str(uuid.uuid4())
        create_vectorstore(sid, chunks)
        print("Vectorstore created.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_pipeline()
