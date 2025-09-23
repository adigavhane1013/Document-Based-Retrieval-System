import os
import re # Import the regular expressions library for text cleaning
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2

# --- LangChain Imports ---
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# 1. Configuration & Initialization
# ==============================================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# --- Ollama Model Configuration ---
# Switched to the more powerful 'mistral' model for higher accuracy
OLLAMA_MODEL = "tinyllama"

# In-memory storage for the LangChain retrieval chain
retrieval_chain = None

# Ensure the 'uploads' directory exists
os.makedirs("uploads", exist_ok=True)


# --- Initialize LangChain Components ---
llm = Ollama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

# Define the Prompt Template for the final answer generation
# This remains strict to prevent hallucination
answer_prompt = PromptTemplate.from_template("""
You are a precise and factual document assistant.
Your task is to answer the question using ONLY the provided context.
Do not use any of your own general knowledge.
If the information is not present in the context below, you MUST reply with 'The document does not contain this information.'

Context:
{context}

Question: {input}

Answer:""")

# --- NLP Enhancement: LangChain Chain for Query Expansion ---
# This chain rephrases the user's question to be more effective for retrieval
query_expansion_prompt = PromptTemplate.from_template("""
You are an expert at rephrasing questions for a vector database.
Given the original question, generate a new version that is more likely to retrieve relevant documents.
Consider synonyms, alternative phrasings, and key concepts.
Only output the rephrased question, nothing else.

Original Question: {question}
Rephrased Question:""")

query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()


# ==============================================================================
# 2. NLP Pre-processing & Document Functions
# ==============================================================================

def preprocess_text(text: str) -> str:
    """Cleans up text extracted from a PDF."""
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Join hyphenated words that were split across lines
    text = text.replace('-\n', '')
    # Remove standalone numbers (often page numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text

def process_document(file_path: str):
    """
    Loads, cleans, splits, and creates a LangChain retrieval chain for a document.
    """
    global retrieval_chain

    # Load and clean the document text
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += preprocess_text(page_text)

    if not text.strip():
        raise ValueError("Could not extract clean text from the document.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    documents = text_splitter.create_documents([text])

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    # Create the main retrieval chain
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"Successfully processed document and created retrieval chain.")


# ==============================================================================
# 3. API Endpoints
# ==============================================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Ollama RAG backend with LangChain & NLP is running!"})

@app.route("/upload", methods=["POST"])
def upload_file_route():
    # ... (This route remains the same as the previous version)
    if "file" not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    try:
        process_document(file_path)
        return jsonify({"message": f"Successfully processed '{file.filename}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question_route():
    if not retrieval_chain:
        return jsonify({"error": "No document has been processed yet."}), 400

    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # --- NLP Enhancement: Expand the query first ---
        print(f"Original question: {question}")
        expanded_question = query_expansion_chain.invoke({"question": question})
        print(f"Expanded question: {expanded_question}")
        
        # Now, use the expanded question in the main retrieval chain
        response = retrieval_chain.invoke({"input": expanded_question})
        return jsonify({"answer": response['answer']})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    app.run(port=5001, debug=True)