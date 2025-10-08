import os
import re
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# --- API Key Loading and Cleaning ---
raw_openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if raw_openrouter_api_key:
    openrouter_api_key = raw_openrouter_api_key.strip().strip('"').strip("'")
else:
    openrouter_api_key = None

print(f"--- Loaded OpenRouter API Key: [{openrouter_api_key[:10] if openrouter_api_key else 'None'}...{openrouter_api_key[-4:] if openrouter_api_key else ''}] ---")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file. Please create a .env file and add your key.")

# --- Model Configuration ---
OPENROUTER_MODEL = "mistralai/mixtral-8x7b-instruct"
retrieval_chain = None
os.makedirs("uploads", exist_ok=True)

# --- Initialize LangChain Components ---
llm = ChatOpenAI(
    model_name=OPENROUTER_MODEL,
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_kwargs={
        "default_headers": {
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Document AI RAG Backend"
        }
    }
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

answer_prompt = PromptTemplate.from_template("""
You are a highly precise and factual document assistant.
Your task is to answer the question based ONLY on the provided context.
Do not use any of your external knowledge.
If the information to answer the question is not present in the context below, you MUST state: 'The provided document does not contain information on this topic.'

Context:
{context}

Question: {input}

Answer:""")

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('-\n', '')
    return text

def process_document(file_path: str):
    global retrieval_chain
    print(f"Processing document: {file_path}")
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    if not full_text.strip():
        raise ValueError("Could not extract any text from the document.")
    cleaned_text = preprocess_text(full_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.create_documents([cleaned_text])
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("Successfully processed document and created retrieval chain.")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "OpenRouter RAG Backend is running!"})

@app.route("/upload", methods=["POST"])
def upload_file_route():
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
        return jsonify({"error": "No document processed yet. Please upload a file."}), 400
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        print(f"Received question: {question}")
        response = retrieval_chain.invoke({"input": question})
        return jsonify({"answer": response['answer']})
    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK ---")
        traceback.print_exc()
        print("---------------------------------\n")
        return jsonify({"error": f"An internal server error occurred. Check the backend terminal for details. Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)

