🤖 AI Document Search RAG Chatbot
A full-stack AI-powered chatbot that provides factually grounded answers from user-uploaded documents using a Retrieval-Augmented Generation (RAG) architecture.

🚀 Features
Document Upload: Allows users to upload their own documents (e.g., PDFs, TXT) to create a custom knowledge base.

Retrieval-Augmented Generation (RAG): Implements a RAG pipeline to retrieve relevant document chunks and generate contextually accurate answers, significantly reducing hallucinations.

Persistent Knowledge Base: Utilizes a MySQL database to store document text chunks and their corresponding vector embeddings for efficient, long-term retrieval.

Interactive UI: A modern and responsive user interface built with React for seamless user interaction.

Persistent Chat History: Saves the conversation history, allowing users to reference previous questions and answers.

Open-Source LLM: Powered by the Mistral LLM via Ollama for powerful, locally-run language generation.

🧐 Project Overview
This chatbot addresses the challenge of getting reliable and context-specific answers from large language models. Instead of relying on the LLM's generic pre-trained knowledge, this application grounds its responses in the content of user-provided documents. When a user asks a question, the backend retrieves the most relevant text passages from the document database and feeds them to the LLM along with the original question. This ensures the generated answers are accurate, relevant, and directly supported by the source material.

⚙️ Technologies Used
Backend: Python, Flask, LangChain, Ollama

Frontend: React, JavaScript, HTML/CSS

AI/ML: Mistral LLM, Sentence Transformers (for embeddings), RAG

Database: MySQL

🛠️ Setup Instructions
Prerequisites:

Python 3.8+

Node.js and npm

Ollama installed and running with the Mistral model (ollama pull mistral)

A running MySQL server

1. Clone the repository:

Bash

git clone https://github.com/adigavhane1013/AI-Document-Search-RAG-Chatbot.git
cd AI-Document-Search-RAG-Chatbot
2. Backend Setup:

Bash

# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Configure your database connection in a .env file or directly in the code
# Run the backend server
python app.py
3. Frontend Setup:

Bash

# Open a new terminal and navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Run the React application
npm start
4. Access the application:
Open your web browser and navigate to http://localhost:3000.

🎮 Usage
Navigate to the web interface.

Use the upload feature to add a new document to the knowledge base.

Wait for the document to be processed and indexed.

Once ready, ask questions in the chat input box. The chatbot will provide answers based only on the content of your uploaded documents.

💡 Notes
Ensure your Ollama server is running with the Mistral model available before starting the backend.

Database credentials and other sensitive keys should be managed securely, preferably using a .env file in the backend directory.

The initial processing of a large document may take a few moments as embeddings are generated and stored.

👤 Author
Aditya Gavhane

📄 License
This project is licensed under the MIT License.
