# 📚 College Notes RAG Assistant

A simple GenAI application that allows you to ask questions about your uploaded PDF notes. It uses Retrieval-Augmented Generation (RAG) to provide accurate answers directly from your provided documents, referencing the exact page and source for transparency.

## Features
- **Document Ingestion**: Upload PDF notes (`data/notes.pdf`) and split them into chunks.
- **Vector Search**: Embeds chunks using Google Generative AI Embeddings and stores them in Pinecone for fast similarity search.
- **Q&A Interface**: A Streamlit UI to interact with your notes and ask questions.
- **Contextual Answers**: Uses Gemini 2.5 Flash to generate answers restricted to the uploaded documents to prevent hallucinations.

## Tech Stack
- **Framework**: Streamlit
- **LLM & Embeddings**: Google Gemini (`gemini-2.5-flash`, `models/gemini-embedding-001`)
- **Orchestration**: LangChain
- **Vector Database**: Pinecone

## Setup Instructions

### 1. Prerequisites
Ensure you have Python installed. You also need API keys for Google Gemini and Pinecone.

### 2. Clone the repository
```bash
git clone https://github.com/vrajchoksi2005/Simple_Rag.git
cd Simple_Rag
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory (you can copy from `.env.example` if it exists) and add your keys:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

### 5. Ingest the Data
Before running the app, ingest your PDF notes (place them at `data/notes.pdf`) into Pinecone:
```bash
python ingest.py
```

### 6. Run the Application
Start the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

## Usage
Simply type your question in the text box (e.g., "What is normalization in DBMS?"). The assistant will search the database, provide an answer, and list the exact sources (file and page numbers) it used to generate the response.
