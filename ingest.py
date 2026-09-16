import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing in .env")


# =========================================================
# 2. Load PDF
# =========================================================

PDF_PATH = "data/notes.pdf"

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Total pages loaded: {len(documents)}")


# =========================================================
# 3. Check page metadata
# =========================================================

print("\nChecking page metadata...")

for doc in documents[:3]:

    print({
        "page": doc.metadata.get("page"),
        "page_label": doc.metadata.get("page_label"),
        "source": doc.metadata.get("source")
    })


# =========================================================
# 4. Split documents into chunks
# =========================================================

print("\nSplitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# =========================================================
# 5. Create Gemini Embedding Model
# =========================================================

print("\nCreating Gemini embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=1024
)

print("Embedding model ready.")


# =========================================================
# 6. Store chunks + embeddings in Pinecone
# =========================================================

print("\nUploading documents to Pinecone...")

vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("\n======================================")
print("Documents successfully stored!")
print("Pinecone index:", PINECONE_INDEX_NAME)
print("Total chunks:", len(chunks))
print("======================================")