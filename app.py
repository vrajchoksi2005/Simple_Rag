import os

import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. Page configuration
# =========================================================

st.set_page_config(
    page_title="College Notes RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. Load environment variables
# =========================================================

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_INDEX_NAME:
    st.error("PINECONE_INDEX_NAME is missing in .env")
    st.stop()


# =========================================================
# 3. Title
# =========================================================

st.title("📚 College Notes RAG Assistant")

st.write(
    "Ask questions about your uploaded PDF notes."
)


# =========================================================
# 4. Gemini Embedding Model
# =========================================================

@st.cache_resource
def load_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=1024
    )


embeddings = load_embeddings()


# =========================================================
# 5. Connect to Pinecone
# =========================================================

@st.cache_resource
def load_vector_store():

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )


vector_store = load_vector_store()


# =========================================================
# 6. Gemini LLM
# =========================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


llm = load_llm()


# =========================================================
# 7. RAG Prompt
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful college study assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- Give a clear and easy-to-understand answer.
- If the answer cannot be found in the context, say:
  "I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# 8. Format retrieved documents
# =========================================================

def format_docs(docs):

    formatted_docs = []

    for doc in docs:

        page_number = doc.metadata.get("page")

        if page_number is not None:
            page_number += 1

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        formatted_docs.append(
            f"""
Source: {source}
Page: {page_number}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted_docs)


# =========================================================
# 9. RAG function
# =========================================================

def ask_question(question):

    # ---------------------------------------------
    # Retrieve relevant documents
    # ---------------------------------------------

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    # ---------------------------------------------
    # Create context
    # ---------------------------------------------

    context = format_docs(docs)

    # ---------------------------------------------
    # Create prompt
    # ---------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # ---------------------------------------------
    # Generate answer using Gemini
    # ---------------------------------------------

    response = llm.invoke(messages)

    return response.content, docs


# =========================================================
# 10. User Input
# =========================================================

question = st.text_input(
    "🔎 Ask your question",
    placeholder="Example: What is normalization in DBMS?"
)


# =========================================================
# 11. Generate Answer
# =========================================================

if question:

    with st.spinner("Searching your documents..."):

        try:

            answer, docs = ask_question(question)

            # -----------------------------------------
            # Answer
            # -----------------------------------------

            st.subheader("💡 Answer")

            st.write(answer)


            # -----------------------------------------
            # Sources
            # -----------------------------------------

            st.subheader("📚 Sources")

            if docs:

                for i, doc in enumerate(docs):

                    page = doc.metadata.get("page")

                    if page is not None:
                        page += 1
                    else:
                        page = "Unknown"

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    file_name = os.path.basename(source)

                    with st.expander(
                        f"📄 Source {i + 1} — {file_name} — Page {page}"
                    ):

                        st.write(
                            f"**File:** {file_name}"
                        )

                        st.write(
                            f"**Page:** {page}"
                        )

                        st.write(
                            "**Retrieved content:**"
                        )

                        st.write(
                            doc.page_content
                        )

            else:

                st.info("No source documents found.")


        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )