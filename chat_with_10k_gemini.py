import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import time

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="10-K Financial Analyst", layout="wide")
st.title("📊 10-K Financial Document Analyst (Gemini)")
st.caption("Upload SEC 10-K filings to ask questions and compare companies.")

# ─────────────────────────────────────────────
# Sidebar — API key & settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Prefer env variable; fall back to sidebar input
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Set GOOGLE_API_KEY env variable or paste here. Never commit your key to code.",
        )

    st.markdown("---")
    st.markdown("**Retrieval settings**")
    top_k = st.slider("Chunks retrieved per query (k)", min_value=2, max_value=10, value=5)
    chunk_size = st.slider("Chunk size (tokens)", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=300, value=150, step=50)

    st.markdown("---")
    if st.button("🗑️ Clear chat & reload documents"):
        for key in ["vector_store", "messages", "loaded_files"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─────────────────────────────────────────────
# Validate API key before doing anything
# ─────────────────────────────────────────────
if not api_key:
    st.warning("Please enter your Google Gemini API key in the sidebar to get started.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)

# ─────────────────────────────────────────────
# Persona & prompt
# ─────────────────────────────────────────────
PERSONA = """
You are an expert financial analyst specializing in SEC 10-K annual filings.
Your role is to help users understand, compare, and extract insights from 10-K documents.

When answering:
- Cite the specific company, fiscal year, and section (e.g., "Risk Factors", "MD&A", "Financial Statements") when possible.
- Use precise financial figures from the documents (revenue, net income, EPS, etc.).
- When comparing multiple companies, structure your answer clearly by company.
- If a question cannot be answered from the provided documents, say so explicitly — do not fabricate numbers.
- Keep answers concise but thorough. Use bullet points or tables where helpful.
"""

PROMPT_TEMPLATE = PromptTemplate.from_template("""
{persona}

Chat History:
<history>
{chat_history}
</history>

Context from documents:
{context}

Using only the information in the provided documents, answer the following question:
Question: {question}
""")

# ─────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload 10-K PDF files (one or more companies)",
    accept_multiple_files=True,
    type=["pdf"],
)

if uploaded_files:
    # Detect if the set of uploaded files changed
    current_file_names = sorted([f.name for f in uploaded_files])
    previous_file_names = st.session_state.get("loaded_files", [])

    if current_file_names != previous_file_names:
        # Files changed — reset vector store so it rebuilds
        st.session_state.pop("vector_store", None)
        st.session_state.pop("messages", None)

    # Build vector store if not already built
    if "vector_store" not in st.session_state:
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                loader = PyPDFLoader(temp_path)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_file"] = file.name
                documents.extend(loaded)

        # Larger chunks suit 10-K structure (dense tables, long paragraphs)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = splitter.split_documents(documents)

        # ── Batched embedding with rate-limit retry ──────────────────────
        BATCH_SIZE = 80  # stay under 100 req/min free tier limit
        total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

        status_text = st.empty()
        progress_bar = st.progress(0)
        vector_store = None

        for batch_idx in range(total_batches):
            batch = docs[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            status_text.text(f"Embedding batch {batch_idx + 1}/{total_batches} ({len(docs)} chunks total)...")

            # Retry loop for rate-limit errors
            retries = 0
            while True:
                try:
                    batch_store = FAISS.from_documents(batch, embeddings)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_sec = min(60, 15 * (2 ** retries))
                        status_text.text(f"Rate limit hit — waiting {wait_sec}s before retrying...")
                        time.sleep(wait_sec)
                        retries += 1
                    else:
                        raise

            if vector_store is None:
                vector_store = batch_store
            else:
                vector_store.merge_from(batch_store)

            progress_bar.progress((batch_idx + 1) / total_batches)

            # Pace requests between batches to avoid hitting the rate limit
            if batch_idx < total_batches - 1:
                time.sleep(65)

        status_text.empty()
        progress_bar.empty()
        # ────────────────────────────────────────────────────────────────

        st.session_state.vector_store = vector_store
        st.session_state.loaded_files = current_file_names

        file_list = ", ".join(current_file_names)
        st.success(f"✅ Loaded {len(docs)} chunks from: {file_list}")

    # ─────────────────────────────────────────
    # Chat interface
    # ─────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about the 10-K filings (e.g. 'What are the main risk factors for Amazon?')")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build chat history string
        chat_history = ""
        for msg in st.session_state.messages[:-1]:
            role = "User" if msg["role"] == "user" else "Analyst"
            chat_history += f"{role}: {msg['content']}\n\n"

        # Retriever with MMR for diverse, relevant chunks
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 4, "lambda_mult": 0.7},
        )

        with st.spinner("Analyzing documents..."):
            # Retrieve source docs separately so we can display them
            source_docs = retriever.invoke(user_input)
            context = "\n\n".join(doc.page_content for doc in source_docs)

            chain = PROMPT_TEMPLATE | llm | StrOutputParser()
            response_text = chain.invoke({
                "persona": PERSONA,
                "chat_history": chat_history,
                "context": context,
                "question": user_input,
            })

        # Show source chunks in expander
        if source_docs:
            with st.expander("📄 View Retrieved Chunks"):
                for i, doc in enumerate(source_docs):
                    src = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i+1}** — `{src}` | Page {page}")
                    st.markdown(doc.page_content)
                    st.markdown("---")

        # Stream response word-by-word
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for word in response_text.split():
                full_response += word + " "
                placeholder.markdown(full_response)
                time.sleep(0.03)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("Upload one or more 10-K PDF files above to begin.")
