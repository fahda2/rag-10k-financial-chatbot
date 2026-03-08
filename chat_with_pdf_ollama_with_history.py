import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time

# ------------------------------------------------- #
# MODEL CONFIGURATION — same as original
# ------------------------------------------------- #
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "deepseek-r1:7b"

llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.5)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# ------------------------------------------------- #
# PERSONA — same as original
# ------------------------------------------------- #
persona = '''
You are an expert financial analyst specializing in Big Tech companies. 
You have deep knowledge of annual reports (10-K filings) for Alphabet (Google), Amazon, and Microsoft.

Rules:
- ONLY answer using information from the provided PDF documents
- If the answer is not in the documents, say "I cannot find that in the provided 10-K files"
- Always cite which company and which section your answer comes from
- Be precise with numbers — always include units (millions, billions, %)
- For comparisons across companies, structure your answer clearly by company.
Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents
'''

template = """
{persona}
        
Chat History:
<history>
{chat_history}
</history>

Given the context information and not prior knowledge, answer the following question:
Question: {user_input}
"""

# ------------------------------------------------- #
# PAGE CONFIG
# ------------------------------------------------- #
st.set_page_config(
    page_title="FinSight AI — 10-K Analyst",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------- #
# LUXURY CSS ONLY
# ------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Montserrat:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background-color: #080A0F;
    color: #E8E0D0;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0D1520 0%, #080A0F 50%, #0A0D08 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0F1A 0%, #080A0F 100%);
    border-right: 1px solid rgba(197,160,80,0.2);
}
[data-testid="stFileUploader"] {
    background: rgba(197,160,80,0.03);
    border: 1px dashed rgba(197,160,80,0.2);
    border-radius: 2px;
}
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(197,160,80,0.08) !important;
    border-radius: 2px !important;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(197,160,80,0.35) !important;
    color: rgba(197,160,80,0.8) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(197,160,80,0.08) !important;
    border-color: #C5A050 !important;
}
[data-testid="stExpander"] {
    background: rgba(197,160,80,0.03) !important;
    border: 1px solid rgba(197,160,80,0.1) !important;
    border-radius: 2px !important;
}
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #080A0F; }
::-webkit-scrollbar-thumb { background: rgba(197,160,80,0.25); }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------- #
# SIDEBAR
# ------------------------------------------------- #
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem 0;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.5rem; color:#C5A050; letter-spacing:0.25em;">◈ FINSIGHT</div>
        <div style="font-size:0.5rem; letter-spacing:0.3em; color:rgba(197,160,80,0.35); text-transform:uppercase; margin-top:0.3rem;">Intelligence Platform</div>
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(197,160,80,0.3),transparent); margin-bottom:1.2rem;"></div>
    <div style="font-size:0.55rem; letter-spacing:0.3em; text-transform:uppercase; color:rgba(197,160,80,0.45); margin-bottom:0.5rem;">Engine</div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="background:rgba(197,160,80,0.06);border:1px solid rgba(197,160,80,0.2);border-radius:2px;padding:0.35rem 0.7rem;font-size:0.7rem;color:rgba(197,160,80,0.8);margin-bottom:0.4rem;">🧠 {OLLAMA_LLM_MODEL}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(197,160,80,0.06);border:1px solid rgba(197,160,80,0.2);border-radius:2px;padding:0.35rem 0.7rem;font-size:0.7rem;color:rgba(197,160,80,0.8);margin-bottom:0.4rem;">🔢 {OLLAMA_EMBEDDING_MODEL}</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:rgba(197,160,80,0.06);border:1px solid rgba(197,160,80,0.2);border-radius:2px;padding:0.35rem 0.7rem;font-size:0.7rem;color:rgba(197,160,80,0.8);margin-bottom:1rem;">🗄️ FAISS · MMR Retrieval</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.55rem;letter-spacing:0.3em;text-transform:uppercase;color:rgba(197,160,80,0.45);margin-bottom:0.5rem;border-top:1px solid rgba(197,160,80,0.1);padding-top:0.8rem;">Sample Queries</div>', unsafe_allow_html=True)
    for q in [
        "How much cash does Amazon have at end of 2024?",
        "Compare Azure vs Google Cloud revenue",
        "What China risks do these companies mention?",
        "Which company has the highest operating margin?",
        "Compare R&D spending across all three companies",
        "What is AWS operating income in 2024?",
    ]:
        st.markdown(f'<div style="font-size:0.7rem;color:rgba(232,224,208,0.45);padding:0.35rem 0;border-bottom:1px solid rgba(197,160,80,0.06);line-height:1.4;">› {q}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("◈  Clear Conversation"):
        st.session_state.messages = []
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()

# ------------------------------------------------- #
# HEADER
# ------------------------------------------------- #
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1.5rem 0;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:3.2rem; font-weight:300; letter-spacing:0.15em;
        background:linear-gradient(135deg,#C5A050 0%,#F0D080 40%,#C5A050 60%,#8B6914 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1.1;">
        FinSight AI
    </div>
    <div style="width:120px; height:1px; background:linear-gradient(90deg,transparent,#C5A050,transparent); margin:0.8rem auto;"></div>
    <div style="font-size:0.65rem; letter-spacing:0.35em; color:rgba(197,160,80,0.55); text-transform:uppercase; font-weight:500;">
        10-K Annual Report Intelligence &nbsp;·&nbsp; Alphabet &nbsp;·&nbsp; Amazon &nbsp;·&nbsp; Microsoft
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------- #
# EVERYTHING BELOW IS IDENTICAL TO ORIGINAL WORKING CODE
# ------------------------------------------------- #
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []

    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDFs uploaded and processed! You can now start chatting.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
        )

        chat_history = ""
        if len(st.session_state.messages) > 1:
            for i, msg in enumerate(st.session_state.messages[:-1]):
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"

        prompt_template = PromptTemplate.from_template("""
{persona}

Chat History:
{chat_history}

Use the following context from the documents to answer the question.
Context:
{context}

Question: {question}

Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "persona": lambda _: persona,
                "chat_history": lambda _: chat_history,
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            source_docs = retriever.invoke(user_input)
            response_text = chain.invoke(user_input)

            with st.expander("View Retrieved Chunks (Context)"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(f"**Content:** {doc.page_content}")
                    st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                    st.markdown("---")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; margin:2rem 0;">
        <div style="background:linear-gradient(135deg,rgba(197,160,80,0.05),rgba(255,255,255,0.02));
            border:1px solid rgba(197,160,80,0.15); border-radius:2px; padding:1.6rem; position:relative;">
            <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#C5A050,transparent);border-radius:2px 0 0 2px;"></div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;color:#C5A050;letter-spacing:0.1em;">GOOGL</div>
            <div style="font-size:0.6rem;letter-spacing:0.2em;color:rgba(232,224,208,0.45);text-transform:uppercase;margin-top:0.2rem;">Alphabet Inc.</div>
            <div style="font-size:0.72rem;color:rgba(232,224,208,0.55);margin-top:0.9rem;line-height:1.6;font-weight:300;">Search, YouTube, Google Cloud, Waymo & Other Bets. FY2024 Annual Report.</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(197,160,80,0.05),rgba(255,255,255,0.02));
            border:1px solid rgba(197,160,80,0.15); border-radius:2px; padding:1.6rem; position:relative;">
            <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#C5A050,transparent);border-radius:2px 0 0 2px;"></div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;color:#C5A050;letter-spacing:0.1em;">AMZN</div>
            <div style="font-size:0.6rem;letter-spacing:0.2em;color:rgba(232,224,208,0.45);text-transform:uppercase;margin-top:0.2rem;">Amazon.com Inc.</div>
            <div style="font-size:0.72rem;color:rgba(232,224,208,0.55);margin-top:0.9rem;line-height:1.6;font-weight:300;">AWS, North America Retail, International & Advertising. FY2024 Annual Report.</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(197,160,80,0.05),rgba(255,255,255,0.02));
            border:1px solid rgba(197,160,80,0.15); border-radius:2px; padding:1.6rem; position:relative;">
            <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#C5A050,transparent);border-radius:2px 0 0 2px;"></div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;color:#C5A050;letter-spacing:0.1em;">MSFT</div>
            <div style="font-size:0.6rem;letter-spacing:0.2em;color:rgba(232,224,208,0.45);text-transform:uppercase;margin-top:0.2rem;">Microsoft Corp.</div>
            <div style="font-size:0.72rem;color:rgba(232,224,208,0.55);margin-top:0.9rem;line-height:1.6;font-weight:300;">Azure, Office 365, LinkedIn, Dynamics & Gaming. FY2024 Annual Report.</div>
        </div>
    </div>
    <div style="text-align:center;margin-top:2rem;color:rgba(197,160,80,0.3);font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;">
        Upload your 10-K PDF files above to begin analysis
    </div>
    """, unsafe_allow_html=True)