"""
FinBot — Ultimate RAG Chatbot
BU.520.710 AI Essentials for Business — Final Project
Features:
  - Beautiful Bloomberg-terminal dark UI
  - Auto chart/graph generation (Plotly)
  - Multi-company comparison mode
  - Source citations per answer
  - Chat history export
  - Fully local by default: DeepSeek / Mistral / Llama via Ollama (free, private)
  - Optional Gemini API (only if you have quota)
"""

import streamlit as st
import os
import time
import json
import re
import tempfile
from datetime import datetime

# ── PDF & LangChain ──────────────────────────────────────────────────────────
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ── LLM options ──────────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# ── Chart / Plotly ────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — fill in your keys here
# ─────────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = "YOUR_GOOGLE_API_KEY"   # <-- paste your Gemini key
OPENAI_API_KEY  = ""                       # optional

OLLAMA_LLM_MODEL       = "deepseek-r1:7b"   # or mistral / llama3.1
OLLAMA_EMBED_MODEL     = "mxbai-embed-large"  # faster on CPU than nomic-embed-text

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinBot | 10-K Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Bloomberg Terminal Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    color: #e8edf5 !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 17px !important;
}
p, li, div, span {
    font-size: 16px !important;
    line-height: 1.7 !important;
}
[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 2px solid #1e3d6e !important;
}
[data-testid="stSidebar"] * {
    font-size: 15px !important;
    color: #d0dff0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Header banner ── */
.fin-header {
    background: linear-gradient(135deg, #0d2140 0%, #0a1830 100%);
    border: 2px solid #2a5a9e;
    border-radius: 12px;
    padding: 22px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 0 60px rgba(0,150,255,0.15);
}
.fin-header h1 {
    font-family: 'Exo 2', sans-serif;
    font-weight: 800;
    font-size: 3.2rem !important;
    color: #5dd5ff;
    margin: 0;
    letter-spacing: 4px;
    text-shadow: 0 0 40px rgba(93,213,255,0.8), 0 0 80px rgba(93,213,255,0.3);
}
.fin-header .subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem !important;
    color: #7a9dc0;
    letter-spacing: 3px;
    margin-top: 6px;
}
.live-badge {
    background: #0a2a15;
    border: 2px solid #1e7a40;
    color: #4dff80;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem !important;
    padding: 5px 14px;
    border-radius: 4px;
    letter-spacing: 2px;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── Chat bubbles ── */
.msg-user {
    background: linear-gradient(135deg, #0f2a45, #0c1f35);
    border: 2px solid #2a6aaa;
    border-radius: 16px 16px 4px 16px;
    padding: 18px 22px;
    margin: 18px 0;
    margin-left: 40px;
    color: #b8deff;
    font-size: 16px !important;
    line-height: 1.7;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,100,200,0.15);
}
.msg-user::before {
    content: "YOU";
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem !important;
    color: #5a9fd4;
    position: absolute;
    top: -22px;
    right: 4px;
    letter-spacing: 3px;
    font-weight: bold;
}
.msg-bot {
    background: linear-gradient(135deg, #0a2010, #08180c);
    border: 2px solid #1e6a30;
    border-radius: 16px 16px 16px 4px;
    padding: 18px 22px;
    margin: 18px 0;
    margin-right: 40px;
    color: #c8ffcc;
    font-size: 16px !important;
    line-height: 1.7;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,150,50,0.12);
}
.msg-bot::before {
    content: "Team 2's Finbot";
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem !important;
    color: #3dbb60;
    position: absolute;
    top: -22px;
    left: 4px;
    letter-spacing: 3px;
    font-weight: bold;
}
.msg-thinking {
    background: #0a1020;
    border: 2px solid #2a4080;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 10px 0;
    color: #7ab0e0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem !important;
    letter-spacing: 2px;
    animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

/* ── Source badge ── */
.source-badge {
    display: inline-block;
    background: #1f1a05;
    border: 2px solid #6a5a10;
    color: #ffe066;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem !important;
    padding: 4px 12px;
    border-radius: 4px;
    margin: 3px 4px;
    letter-spacing: 1px;
    font-weight: bold;
}
.source-section {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #2a3a5a;
    font-size: 0.8rem !important;
    color: #7a9dc0;
}

/* ── Input box ── */
.stTextInput > div > div > input {
    background: #0a1428 !important;
    border: 2px solid #2a5a9e !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 18px !important;
    padding: 14px 18px !important;
}
.stTextInput > div > div > input::placeholder {
    color: #FFFFF !important;
    font-size: 20px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5dd5ff !important;
    box-shadow: 0 0 20px rgba(93,213,255,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0f2a45, #0c1f35) !important;
    border: 2px solid #5dd5ff !important;
    color: #5dd5ff !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a4060, #162a45) !important;
    box-shadow: 0 0 24px rgba(93,213,255,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0a1428 !important;
    border: 2px solid #2a5a9e !important;
    color: #e8edf5 !important;
    font-size: 15px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0a1428 !important;
    border: 2px dashed #2a5a9e !important;
    border-radius: 10px !important;
}

/* ── Sidebar section labels ── */
.sidebar-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem !important;
    color: #5a9fd4;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 18px 0 10px 0;
    padding-bottom: 5px;
    border-bottom: 2px solid #1e3d6e;
}

/* ── Divider ── */
hr { border-color: #1e3d6e !important; border-width: 2px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c16; }
::-webkit-scrollbar-thumb { background: #2a5a9e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a7abe; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0a1428 !important;
    border: 2px solid #1e3d6e !important;
    border-radius: 8px !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 15px !important; }

/* ── Plotly ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; border: 2px solid #1e3d6e; }

/* ── Markdown inside bot messages ── */
.msg-bot strong { color: #7dffaa !important; font-size: 16px !important; }
.msg-bot ul, .msg-bot ol { padding-left: 20px; margin: 8px 0; }
.msg-bot li { margin: 6px 0; color: #c8ffcc; font-size: 16px !important; }
.msg-bot h3 { color: #5dd5ff !important; font-size: 18px !important; margin: 12px 0 6px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "vector_store": None,
    "docs_loaded": False,
    "docs_info": {},
    "qa_chain": None,
    "total_queries": 0,
    "avg_response_time": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def build_vector_store(texts: dict, llm_choice: str):
    """texts = {company_name: raw_text}"""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for company, text in texts.items():
        chunks = splitter.create_documents([text], metadatas=[{"source": company}] * len(splitter.split_text(text)))
        all_docs.extend(chunks)

    # Always use local Ollama embeddings — free, private, no API limits
    # Gemini embeddings only if explicitly selected AND key is set
    if llm_choice == "Gemini (Remote — optional)" and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        # nomic-embed-text: 8192 token context window, handles all chunk sizes fine
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vs = FAISS.from_documents(all_docs, embeddings)
    return vs


def build_qa_chain(vector_store, llm_choice: str):
    """Returns a dict with 'chain' (LCEL) and 'retriever' for source docs."""
    if llm_choice == "Gemini (Remote — optional)":
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY,
        )
    elif llm_choice == "DeepSeek r1:7b (Reasoning)":
        llm = OllamaLLM(model="deepseek-r1:7b", temperature=0.3)
    elif llm_choice == "Llama 3.1 (Balanced)":
        llm = OllamaLLM(model="llama3.1", temperature=0.3)
    else:
        llm = OllamaLLM(model="mistral", temperature=0.3)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 25, "lambda_mult": 0.7},
    )

    SYSTEM_PROMPT = """You are Team 2's Finbot, a warm and brilliant finance expert who has spent 30 years teaching at a top business school. You have been given the official 10-K annual filings for Alphabet (Google), Amazon, and Microsoft to analyze.

Your personality:
- You speak like a great teacher: clear, engaging, and never condescending, with detailed answers coming straight from the documents
- You love breaking down complex financial concepts as well as explaining it into simple language
- You use analogies and real-world comparisons to explain numbers
- You are enthusiastic about finance and it shows — you say things like "Great question!" or "This is a fascinating one..."
- You are honest and humble — if something is unclear you say so
- You occasionally add a touch of wit or a memorable phrase to make things stick

STRICT RULES you must always follow:
1. ONLY use facts from the provided 10-K context below. Never invent numbers or use outside knowledge.
2. If the answer is not in the documents say: "Hmm, I don't see that in the 10-K filings we have — great question to dig into though!"
3. Never fabricate financial figures, percentages, or statistics.
4. If someone greets you (hi, hello, hey), warmly introduce yourself and invite them to ask a financial question. Do NOT generate a report unprompted.
5. Always lead with the key number or direct answer first, then explain it simply.
6. When comparing all three companies use this clear structure:
   📘 ALPHABET | 📦 AMAZON | 🪟 MICROSOFT
7. End every financial answer with a section labeled: 💡 Team 2'S TAKEAWAY: — one insight that ties it all together in plain English.

Context from 10-K filings:
{context}

Conversation History:
{chat_history}

Student Question: {question}

Team 2's Finbot (answer ONLY from the 10-K context above):"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain — no deprecated RetrievalQA
    # Input must be a dict: {"question": str, "chat_history": str}
    from langchain_core.runnables import RunnableLambda

    def get_question(x):
        return x["question"] if isinstance(x, dict) else x

    def get_history(x):
        return x.get("chat_history", "") if isinstance(x, dict) else ""

    chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnableLambda(get_question),
            "chat_history": RunnableLambda(get_history),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def detect_chart_request(query: str):
    """Returns chart type if user asks for a visualization."""
    q = query.lower()
    if any(w in q for w in ["bar chart", "bar graph", "compare.*chart", "chart.*compar", "visualize", "plot", "graph"]):
        return "bar"
    if any(w in q for w in ["pie chart", "pie graph", "breakdown.*pie"]):
        return "pie"
    if any(w in q for w in ["line chart", "line graph", "trend", "over time", "timeline"]):
        return "line"
    return None


def extract_numbers_from_answer(answer: str):
    """
    Naive extractor: pulls (label, value) pairs from answer text.
    Looks for patterns like: "Amazon: $500B", "Google revenue: 307.4 billion"
    """
    patterns = [
        r'(Alphabet|Google|Amazon|Microsoft)[^\$\d]*[\$]?([\d,]+\.?\d*)\s*(billion|million|B|M|trillion|T)?',
    ]
    results = {}
    for pat in patterns:
        for m in re.finditer(pat, answer, re.IGNORECASE):
            company = m.group(1).title()
            raw = m.group(2).replace(",", "")
            multiplier = m.group(3) or ""
            try:
                val = float(raw)
                if multiplier.lower() in ("trillion", "t"):
                    val *= 1000
                elif multiplier.lower() in ("million", "m"):
                    val /= 1000
                results[company] = val
            except ValueError:
                pass
    return results


def render_auto_chart(answer: str, chart_type: str, query: str):
    """Render a Plotly chart from extracted numbers."""
    data = extract_numbers_from_answer(answer)
    if not data:
        st.info("🤖 I detected a chart request but couldn't parse specific numbers. Try asking e.g. 'create a bar chart of revenue for each company'.")
        return

    companies = list(data.keys())
    values = list(data.values())
    colors = ["#4fc3f7", "#66bb6a", "#ffa726"][:len(companies)]

    if chart_type == "bar":
        fig = go.Figure(go.Bar(
            x=companies, y=values,
            marker_color=colors,
            text=[f"${v:.1f}B" for v in values],
            textposition="outside",
            textfont=dict(color="#c8d6e5", size=13),
        ))
        fig.update_layout(
            title=dict(text=f"📊 {query[:60]}...", font=dict(color="#4fc3f7", size=14), x=0.02),
            paper_bgcolor="#0a0d14", plot_bgcolor="#0d1520",
            font=dict(color="#c8d6e5", family="Share Tech Mono"),
            xaxis=dict(gridcolor="#1a2744", zerolinecolor="#1a2744"),
            yaxis=dict(gridcolor="#1a2744", zerolinecolor="#1a2744",
                       title="USD (Billions)", title_font=dict(color="#546e8a")),
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False,
        )

    elif chart_type == "pie":
        fig = go.Figure(go.Pie(
            labels=companies, values=values,
            marker=dict(colors=colors, line=dict(color="#0a0d14", width=2)),
            textfont=dict(color="#c8d6e5"),
            hovertemplate="%{label}: $%{value:.1f}B<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"🥧 {query[:60]}", font=dict(color="#4fc3f7", size=14), x=0.02),
            paper_bgcolor="#0a0d14",
            font=dict(color="#c8d6e5", family="Share Tech Mono"),
            margin=dict(l=20, r=20, t=60, b=20),
        )

    else:  # line (fallback to bar when no time series)
        fig = go.Figure(go.Scatter(
            x=companies, y=values,
            mode="lines+markers",
            line=dict(color="#4fc3f7", width=2),
            marker=dict(color=colors, size=10),
        ))
        fig.update_layout(
            paper_bgcolor="#0a0d14", plot_bgcolor="#0d1520",
            font=dict(color="#c8d6e5", family="Share Tech Mono"),
            margin=dict(l=40, r=20, t=60, b=40),
        )

    st.plotly_chart(fig, use_container_width=True)


def format_sources(source_docs) -> str:
    seen = set()
    badges = []
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            badges.append(f'<span class="source-badge">📄 {src}</span>')
    return "".join(badges)


def export_chat():
    lines = []
    for msg in st.session_state.chat_history:
        role = "YOU" if msg["role"] == "user" else "Team 2's Finbot"
        lines.append(f"[{msg.get('time','?')}] {role}:\n{msg['content']}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:10px 0"><span style="font-family:Share Tech Mono;font-size:1.1rem;color:#4fc3f7;letter-spacing:3px"> Team 2 Finbot</span><br><span style="font-size:0.6rem;color:#3a5a7a;letter-spacing:4px">CONTROL PANEL</span></div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-section">⚙️ LLM Engine</div>', unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "Model",
        ["DeepSeek r1:7b (Reasoning)", "Llama 3.1 (Balanced)", "Mistral (Fast)", "Gemini (Remote — optional)"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">📂 Upload 10-K Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    company_names = []
    if uploaded_files:
        st.markdown('<div class="sidebar-section">🏷️ Company Labels</div>', unsafe_allow_html=True)
        for f in uploaded_files:
            suggested = f.name.replace(".pdf","").replace("_"," ").title()
            name = st.text_input(f"Label for {f.name[:20]}…", value=suggested, key=f"label_{f.name}")
            company_names.append(name)

    if uploaded_files and st.button("🚀  PROCESS & INDEX DOCUMENTS", use_container_width=True):
        prog = st.progress(0, text="📄 Reading PDFs...")
        texts = {}
        for i, (f, name) in enumerate(zip(uploaded_files, company_names)):
            prog.progress(int((i / len(uploaded_files)) * 40), text=f"📄 Reading {name}...")
            texts[name] = load_pdf_text(f)
        prog.progress(50, text="✂️ Chunking documents...")
        st.session_state.vector_store = build_vector_store(texts, llm_choice)
        prog.progress(90, text="🔗 Building QA chain...")
        st.session_state.qa_chain = build_qa_chain(st.session_state.vector_store, llm_choice)
        st.session_state.docs_loaded = True
        st.session_state.docs_info = {
            name: f"~{len(t.split())//200} pages (~{len(t):,} chars)"
            for name, t in texts.items()
        }
        prog.progress(100, text="✅ Done!")
        st.success("✅ Vector store ready!")

    if st.session_state.docs_loaded:
        st.markdown('<div class="sidebar-section">📊 Indexed Documents</div>', unsafe_allow_html=True)
        for name, info in st.session_state.docs_info.items():
            st.markdown(f'<div style="font-family:Share Tech Mono;font-size:0.7rem;color:#546e8a;padding:3px 0">✓ <span style="color:#4fc3f7">{name}</span><br>&nbsp;&nbsp;{info}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-section">📈 Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:Share Tech Mono;font-size:0.7rem;color:#546e8a">
    QUERIES ........ <span style="color:#4fc3f7">{st.session_state.total_queries}</span><br>
    AVG LATENCY ... <span style="color:#4fc3f7">{st.session_state.avg_response_time:.1f}s</span><br>
    MESSAGES ....... <span style="color:#4fc3f7">{len(st.session_state.chat_history)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.rerun()
    with col2:
        if st.session_state.chat_history:
            st.download_button(
                "💾 Export",
                data=export_chat(),
                file_name=f"finbot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fin-header">
  <div>
    <div class="fin-header h1" style="font-family:'Exo 2',sans-serif;font-weight:800;font-size:5.2rem;color:#5dd5ff;letter-spacing:4px;text-shadow:0 0 40px rgba(93,213,255,0.8)">📊 Team 2's Finbot </div>
    <div class="subtitle" style="font-family:'Share Tech Mono',monospace;font-size:1rem;color:#7ab0e0;letter-spacing:3px;margin-top:6px">10-K INTELLIGENCE ENGINE · ALPHABET · AMAZON · MICROSOFT</div>
  </div>
  <div style="margin-left:auto"><span class="live-badge">● LIVE</span></div>
</div>
""", unsafe_allow_html=True)

# ── Quick-fire sample questions ──
if not st.session_state.chat_history:
    st.markdown('<div style="font-family:Share Tech Mono;font-size:0.65rem;color:#3a5a7a;letter-spacing:3px;margin-bottom:8px">QUICK QUERIES ↓</div>', unsafe_allow_html=True)
    sample_qs = [
        "What is Amazon's cash position at end of 2024?",
        "Compare cloud revenue for all three companies",
        "Create a bar chart of total revenue for each company",
        "What are the biggest risk factors for Google in China?",
        "Which company has the highest net income margin?",
        "Compare R&D spending across all three companies",
    ]
    cols = st.columns(3)
    for i, q in enumerate(sample_qs):
        with cols[i % 3]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state._pending_query = q

# ── Chat history display ──
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown(f'<div class="source-section">📎 Sources: {msg["sources"]}</div>', unsafe_allow_html=True)
            # Expandable source chunks
            if msg.get("source_chunks"):
                with st.expander(f"🔍 View {len(msg['source_chunks'])} source chunks used for this answer"):
                    for i, chunk in enumerate(msg["source_chunks"]):
                        company = chunk.get("source", "Unknown")
                        text = chunk.get("text", "")
                        st.markdown(f"""
<div style="background:#0a1428;border:2px solid #2a5a9e;border-radius:8px;padding:14px 18px;margin:10px 0;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#5dd5ff;letter-spacing:2px;margin-bottom:8px;">
    CHUNK {i+1} · 📄 {company}
  </div>
  <div style="color:#c8d6e5;font-size:15px;line-height:1.6;">{text}</div>
</div>""", unsafe_allow_html=True)
            if msg.get("chart_data"):
                render_auto_chart(msg["chart_data"]["answer"], msg["chart_data"]["type"], msg["chart_data"]["query"])

# ── Input ──
st.divider()
col_input, col_send = st.columns([8, 1])
with col_input:
    user_input = st.text_input(
        "Ask FinBot",
        placeholder="e.g. Compare operating income margins across all three companies…",
        label_visibility="collapsed",
        key="main_input",
    )
with col_send:
    send_clicked = st.button("➤", use_container_width=True)

# Handle quick-query injection
pending = st.session_state.pop("_pending_query", None)
query = pending or (user_input if send_clicked else None)

if query:
    if not st.session_state.docs_loaded:
        st.warning("⚠️ Please upload and process your 10-K PDF files first using the sidebar.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        chart_type = detect_chart_request(query)

        with st.spinner(""):
            st.markdown('<div class="msg-thinking">⬛ FINBot IS THINKING ...</div>', unsafe_allow_html=True)
            t0 = time.time()

            # Build history string
            history_str = ""
            for h in st.session_state.chat_history[-6:]:
                role = "User" if h["role"] == "user" else "FinBot"
                history_str += f"{role}: {h['content']}\n"

            qa = st.session_state.qa_chain
            source_docs = qa["retriever"].invoke(query)
            answer = qa["chain"].invoke({
                "question": query,
                "chat_history": history_str,
            })
            if not answer:
                answer = "I could not find an answer in the provided documents."
            elapsed = time.time() - t0

        # Update stats
        n = st.session_state.total_queries
        st.session_state.avg_response_time = (st.session_state.avg_response_time * n + elapsed) / (n + 1)
        st.session_state.total_queries += 1

        sources_html = format_sources(source_docs)

        # Store the actual chunk text for the expandable view
        source_chunks = []
        for doc in source_docs:
            source_chunks.append({
                "source": doc.metadata.get("source", "Unknown"),
                "text": doc.page_content.strip()
            })

        msg_data = {
            "role": "assistant",
            "content": answer,
            "sources": sources_html,
            "source_chunks": source_chunks,
            "time": datetime.now().strftime("%H:%M:%S"),
        }

        if chart_type:
            msg_data["chart_data"] = {
                "answer": answer,
                "type": chart_type,
                "query": query,
            }

        st.session_state.chat_history.append(msg_data)
        st.rerun()

# ── Footer ──
st.markdown("""
<div style="text-align:center;margin-top:40px;font-family:Share Tech Mono;font-size:0.9rem;color:#FFFFF;letter-spacing:3px">
Fahda Alajmi · FINBOT · NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
