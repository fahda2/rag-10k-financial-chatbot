import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time


persona = '''
You are an expert financial analyst specializing in Big Tech companies. 
You have deep knowledge of annual reports (10-K filings) for Alphabet, Amazon, and Microsoft.

Rules:
- ONLY answer using information from the provided PDF documents
- If the answer is not in the documents, say "I cannot find that in the provided 10-K files"
- Always cite which company and which section your answer comes from
- Be precise with numbers — always include units (millions, billions, %)
- For comparisons across companies, structure your answer clearly by company.
Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
'''

# Set Google API key
GOOGLE_API_KEY = 'AIzaSyBURX-GRYfbNKJ4ASe5NrH5Y1ZWwatgV2M'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="💼 10-K Financial Analyst Chatbot", page_icon="📊")

st.title("📊💬 10-K Financial Analyst Chatbot")
st.caption("Analyzing Alphabet, Amazon & Microsoft annual reports")

# File uploader
uploaded_files = st.file_uploader("Upload 10-K PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []

    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs... this may take a minute ⏳"):
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)

                # Generate embeddings and store in FAISS
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDFs processed! Start chatting below.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about the 10-K filings...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
        )

        # Build chat history string
        chat_history = ""
        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"

        # Build prompt
        prompt_template = PromptTemplate.from_template("""
{persona}

Chat History:
{chat_history}

Use the following context from the 10-K documents to answer the question.
Context:
{context}

Question: {question}

Answer:""")

        # Helper to format retrieved docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Modern LCEL chain replacing RetrievalQA
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

        with st.spinner("Thinking... 🤔"):
            # Also retrieve source docs for display
            source_docs = retriever.invoke(user_input)
            response_text = chain.invoke(user_input)

        # Show source chunks
        with st.expander("📄 View Retrieved Chunks (Context)"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Chunk {i+1}**")
                st.markdown(f"**Content:** {doc.page_content}")
                st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                st.markdown("---")

        # Stream response word by word
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.03)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("👆 Please upload your 10-K PDF files to begin.")
    st.markdown("""
    **Suggested questions to try:**
    - How much cash does Amazon have at the end of 2024?
    - What are the main revenue sources for Google vs Amazon vs Microsoft?
    - Do these companies mention risks related to China or India in cloud services?
    - What is Microsoft Azure's revenue growth compared to Google Cloud?
    """)