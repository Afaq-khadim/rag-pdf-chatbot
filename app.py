# ============================================================
# RAG System — Streamlit UI
# Chat with your PDF using Groq + Llama 3.3
# ============================================================

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #dcf8c6;
        text-align: right;
    }
    .bot-message {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────
st.title("📄 RAG System — Chat with your PDF")
st.markdown("Upload a PDF and ask any question about it!")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Groq API Key loaded securely from .env
    if groq_api_key:
        st.sidebar.success("Groq API key loaded successfully!")
    else:
        st.sidebar.error("Groq API key not found. Check your .env file.")
        
    st.divider()

    # PDF Upload
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    # Process button
    process_btn = st.button("🚀 Process PDF", use_container_width=True)

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("🦙 Llama 3.3 via Groq")
    st.markdown("🤗 HuggingFace Embeddings")
    st.markdown("🔍 FAISS Vector Database")
    st.markdown("⚡ LangChain + Streamlit")

# ── Initialize Session State ─────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ── Process PDF ──────────────────────────────────────────────
if process_btn:
    if not groq_api_key:
        st.sidebar.error("⚠️ Please enter your Groq API Key!")
    elif not uploaded_file:
        st.sidebar.error("⚠️ Please upload a PDF file!")
    else:
        with st.spinner("⏳ Processing your PDF... Please wait!"):
            try:
                # Read PDF
                reader = PdfReader(uploaded_file)
                raw_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(raw_text)

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Vector Store
                vectorstore = FAISS.from_texts(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # LLM
                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.5
                )

                # Prompt
                prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")
                # Format docs
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # RAG Chain
                st.session_state.rag_chain = (
                    {"context": retriever | format_docs,
                     "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                st.session_state.pdf_processed = True
                st.sidebar.success(f"✅ PDF Processed!")
                st.sidebar.info(f"📃 Pages: {len(reader.pages)}\n\n🔪 Chunks: {len(chunks)}")

            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# ── Chat Interface ───────────────────────────────────────────
if st.session_state.pdf_processed:

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Input box
    question = st.chat_input("💬 Ask anything about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
                st.write(answer)

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

else:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1️⃣**\n\nEnter your Groq API Key in the sidebar")
    with col2:
        st.info("**Step 2️⃣**\n\nUpload your PDF file in the sidebar")
    with col3:
        st.info("**Step 3️⃣**\n\nClick Process PDF and start chatting!")