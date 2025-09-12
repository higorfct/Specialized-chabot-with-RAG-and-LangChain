import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile
import os

st.set_page_config(page_title="💳 Chatbot Fintech RAG - Free API", layout="wide")
st.title("💳 Chatbot Fintech RAG Interativo (API Gratuita)")

# --- Histórico de conversa ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Função para processar PDFs ---
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    return docs_split

# --- Atualizar vetorstore ---
def update_vectorstore(new_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(new_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(new_docs)

# --- Upload PDFs ---
uploaded_files = st.file_uploader("Envie PDFs da fintech", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        new_docs = process_pdf(file)
        update_vectorstore(new_docs)
    st.success("PDF(s) processado(s) e índice atualizado!")

# --- Configuração do RAG ---
def get_qa_chain():
    if st.session_state.vectorstore is None:
        return None
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    # HuggingFaceHub LLM gratuito (ex: "google/flan-t5-small")
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":512})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

qa_chain = get_qa_chain()

# --- Input do usuário ---
user_input = st.text_input("Digite sua pergunta sobre a fintech:")

if user_input and qa_chain:
    result = qa_chain({"query": user_input})
    answer = result['result']
    source_docs = result['source_documents']

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": answer,
        "sources": [doc.metadata.get("source", "Documento") for doc in source_docs]
    })

# --- Exibir histórico ---
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin-bottom:5px;'><b>Você:</b> {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin-bottom:5px;'><b>Bot:</b> {chat['bot']}<br><i>Fonte(s): {', '.join(chat['sources'])}</i></div>", unsafe_allow_html=True)

