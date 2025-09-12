# chatbot_fintech_rag_interactive.py

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

# -------- CONFIGURAÇÃO STREAMLIT --------
st.set_page_config(page_title="💳 Chatbot Fintech RAG Interativo", layout="wide")
st.title("💳 Chatbot Fintech RAG Interativo")
st.markdown("Faça perguntas sobre produtos, investimentos e serviços da fintech. Você também pode enviar PDFs em tempo real!")

# -------- HISTÓRICO DE CONVERSA --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------- FUNÇÃO PARA PROCESSAR PDF --------
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)
    return docs_split

# -------- FUNÇÃO PARA ATUALIZAR VETORSTORE --------
def update_vectorstore(new_docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(new_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(new_docs)

# -------- UPLOAD DE PDF --------
uploaded_files = st.file_uploader("Envie PDFs da fintech", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        new_docs = process_pdf(file)
        update_vectorstore(new_docs)
    st.success("PDF(s) processado(s) e índice atualizado!")

# -------- CONFIGURAÇÃO DO RAG --------
def get_qa_chain():
    if st.session_state.vectorstore is None:
        return None
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

qa_chain = get_qa_chain()

# -------- INPUT DO USUÁRIO --------
user_input = st.text_input("Digite sua pergunta sobre a fintech:")

if user_input and qa_chain:
    result = qa_chain({"query": user_input})
    answer = result['result']
    source_docs = result['source_documents']

    # Adiciona ao histórico
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": answer,
        "sources": [doc.metadata.get("source", "Documento") for doc in source_docs]
    })

# -------- EXIBIÇÃO DO HISTÓRICO --------
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin-bottom:5px;'><b>Você:</b> {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin-bottom:5px;'><b>Bot:</b> {chat['bot']}<br><i>Fonte(s): {', '.join(chat['sources'])}</i></div>", unsafe_allow_html=True)
