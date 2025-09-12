# type: ignore
import os
import logging
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# Free LLM from HuggingFace Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # lightweight, free model
    model_kwargs={"temperature": 0.6, "max_length": 512}
)

# Free embeddings from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Fintech custom prompt
prompt_template = """
You are a financial assistant for a fintech company.
Use the following context to answer the question at the end.

Context:
{context}

Question: {question}

Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load PDFs
def load_documents(file_paths):
    all_text = []
    for file in file_paths:
        reader = PdfReader(file)
        text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        all_text.append(text)
    return "\n\n".join(all_text)

# Split text into chunks
def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    return text_splitter.split_text(text)

# Create vectorstore with Chroma
def get_vectorstore(chunks):
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=".chromadb"
    )
    return vectorstore

# RAG chain
def rag_chain(vectorstore, question):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa.run(question)

# Temp file
def _get_file_path(file_upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_upload.name)
    with open(file_path, "wb") as f:
        f.write(file_upload.getbuffer())
    return file_path

# Main Streamlit app
def main():
    st.set_page_config(page_title="💳 Fintech Knowledge Chatbot", layout="wide")
    st.title("💳 Fintech Knowledge Chatbot (Free HuggingFace Version)")
    st.markdown(
        "Welcome! Ask me about fintech services (cards, investments, loans, etc). "
        "📂 Upload **PDFs** with company knowledge to improve answers."
    )
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your fintech assistant. Upload documents and ask me anything."}
        ]

    # File uploader
    file_upload = st.sidebar.file_uploader(
        label="Upload Fintech PDFs",
        type=["pdf"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if file_upload:     
        st.success("File(s) uploaded successfully! You can now ask your question.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    user_prompt = st.chat_input("Enter your question about the fintech")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            logging.info("Generating response...")
            with st.spinner("Processing..."): 
                if file_upload:
                    file_paths = [_get_file_path(f) for f in file_upload]
                    text = load_documents(file_paths)
                    chunks = split_text(text)
                    vectorstore = get_vectorstore(chunks)
                    assistant_reply = rag_chain(vectorstore, user_prompt)
                else:
                    assistant_reply = "Please upload at least one fintech document."

                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                st.markdown(assistant_reply)

if __name__ == '__main__':
    main()
