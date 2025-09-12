# type: ignore
import os
import logging
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain import hub

load_dotenv()

# Fintech adapted RAG prompt
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")

# Load text from PDFs
def load_documents(file_paths):
    all_text = []
    for file in file_paths:
        reader = PdfReader(file)
        text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        all_text.append(text)
    return "\n\n".join(all_text)

# Split into chunks
def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=300,
    )
    return text_splitter.split_text(text)

# Create embeddings and Chroma vectorstore
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=".chromadb"  # persisted locally
    )
    return vectorstore

# Format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
def rag_chain(vectorstore, question):
    qa_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)

# Save temp file
def _get_file_path(file_upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_upload.name)
    with open(file_path, "wb") as f:
        f.write(file_upload.getbuffer())
    return file_path

# Main app
def main():
    st.set_page_config(page_title="💳 Fintech Knowledge Chatbot", layout="wide")
    st.title("💳 Fintech Knowledge Chatbot")
    st.markdown(
        "Welcome! Ask questions about cards, investments, and other fintech services.\n\n"
        "📂 You can also upload **PDFs** with domain knowledge."
    )
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your fintech assistant. "
                    "Upload documents and ask me anything about financial products."
                )
            }
        ]

    # File uploader
    file_upload = st.sidebar.file_uploader(
        label="Upload Fintech Documents (PDF only for now)",
        type=["pdf"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if file_upload:     
        st.success("File(s) uploaded successfully! You can now ask your question.")

    # Display messages
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
                    chunked_text = split_text(text)
                    vectorstore = get_vectorstore(chunked_text)
                    assistant_reply = rag_chain(vectorstore, user_prompt)
                else:
                    assistant_reply = (
                        "Please upload at least one fintech document "
                        "so I can answer based on real content."
                    )

                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                st.markdown(assistant_reply)

if __name__ == '__main__':
    main()


