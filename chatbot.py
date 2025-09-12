import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Fintech domain knowledge (pre-loaded manual)
# -----------------------------
FINTECH_MANUAL = """
Introduction:
Welcome to our Fintech! We are committed to revolutionizing the way you manage your money.
Our platform provides innovative credit card solutions and investment products designed to give
you financial freedom and growth opportunities.

Credit Card Products:
Flexible payment options, cashback rewards, advanced security features, mobile app management, 
spending limits, and real-time transaction tracking.

Investment Products:
Mutual funds, ETFs, stocks, fixed income products, AI-driven advisory tool, portfolio tailored 
to financial goals and risk profile.

Getting Started:
Download the app, create your account, complete identity verification, apply for credit card 
or start investing, monitor your wealth via dashboard.

Security & Compliance:
Multi-factor authentication, data encryption, fraud detection systems, comply with international 
financial regulations.

Customer Support:
24/7 support via chat, phone, email, knowledge base with guides and FAQs.

Benefits Comparison:
Credit Card: cashback & points, installments & limits, fraud protection.
Investments: dividend & capital gains, diversified portfolio, regulated market.
"""

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Fintech Chatbot", page_icon="💳", layout="wide")
st.title("💳 Fintech Specialized RAG Agent (Groq + Streamlit)")

# -----------------------------
# Text splitting & embeddings
# -----------------------------
# Split fintech manual into chunks for semantic search
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(FINTECH_MANUAL)

# HuggingFace embeddings (lightweight, works in Cloud)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore from fintech chunks
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------------
# Groq LLaMA 3 LLM setup
# -----------------------------
# You need to set your GROQ_API_KEY in Streamlit Cloud secrets
# (Settings > Secrets > add GROQ_API_KEY)
llm = ChatGroq(model="llama3-8b-8192", temperature=0.3)

# -----------------------------
# Memory for conversation
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Prompt engineering
# -----------------------------
def create_prompt(user_query: str) -> str:
    return f"""
You are a helpful and specialized assistant for our Fintech company.
You must use the given context to answer user questions about our credit card and investment products.
Always explain clearly, concisely, and with professional tone.
User question: {user_query}
Answer:
"""

# -----------------------------
# Conversational RAG chain
# -----------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": create_prompt("")}
)

# -----------------------------
# User interaction
# -----------------------------
user_input = st.text_input("Ask your question about our Fintech products:")

if user_input:
    response = qa_chain.run(user_input)
    st.markdown(f"**Answer:** {response}")
