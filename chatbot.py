import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Fintech Manual (pre-loaded)
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
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fintech Chatbot", page_icon="💳", layout="wide")
st.title("💳 Fintech Specialized RAG Agent (Hugging Face API)")

# -----------------------------
# Split text into chunks for retrieval
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(FINTECH_MANUAL)

# Embeddings (HuggingFace lightweight model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS vectorstore for RAG
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------------
# Hugging Face API LLM setup
# -----------------------------
# Set your HF_API_KEY in Streamlit secrets or environment variable
# Go to https://huggingface.co/settings/tokens to create a free token
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",  # lightweight free model
    model_kwargs={"temperature":0.3, "max_new_tokens":256},
    huggingfacehub_api_token=st.secrets["HF_API_KEY"]  # get from Streamlit secrets
)

# -----------------------------
# Conversation memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Prompt engineering
# -----------------------------
def create_prompt(user_query: str) -> str:
    return f"""
You are a specialized Fintech assistant.
Answer questions based only on the provided Fintech information.
Explain clearly, concisely, and professionally.
User question: {user_query}
Answer:
"""

# -----------------------------
# Conversational Retrieval Chain (RAG)
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

