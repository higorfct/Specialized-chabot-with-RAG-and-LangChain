import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Fintech manual content (pre-loaded)
# -----------------------------
# This is the domain knowledge of our chatbot.
# It includes credit card info, investment products, security, and support.
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
# Streamlit UI setup
# -----------------------------
# Configure the web app page and display title
st.set_page_config(page_title="Fintech Chatbot", page_icon="💳", layout="wide")
st.title("💳 Fintech Specialized RAG Agent (Ready-to-Use)")

# -----------------------------
# Split the manual into chunks
# -----------------------------
# Text splitter divides large text into smaller pieces for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(FINTECH_MANUAL)

# -----------------------------
# Initialize embeddings and LLaMA LLM
# -----------------------------
# Provide the path to your LLaMA model file
llama_model_path = "path_to_your_llama_model/llama-13b-4bit.gguf"  # replace with your model path

# Create embeddings model (converts text into vector representations)
embeddings_model = LlamaCppEmbeddings(model_path=llama_model_path)

# Create vectorstore using FAISS (stores embeddings and enables semantic search)
vectorstore = FAISS.from_texts(chunks, embeddings_model)

# Create a retriever to search within the vectorstore
retriever = vectorstore.as_retriever()

# Initialize the LLaMA language model for generation
llm = LlamaCpp(model_path=llama_model_path, n_ctx=2048, temperature=0.3)

# -----------------------------
# Conversation memory
# -----------------------------
# Stores chat history to provide context-aware responses
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Prompt engineering function
# -----------------------------
# Creates a custom prompt for the LLaMA model
# Instructs the model to answer clearly and adaptively based on the question
def create_prompt(user_query):
    return f"""
You are an expert assistant specialized in our Fintech products.
Use the provided information to answer the user's query accurately and clearly.
Provide concise but informative answers, adapting style based on the question.
User question: {user_query}
Answer:
"""

# -----------------------------
# Create the RAG chain
# -----------------------------
# Combines retrieval of relevant chunks and generation of answers with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": create_prompt("")}  # initial prompt template
)

# -----------------------------
# User input and interaction
# -----------------------------
# Streamlit text input for user questions
user_input = st.text_input("Ask your question about our Fintech products:")

# Process the user's question
if user_input:
    # Run the RAG chain to generate an answer based on internal manual
    response = qa_chain.run(user_input)
    
    # Display the answer in Streamlit
    st.markdown(f"**Answer:** {response}")
