# Specialized RAG Agent for Fintech Customer Support  

## 📌 Project Overview  
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** specialized in **customer support for fintech financial products**.  
The agent uses **LangChain**, **OpenAI GPT models**, and **FAISS vector stores** to provide accurate answers based on a reference PDF (`fintech_manual.pdf`).  

The chatbot:  
- Answers questions about **credit cards, investment products, onboarding, security, compliance, and customer support**.  
- Politely refuses to answer irrelevant questions.  
- Maintains **conversational memory** across sessions.  



**Final summarized results (artificial scenario):**
- Time saved: **~36.67 hours per day** (≈ **733.33 hours/month**, **8,800 hours/year**).
- Money saved: **$11,000 per month** → **$132,000 per year**.
- Daily operational cost: **before $625/day**, **after $75/day**.
- Monthly operational cost (20 days): **before $12,500**, **after $1,500**.

Impact note: these savings assume the stated AI resolution rate (70%), handling times, and 20 working days/month. Adjust any input (resolution rate, queries/day, minutes per query, cost/hour, working days) to recompute exact savings for your scenario.

---

## 🔒 Security Note
⚠️ Do **not** hardcode your API key in the notebook. Use environment variables or secret managers for production use.

---

## 📌 Next Steps
- Add a **web interface** with Streamlit or Gradio.
- Extend support for multiple documents.
- Enable **retrieval source citation**.


---

## ⚙️ Tech Stack  
- **Python**  
- [LangChain](https://www.langchain.com/)  
- [OpenAI API](https://platform.openai.com/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [PyPDFLoader](https://pypi.org/project/pypdf/)  

---



