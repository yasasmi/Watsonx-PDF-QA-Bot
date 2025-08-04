# Watsonx-PDF-QA-Bot

A PDF Question-Answering chatbot powered by IBM Watsonx and LangChain. Upload any PDF and ask natural-language questions — the bot will read, understand, and answer based on the document's contents.

## 🚀 Features

- 💬 Natural Language Question Answering
- 📄 PDF document ingestion
- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 IBM Watsonx LLM + Embeddings
- 🧱 LangChain for chaining and retrieval
- 🖥️ Runs locally via Gradio UI

---

## 🧰 Requirements

- Python 3.8+
- IBM Watsonx credentials
- pip packages (see below)

### 🛠️ Install Dependencies

```bash
pip install langchain langchain_ibm ibm-watsonx-ai gradio chromadb PyPDF2
