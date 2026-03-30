# msme_bot

Conversational chatbot that helps micro and small business owners discover relevant government schemes and improve their digital and financial literacy.

## Features

- **Semantic Search**: FAISS vector index for fast, relevant scheme matching based on user queries
- **Scheme Database**: Curated collection of government schemes for MSMEs
- **Conversational Interface**: Natural language Q&A powered by LLMs
- **Streamlit UI**: Simple, accessible web interface

## Tech Stack

- Python, Streamlit
- FAISS (vector similarity search)
- OpenAI / LLM APIs
- Custom data pipeline for scheme ingestion

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Precompute FAISS index: `python precompute_faiss_index.py`
3. Run: `streamlit run app.py`
