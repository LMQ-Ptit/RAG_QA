# RAG-QA System

A Question-Answering system using RAG (Retrieval-Augmented Generation) combining FAISS and Llama 3.2.

## Project Structure

```
RAG_QA/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── .env                 # Tokens & config
├── .gitignore
└── src/
    ├── embedding.py     # DistilBERT - embed questions
    ├── retriever.py     # FAISS - retrieve context
    ├── qa_model.py      # Llama 3.2 - generate answers
    ├── api.py           # FastAPI endpoint
    └── ui.py            # Gradio UI
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit .env (add your tokens)
cp .env .env.local
```

## Running

```bash
# Run both API + UI
python main.py

# API only
python main.py api

# UI only
python main.py ui
```

## API

```bash
# Health check
GET /health

# Ask question
POST /ask
Body: {"question": "...", "top_k": 3}
```

## Quick Test

Run `notebooks/test.ipynb` to quickly test each component:

1. **Test Embedding** - Verify DistilBERT embeddings
2. **Test Retrieval** - Search context with FAISS
3. **Test QA Model** - Generate answers
4. **Test Full Pipeline** - Complete RAG pipeline

## Test Interface

### Main UI
![Main Interface](./images/image.png)


## Configuration (.env)

```env
NGROK_TOKEN=your_ngrok_token
HF_TOKEN=your_hf_token
API_PORT=8000
UI_PORT=7860
```