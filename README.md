# RAG-QA System

Hệ thống Hỏi-Đáp sử dụng RAG (Retrieval-Augmented Generation) kết hợp FAISS và Llama 3.2.

## Cấu trúc

```
RAG_QA/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── .env                 # Tokens & config
├── .gitignore
└── src/
    ├── embedding.py     # DistilBERT - embed câu hỏi
    ├── retriever.py     # FAISS - tìm context
    ├── qa_model.py      # Llama 3.2 - generate câu trả lời
    ├── api.py           # FastAPI endpoint
    └── ui.py            # Gradio UI
```

## Setup

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Copy và edit .env (thêm tokens của bạn)
cp .env .env.local
```

## Chạy

```bash
# Chạy cả API + UI
python main.py

# Chỉ API
python main.py api

# Chỉ UI
python main.py ui
```

## API

```bash
# Health check
GET /health

# Hỏi đáp
POST /ask
Body: {"question": "...", "top_k": 3}
```

## Test nhanh

Chạy file `notebooks/test.ipynb` để test nhanh từng component:

1. **Test Embedding** - Kiểm tra DistilBERT
2. **Test Retrieval** - Tìm context với FAISS
3. **Test QA Model** - Generate câu trả lời
4. **Test Full Pipeline** - RAG hoàn chỉnh

## Config (.env)

```env
NGROK_TOKEN=your_ngrok_token
HF_TOKEN=your_hf_token
API_PORT=8000
UI_PORT=7860
```