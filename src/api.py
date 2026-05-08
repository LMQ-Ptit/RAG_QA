from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio

from .embedding import get_embeddings
from .retriever import search_similar
from .qa_model import generate_answer


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

class QAItem(BaseModel):
    answer: str
    context: str

class ChatResponse(BaseModel):
    question: str
    results: list[QAItem]


app = FastAPI(title="SQuAD RAG API")


@app.post("/ask", response_model=ChatResponse)
async def ask_bot(request: ChatRequest):
    try:
        input_quest_embedding = get_embeddings([request.question]).cpu().detach().numpy()
        scores, samples = search_similar(input_quest_embedding, k=request.top_k)

        results = []
        for context in samples['context']:
            answer = generate_answer(context, request.question)
            results.append(QAItem(answer=answer, context=context))

        return ChatResponse(question=request.question, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))