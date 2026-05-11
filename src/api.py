from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
        embedding = await asyncio.to_thread(get_embeddings, [request.question])
        input_quest_embedding = embedding.cpu().numpy()
        scores, samples = search_similar(input_quest_embedding, k=request.top_k)

        results = await asyncio.gather(*[
            asyncio.to_thread(generate_answer, context, request.question)
            for context in samples['context']
        ])
        results = [QAItem(answer=answer, context=context)
                    for answer, context in zip(results, samples['context'])]

        return ChatResponse(question=request.question, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))