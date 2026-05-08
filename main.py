import os
from dotenv import load_dotenv

load_dotenv()

from src.embedding import get_embeddings
from src.retriever import search_similar, embeddings_dataset
from src.qa_model import generate_answer, model_QA, tokenizer_QA
from src.api import app


def run_api():
    import asyncio
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn

    nest_asyncio.apply()

    ngrok_token = os.getenv("NGROK_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    public_url = ngrok.connect(port)
    print("=" * 50)
    print(f"🚀 API ONLINE: {public_url.public_url}")
    print(f"📖 Docs: {public_url.public_url}/docs")
    print("=" * 50)

    config = uvicorn.Config(app, host=host, port=port, loop="asyncio")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())


def run_ui():
    from src.ui import launch_ui
    launch_ui(share=True)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "api":
        run_api()
    elif mode == "ui":
        run_ui()
    else:
        print("🚀 Starting RAG-QA System...")
        run_api()
        print("\n🎨 Starting UI...")
        run_ui()