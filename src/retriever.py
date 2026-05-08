import os
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import hf_hub_download

load_dotenv()

DATASET_NAME = 'MinhQuy24/SQuAD_QA_Vector_Database'
INDEX_FILE = "my_index.faiss"
EMBEDDING_COLUMN = "question_embedding"
HF_TOKEN = os.getenv("HF_TOKEN") or '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'

embeddings_dataset = load_dataset(DATASET_NAME, split="train", token=HF_TOKEN)

index_file_path = hf_hub_download(
    repo_id=DATASET_NAME, filename=INDEX_FILE, repo_type="dataset", token=HF_TOKEN
)
embeddings_dataset.load_faiss_index(EMBEDDING_COLUMN, index_file_path)


def search_similar(question_embedding, top_k=3):
    scores, samples = embeddings_dataset.get_nearest_examples(
        EMBEDDING_COLUMN, question_embedding, k=top_k
    )
    return scores, samples