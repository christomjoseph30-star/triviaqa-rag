"""
Jina v5 embedding wrapper compatible with LangChain / ChromaDB.
Uses task-specific prompts for retrieval (query vs passage).
"""
from __future__ import annotations
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Embedding model loaded.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            task="retrieval",
            prompt_name="document",
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


def get_embeddings() -> JinaEmbeddings:
    return JinaEmbeddings()
