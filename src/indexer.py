"""
Build and persist a ChromaDB vector store from TriviaQA passages.
"""
from __future__ import annotations
import os
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import get_embeddings
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL


def build_index(
    passage_records: list[tuple[str, str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Index passages into ChromaDB.

    Args:
        passage_records: list of (passage_id, title, text)
        chunk_size: token window for splitting long passages
        chunk_overlap: overlap between chunks
        force_rebuild: wipe existing index and rebuild

    Returns:
        Chroma vector store instance
    """
    embeddings = get_embeddings()

    if force_rebuild and os.path.exists(CHROMA_DIR):
        print("Removing existing ChromaDB index …")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        print("Loading existing ChromaDB index …")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION,
        )


    print(f"Building ChromaDB index from {len(passage_records)} passages (OpenAI {EMBEDDING_MODEL}) …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = []
    for pid, title, text in passage_records:
        chunks = splitter.split_text(text)
        for ci, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"passage_id": pid, "title": title, "chunk_idx": ci},
            ))

    print(f"Indexing {len(docs)} chunks into ChromaDB …")
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )
    print("Indexing complete.")
    return vs


def load_index() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )
