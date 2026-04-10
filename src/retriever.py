"""
Retrieval: given a question, fetch top-k passages and compute Recall@k.
"""
from __future__ import annotations
from langchain_chroma import Chroma
from config import TOP_K


def retrieve(question: str, vector_store: Chroma, k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a question.

    Returns list of dicts with keys: page_content, title, passage_id, score.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(question)
    results = []
    for doc in docs:
        results.append({
            "page_content": doc.page_content,
            "title"       : doc.metadata.get("title", ""),
            "passage_id"  : doc.metadata.get("passage_id", ""),
        })
    return results


def recall_at_k(
    retrieved: list[dict],
    answer: str,
    aliases: list[str],
    k: int = TOP_K,
) -> bool:
    """
    Return True if any of the top-k retrieved chunks contains the answer
    (or one of its aliases), case-insensitive substring match.
    """
    all_answers = [answer] + (aliases or [])
    for chunk in retrieved[:k]:
        text_lower = chunk["page_content"].lower()
        for ans in all_answers:
            if ans.lower() in text_lower:
                return True
    return False


def evaluate_retrieval(
    samples: list[dict],
    vector_store: Chroma,
    k: int = TOP_K,
) -> tuple[list[list[dict]], float]:
    """
    Run retrieval for all samples and return:
      - retrieved_per_sample: list of retrieved chunk lists
      - recall_at_k_score: fraction of questions where answer was found in top-k
    """
    from tqdm import tqdm

    retrieved_per_sample = []
    hits = 0

    for s in tqdm(samples, desc=f"Retrieving (k={k})"):
        chunks = retrieve(s["question"], vector_store, k=k)
        retrieved_per_sample.append(chunks)
        if recall_at_k(chunks, s["answer"], s.get("aliases", []), k=k):
            hits += 1

    score = hits / len(samples) if samples else 0.0
    print(f"Recall@{k}: {score:.3f}  ({hits}/{len(samples)})")
    return retrieved_per_sample, score
