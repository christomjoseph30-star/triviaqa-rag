"""
Answer quality metrics: Exact Match and Token F1 (standard QA evaluation).
"""
from __future__ import annotations
import re
import string
import json
import os
from datetime import datetime
import pandas as pd
from config import RESULTS_DIR, TOP_K, EMBEDDING_MODEL, OPENAI_MODEL, LLM_BACKEND, MODEL_PATH


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred_norm = _normalize(prediction)
    return any(pred_norm == _normalize(a) for a in gold_answers)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = _normalize(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall    = len(common) / len(gold_tokens) if gold_tokens else 0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def evaluate_answers(
    samples: list[dict],
    retrieved_per_sample: list[list[dict]],
    generated_answers: list[str],
    k: int = TOP_K,
) -> dict:
    """
    Compute per-question and aggregate metrics. Saves results to CSV and JSON.
    """
    rows = []
    for s, chunks, pred in zip(samples, retrieved_per_sample, generated_answers):
        gold_answers = [s["answer"]] + s.get("aliases", [])
        retrieval_hit = any(
            any(a.lower() in c["page_content"].lower() for a in gold_answers)
            for c in chunks[:k]
        )
        em  = exact_match(pred, gold_answers)
        f1  = token_f1(pred, gold_answers)

        rows.append({
            "question"        : s["question"],
            "gold_answer"     : s["answer"],
            "predicted_answer": pred,
            "retrieval_hit"   : retrieval_hit,
            "exact_match"     : em,
            "token_f1"        : round(f1, 4),
            "top_passages"    : " | ".join(c["title"] for c in chunks[:k]),
            "retrieved_chunks": "\n---\n".join(
                f"[{i+1}] {c['title']}\n{c['page_content']}"
                for i, c in enumerate(chunks[:k])
            ),
        })

    df = pd.DataFrame(rows)

    # Aggregate
    results = {
        f"recall_at_{k}"      : round(df["retrieval_hit"].mean(), 4),
        "exact_match"         : round(df["exact_match"].mean(), 4),
        "mean_token_f1"       : round(df["token_f1"].mean(), 4),
        "num_questions"       : len(df),
        "retrieval_hits"      : int(df["retrieval_hit"].sum()),
        "em_correct"          : int(df["exact_match"].sum()),
    }

    # Build run tag: embedding_model + llm + timestamp
    emb_tag = EMBEDDING_MODEL.split("/")[-1]
    llm_tag = OPENAI_MODEL if LLM_BACKEND == "openai" else os.path.splitext(os.path.basename(MODEL_PATH))[0]
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{emb_tag}__{llm_tag}__{ts}"

    csv_path  = os.path.join(RESULTS_DIR, f"results__{run_tag}.csv")
    json_path = os.path.join(RESULTS_DIR, f"metrics__{run_tag}.json")
    df.to_csv(csv_path, index=False)
    results["embedding_model"] = EMBEDDING_MODEL
    results["llm_model"]       = llm_tag
    results["timestamp"]       = ts
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n-- Evaluation Results ----------------------------------")
    for k_, v in results.items():
        print(f"  {k_:<22}: {v}")
    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Metrics saved to         : {json_path}")
    return results


def print_failure_cases(
    samples: list[dict],
    retrieved_per_sample: list[list[dict]],
    generated_answers: list[str],
    k: int = TOP_K,
    max_cases: int = 5,
):
    """Print examples of the two main failure modes."""
    gold_answers_list = [[s["answer"]] + s.get("aliases", []) for s in samples]

    # Failure type 1: retrieval miss (answer not in top-k passages)
    retrieval_misses = [
        (s, pred)
        for s, chunks, pred in zip(samples, retrieved_per_sample, generated_answers)
        if not any(
            any(a.lower() in c["page_content"].lower() for a in [s["answer"]] + s.get("aliases", []))
            for c in chunks[:k]
        )
    ]

    # Failure type 2: retrieval hit but wrong answer
    retrieval_hit_wrong = [
        (s, chunks, pred)
        for s, chunks, pred in zip(samples, retrieved_per_sample, generated_answers)
        if (
            any(
                any(a.lower() in c["page_content"].lower() for a in [s["answer"]] + s.get("aliases", []))
                for c in chunks[:k]
            )
            and not exact_match(pred, [s["answer"]] + s.get("aliases", []))
        )
    ]

    print("\n-- Failure Cases -----------------------------------------------")
    print(f"\nType 1: Retrieval miss (answer not found in top-{k}) — {len(retrieval_misses)} cases")
    for s, pred in retrieval_misses[:max_cases]:
        print(f"  Q: {s['question']}")
        print(f"  Gold: {s['answer']}  |  Predicted: {pred[:80]}")

    print(f"\nType 2: Retrieval hit but wrong answer — {len(retrieval_hit_wrong)} cases")
    for s, chunks, pred in retrieval_hit_wrong[:max_cases]:
        print(f"  Q: {s['question']}")
        print(f"  Gold: {s['answer']}  |  Predicted: {pred[:80]}")
