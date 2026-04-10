"""
Full RAG evaluation pipeline:
  1. Load 50 TriviaQA RC-Wikipedia questions
  2. Index all associated passages in ChromaDB (Qwen3-Embedding)
  3. Retrieve top-k passages per question
  4. Generate answers with Qwen3-4B-Q4_K_M.gguf
  5. Evaluate retrieval (Recall@k) and answer quality (EM, Token F1)
"""
import sys
import os

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader  import load_triviaqa, collect_all_passages
from src.indexer      import build_index
from src.retriever    import evaluate_retrieval
from src.generator    import generate_all
from src.evaluator    import evaluate_answers, print_failure_cases
from config           import TOP_K, NUM_QUESTIONS


def run(force_rebuild: bool = False):
    # 1. Load dataset
    samples = load_triviaqa(NUM_QUESTIONS)

    # 2. Build / load ChromaDB index
    passage_records = collect_all_passages(samples)
    vector_store    = build_index(passage_records, force_rebuild=force_rebuild)

    # 3. Retrieve
    retrieved_per_sample, recall = evaluate_retrieval(samples, vector_store, k=TOP_K)

    # 4. Generate
    generated_answers = generate_all(samples, retrieved_per_sample)

    # 5. Evaluate
    metrics = evaluate_answers(samples, retrieved_per_sample, generated_answers, k=TOP_K)
    print_failure_cases(samples, retrieved_per_sample, generated_answers, k=TOP_K)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the ChromaDB index even if it exists")
    args = parser.parse_args()
    run(force_rebuild=args.rebuild)
