"""
Load 50 questions from TriviaQA RC-Wikipedia and extract their associated passages.
"""
from __future__ import annotations
from datasets import load_dataset
from config import DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, NUM_QUESTIONS


def load_triviaqa(num_questions: int = NUM_QUESTIONS) -> list[dict]:
    """
    Returns a list of dicts, each with:
      - question      : str
      - answer        : str  (canonical answer value)
      - aliases       : list[str]  (all valid answer strings)
      - passages      : list[str]  (Wikipedia passage chunks for this question)
      - passage_titles: list[str]
    """
    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}) …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    samples = []
    for item in ds:
        if len(samples) >= num_questions:
            break

        wiki = item.get("entity_pages", {})
        contexts = wiki.get("wiki_context", []) or []
        titles   = wiki.get("title", []) or []

        # skip questions with no Wikipedia context
        if not contexts:
            continue

        answer_dict = item.get("answer", {})
        samples.append({
            "question"       : item["question"],
            "answer"         : answer_dict.get("value", ""),
            "aliases"        : answer_dict.get("aliases", []),
            "normalized_aliases": answer_dict.get("normalized_aliases", []),
            "passages"       : contexts,
            "passage_titles" : titles,
        })

    print(f"Loaded {len(samples)} questions with Wikipedia context.")
    return samples


def collect_all_passages(samples: list[dict]) -> list[tuple[str, str, str]]:
    """
    Returns unique (passage_id, title, text) tuples across all samples.
    passage_id = f"{question_index}_{passage_index}"
    """
    seen, records = set(), []
    for qi, s in enumerate(samples):
        for pi, (title, text) in enumerate(zip(s["passage_titles"], s["passages"])):
            key = title + text[:100]   # deduplicate by title + start
            if key not in seen:
                seen.add(key)
                records.append((f"q{qi}_p{pi}", title, text))
    print(f"Total unique passages: {len(records)}")
    return records
