"""
Answer generation — supports two backends:
  "openai" : OpenAI API via langchain-openai (gpt-4.1, gpt-5.4-nano, etc.)
  "local"  : Qwen3-4B-Q4_K_M.gguf via llama-cpp-python (CPU)

Switch with LLM_BACKEND in config.py.
"""
from __future__ import annotations
import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import (
    LLM_BACKEND, TEMPERATURE, MAX_TOKENS, MAX_CONTEXT_PASSAGES,
    OPENAI_API_KEY, OPENAI_MODEL,
    MODEL_PATH, N_CTX, N_GPU_LAYERS, REPEAT_PENALTY, STOP_TOKENS,
)

_llm = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        print(f"Loading OpenAI LLM: {OPENAI_MODEL} …")
        _llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    else:
        from langchain_community.llms import LlamaCpp
        print(f"Loading local LLM: {MODEL_PATH} …")
        _llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            repeat_penalty=REPEAT_PENALTY,
            stop=STOP_TOKENS,
            verbose=False,
        )
    print("LLM loaded.")
    return _llm


PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a trivia question answering assistant. Answer using ONLY the provided context.

Rules:
- Answer with the shortest possible phrase — a name, place, date, or single fact
- Do NOT write a full sentence
- Do NOT explain or add context

Examples:
Q: What was Michael Jackson's autobiography?  → Moonwalk
Q: Which volcano in Tanzania is Africa's highest?  → Kilimanjaro
Q: Who directed Stagecoach?  → John Ford

Context:
{context}

Question: {question}
Answer:"""
)


def format_context(chunks: list[dict], max_passages: int = MAX_CONTEXT_PASSAGES) -> str:
    parts = []
    for i, chunk in enumerate(chunks[:max_passages], 1):
        title = chunk.get("title", "")
        text  = chunk.get("page_content", "")
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = format_context(retrieved_chunks)
    chain = PROMPT_TEMPLATE | get_llm() | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})
    # Strip Qwen3 thinking block if present (local model)
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return answer


def generate_all(
    samples: list[dict],
    retrieved_per_sample: list[list[dict]],
) -> list[str]:
    from tqdm import tqdm

    answers = []
    for s, chunks in tqdm(
        zip(samples, retrieved_per_sample),
        total=len(samples),
        desc="Generating answers",
    ):
        ans = generate_answer(s["question"], chunks)
        answers.append(ans)
    return answers
