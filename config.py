import os

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-nano"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "triviaqa_passages"

# ── LLM backend: "openai" or "local" ──────────────────────────────────────────
LLM_BACKEND  = "local"

# ── LLM (OpenAI) ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4.1"

# ── LLM (Local Qwen3-4B GGUF via llama-cpp) ───────────────────────────────────
MODEL_PATH     = r"C:\Users\MY PC\Desktop\document-rag\models\Qwen3-4B-Q4_K_M.gguf"
N_CTX          = 4096
N_GPU_LAYERS   = 0
REPEAT_PENALTY = 1.3
STOP_TOKENS    = ["Question:", "\nQuestion", "\nContext", "<|im_end|>", "<|end|>", "<|EOT|>"]

# ── LLM shared ────────────────────────────────────────────────────────────────
TEMPERATURE  = 0.1
MAX_TOKENS   = 256

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 5                  # passages retrieved per question
MAX_CONTEXT_PASSAGES = 3   # passages passed to LLM

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "trivia_qa"
DATASET_CONFIG = "rc.wikipedia"
DATASET_SPLIT  = "validation"
NUM_QUESTIONS  = 50

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
