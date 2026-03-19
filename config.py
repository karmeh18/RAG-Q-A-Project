"""
config.py
---------
Central configuration for the RAG pipeline.
Change settings here — no need to touch any other file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-3.5-turbo")   # or "gpt-4o"
LLM_TEMPERATURE = 0     # 0 = deterministic, factual answers
MAX_TOKENS      = 512   # max tokens in the LLM response

# ── Embedding model ───────────────────────────────────────────────────────────
# Runs locally, no API key needed.
# all-MiniLM-L6-v2: fast, lightweight (~80MB), 384-dim vectors
# all-mpnet-base-v2: slower, more accurate, 768-dim vectors
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Document chunking ─────────────────────────────────────────────────────────
# CHUNK_SIZE:    max characters per chunk
# CHUNK_OVERLAP: characters shared between adjacent chunks (preserves context at boundaries)
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP",  50))

# ── Retrieval ─────────────────────────────────────────────────────────────────
# How many chunks to retrieve per query.
# More = richer context, but costs more tokens and may add noise.
TOP_K = int(os.getenv("TOP_K", 4))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"   # put your PDFs / TXTs / DOCXs here
DB_DIR    = "db"     # FAISS index saved here after first run

# ── Prompt strategy ───────────────────────────────────────────────────────────
# Options: "zero_shot" | "few_shot" | "chain_of_thought"
PROMPT_STRATEGY = os.getenv("PROMPT_STRATEGY", "zero_shot")
