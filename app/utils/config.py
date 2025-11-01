from pathlib import Path

# ---------- Paths ----------
# Repo root (…/PodPredictionRAG)
BASE_DIR: Path = Path(__file__).resolve().parents[1]

# Knowledge base folder (files you upload or ship with the app)
KB_FOLDER: Path = BASE_DIR / "knowledge_base"

# FAISS index folder (created automatically)
FAISS_INDEX_PATH: Path = BASE_DIR / "vectorstore"


# RAG / Embeddings
# app/utils/config.py
EMBED_MODEL: str = "text-embedding-3-small"   # or "text-embedding-3-large"



# Chunking for document splitting
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# ---------- Forecasting guardrails ----------
# Keep per-pod utilization under this (0.75 = 75%)
MAX_UTIL_RATIO: float = 0.75
