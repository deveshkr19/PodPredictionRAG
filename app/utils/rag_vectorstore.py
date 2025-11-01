from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import csv
import pickle
import re

import faiss                   # type: ignore
import numpy as np
from openai import OpenAI
import streamlit as st

from .config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


# ---------- Embedding client ----------
def _client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing in Streamlit secrets.")
    return OpenAI(api_key=key)


def embed_texts(texts: List[str], model: str) -> np.ndarray:
    """
    Returns a (N, D) float32 numpy array of embeddings using OpenAI embeddings API.
    We do simple batching to stay efficient and resilient.
    """
    cli = _client()
    batch = []
    vecs = []
    BATCH_SIZE = 256
    for t in texts:
        batch.append(t)
        if len(batch) == BATCH_SIZE:
            resp = cli.embeddings.create(model=model, input=batch)
            vecs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
            batch = []
    if batch:
        resp = cli.embeddings.create(model=model, input=batch)
        vecs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(vecs).astype("float32")


# ---------- Document loading / chunking ----------
def load_documents(kb_folder: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, text).
    Supports .txt, .md, .csv (joins rows by comma).
    """
    kb_folder = Path(kb_folder)
    kb_folder.mkdir(parents=True, exist_ok=True)
    docs: List[Tuple[str, str]] = []

    for p in sorted(kb_folder.glob("*")):
        suf = p.suffix.lower()
        if suf in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append((p.name, text))
        elif suf == ".csv":
            rows = []
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(", ".join(row))
            docs.append((p.name, "\n".join(rows)))
        else:
            continue
    return docs


def split_into_chunks(docs: List[Tuple[str, str]],
                      chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, str]]:
    """
    Simple character-based splitter with overlap.
    Returns list of (source_name, chunk_text).
    """
    chunks: List[Tuple[str, str]] = []
    for src, text in docs:
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            continue
        i = 0
        L = len(t)
        while i < L:
            end = min(i + chunk_size, L)
            chunks.append((src, t[i:end]))
            if end == L:
                break
            i = end - overlap if end - overlap > i else end
    return chunks


# ---------- FAISS save/load/search ----------
def create_faiss_index(chunks: List[Tuple[str, str]], index_dir: Path, model: str):
    """
    Builds FAISS index and saves metadata.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [c[1] for c in chunks]
    sources = [c[0] for c in chunks]
    X = embed_texts(texts, model=model)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    # normalize
    faiss.normalize_L2(X)
    index.add(X)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"sources": sources, "texts": texts, "model": model}, f)


def load_faiss_index(index_dir: Path):
    index_dir = Path(index_dir)
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "meta.pkl"
    if not index_path.exists() or not meta_path.exists():
        return None, None
    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(index, meta, query: str, model: str, k: int = 6) -> List[Tuple[str, str, float]]:
    """
    Returns top-k: (source_name, text, score)
    """
    q = embed_texts([query], model=model)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    results: List[Tuple[str, str, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append((meta["sources"][idx], meta["texts"][idx], float(score)))
    return results
