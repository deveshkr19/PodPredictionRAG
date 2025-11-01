from __future__ import annotations
from pathlib import Path

from .config import EMBED_MODEL
from .rag_vectorstore import (
    load_documents,
    split_into_chunks,
    create_faiss_index,
    load_faiss_index,
    search,
)


def ensure_index(kb_folder: Path, index_path: Path):
    """
    If the FAISS index does not exist but KB has docs, build it.
    Returns a tuple (index, meta) or (None, None).
    """
    index, meta = load_faiss_index(index_path)
    if index is not None:
        return (index, meta)

    docs = load_documents(kb_folder)
    if not docs:
        return (None, None)

    chunks = split_into_chunks(docs)
    create_faiss_index(chunks, index_path, model=EMBED_MODEL)
    return load_faiss_index(index_path)


def ask(db_tuple, question: str, k: int = 6) -> str:
    """
    db_tuple is (index, meta). Retrieve top-k and build a context string with [filename] prefixes.
    """
    index, meta = db_tuple
    if index is None or meta is None:
        return ""
    hits = search(index, meta, question, model=meta["model"], k=k)
    parts = [f"[{src}] {text}" for (src, text, _score) in hits]
    return "\n\n".join(parts)
