from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def _embeddings():
    # Normalize vectors; CPU works fine on Streamlit Cloud
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True, "device": "cpu"},
    )


def load_documents(kb_folder: Path):
    """Load .txt/.md/.csv from kb_folder and ensure 'source' metadata is the filename."""
    kb_folder = Path(kb_folder)
    kb_folder.mkdir(parents=True, exist_ok=True)

    docs = []
    for p in kb_folder.glob("*"):
        suf = p.suffix.lower()
        if suf in {".txt", ".md"}:
            items = TextLoader(str(p), autodetect_encoding=True).load()
        elif suf == ".csv":
            items = CSVLoader(str(p)).load()
        else:
            continue

        # Normalize metadata so citations look like [filename.ext]
        for d in items:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p.name
        docs.extend(items)

    return docs


def split_into_chunks(documents) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def create_faiss_index(chunks, index_path: Path):
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, _embeddings())
    db.save_local(str(index_path))


def load_faiss_index(index_path: Path):
    # allow_dangerous_deserialization handles minor version differences safely
    return FAISS.load_local(
        str(index_path),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
