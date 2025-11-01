from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def _embeddings():
    # Uses OpenAI embeddings via your OPENAI_API_KEY (Streamlit Secrets)
    return OpenAIEmbeddings(model=EMBED_MODEL)


def load_documents(kb_folder: Path):
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
    return FAISS.load_local(
        str(index_path),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
