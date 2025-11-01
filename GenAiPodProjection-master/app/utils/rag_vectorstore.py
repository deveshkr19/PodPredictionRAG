import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith((".txt", ".csv", ".md")):
            loader = TextLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

def split_into_chunks(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_faiss_index(chunks, save_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)
    return db

def load_faiss_index(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
    )
    return FAISS.load_local(path, embeddings)

def retrieve_context(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])
