import os
from utils.rag_vectorstore import (
    load_documents,
    split_into_chunks,
    create_faiss_index,
    load_faiss_index,
    retrieve_context
)

def handle_knowledge_base(kb_files, kb_folder, index_path):
    # If user uploaded files, save to kb_folder
    if kb_files:
        os.makedirs(kb_folder, exist_ok=True)
        for file in kb_files:
            file_path = os.path.join(kb_folder, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

    # Check if FAISS index already exists
    if os.path.exists(index_path):
        try:
            db = load_faiss_index(index_path)
            return db
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")

    # Try to build index from local files if available
    docs = load_documents(kb_folder)
    if docs:
        chunks = split_into_chunks(docs)
        db = create_faiss_index(chunks, index_path)
        print(f"✅ Created FAISS index from {len(chunks)} document chunks.")
        return db
    else:
        print("⚠️ No documents found to build RAG knowledge base.")
        return None


def handle_user_question(db, question, df, api_key):
    if not db:
        return "⚠️ No RAG documents available. Please upload a knowledge base.", None

    context = retrieve_context(db, question)

    # Add a sample data row from the CSV to help GPT
    try:
        sample_row = df.sample(1).to_dict(orient="records")[0]
        context += "\n\nExample Data Row:\n" + str(sample_row)
    except Exception:
        pass  # Just in case df is empty or has issues

    prompt = f"""You are a performance engineer helping developers tune OpenShift pods.

Context:
{context}

Question:
{question}

Answer like an expert performance architect. Be specific and suggest root causes and improvements.
"""

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip(), context
    except Exception as e:
        return f"❌ OpenAI Error: {e}", context
