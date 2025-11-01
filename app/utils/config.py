import streamlit as st

# Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# Thresholds
CPU_THRESHOLD = 75
MEMORY_THRESHOLD = 75

# Paths
FAISS_INDEX_PATH = "vectorstore"
KB_FOLDER = "knowledge_base"
