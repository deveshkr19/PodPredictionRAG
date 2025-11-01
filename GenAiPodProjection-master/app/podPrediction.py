import streamlit as st
from utils.data_loader import load_performance_data
from utils.model_training import train_models
from utils.pod_estimator import predict_pods
from utils.rag_handler import handle_knowledge_base, handle_user_question
from utils.ui_components import render_sliders, display_results, show_footer
from utils import config

# --- App Title & Intro ---
st.title("Smart Performance Forecasting for OpenShift Pods")
st.write("""
This app forecasts the optimal number of pods required in OpenShift based on LoadRunner performance data.  
It uses ML + RAG (Retrieval-Augmented Generation) to enhance GPT responses with contextual knowledge.
""")

# --- Upload Performance CSV ---
uploaded_csv = st.file_uploader("Upload LoadRunner Performance Report", type=["csv"])

# --- Upload Knowledge Base Files ---
kb_files = st.file_uploader("Upload Knowledge Base Files (txt, csv, md)", type=["txt", "csv", "md"], accept_multiple_files=True)

# --- CSV Handling ---
if uploaded_csv:
    df = load_performance_data(uploaded_csv)
    if df is not None:
        st.write("Sample Data Preview:", df.head())

        # Train the regression models
        cpu_model, mem_model, cpu_r2, mem_r2 = train_models(df)
        st.write(f"Model Accuracy: CPU R² = {cpu_r2:.2f}, Memory R² = {mem_r2:.2f}")

        # User input sliders
        tps, cpu, mem, resp = render_sliders()

        # Predict optimal pod count
        pod_count, pred_cpu, pred_mem, status_msg = predict_pods(
            tps, cpu, mem, resp, cpu_model, mem_model,
            config.CPU_THRESHOLD, config.MEMORY_THRESHOLD
        )
        display_results(pod_count, pred_cpu, pred_mem, status_msg)

        # --- RAG Setup ---
        if kb_files:
            rag_db = handle_knowledge_base(kb_files, config.KB_FOLDER, config.FAISS_INDEX_PATH)
        else:
            try:
                rag_db = handle_knowledge_base(None, config.KB_FOLDER, config.FAISS_INDEX_PATH)
            except Exception:
                rag_db = None
                st.warning("⚠️ No existing FAISS index found. RAG will not work.")

        # --- GPT-4 Question Box ---
        question = st.text_input("Ask a performance-related question")
        if question:
            response, context_used = handle_user_question(rag_db, question, df, config.OPENAI_API_KEY)

            st.markdown("### AI Response:")
            st.write(response)

            if context_used:
                with st.expander("🔍 Context Retrieved from RAG"):
                    st.code(context_used, language="markdown")
            else:
                st.warning("⚠️ No context was retrieved. GPT used without RAG.")

# --- Footer ---
show_footer()
