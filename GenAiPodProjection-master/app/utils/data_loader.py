import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_sec", "CPU_Load", "Memory_Load"]

def load_performance_data(uploaded_csv):
    try:
        df = pd.read_csv(uploaded_csv)
        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            st.error(f"CSV must include columns: {REQUIRED_COLUMNS}")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None
