```python
# app/podPrediction.py

import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Local utils
from utils.config import (
    BASE_DIR,
    KB_FOLDER,
    FAISS_INDEX_PATH,
    MAX_UTIL_RATIO,
)
from utils.rag_handler import ensure_index, ask
from utils.llm import answer_with_openai


# ----------------------------- Page Setup -----------------------------
st.set_page_config(
    page_title="Smart Performance Forecasting for OpenShift Pods",
    page_icon="📈",
    layout="wide",
)

st.title("Smart Performance Forecasting for OpenShift Pods")
st.caption(
    "Forecast optimal OpenShift pod counts from LoadRunner CSVs with simple, transparent models. "
    "Use RAG (FAISS + OpenAI) to answer performance questions grounded in your own knowledge base."
)


# ----------------------------- Helpers -----------------------------
REQUIRED_COLS = {
    "TPS",
    "CPU_Cores",
    "Memory_GB",
    "ResponseTime_sec",
    "CPU_Load",
    "Memory_Load",
}


@st.cache_data(show_spinner=False)
def load_lr_csv(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # Basic cleaning
    for c in ["TPS", "CPU_Load", "Memory_Load", "CPU_Cores", "Memory_GB", "ResponseTime_sec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TPS", "CPU_Load", "Memory_Load"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def fit_util_models(df: pd.DataFrame):
    """
    Fit simple linear models:
      CPU_Load%   ~= a_cpu * TPS + b_cpu
      Memory_Load%~= a_mem * TPS + b_mem
    """
    X = df[["TPS"]].values
    cpu_m = LinearRegression().fit(X, df["CPU_Load"].values)
    mem_m = LinearRegression().fit(X, df["Memory_Load"].values)
    return cpu_m, mem_m


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pods_from_util(cpu_pct_1pod: float, mem_pct_1pod: float, max_util_ratio: float) -> int:
    """
    Given predicted total utilization for a single pod at target TPS,
    compute pods needed to keep both CPU/Mem under max_util_ratio.
    """
    max_util_pct = max_util_ratio * 100.0
    need_cpu = math.ceil(cpu_pct_1pod / max_util_pct)
    need_mem = math.ceil(mem_pct_1pod / max_util_pct)
    return max(1, int(max(need_cpu, need_mem)))


def utilization_for_tps(cpu_model, mem_model, tps: float) -> tuple[float, float]:
    cpu = float(cpu_model.predict(np.array([[tps]]))[0])
    mem = float(mem_model.predict(np.array([[tps]]))[0])
    # keep within sane bounds
    return clamp(cpu, 0, 500), clamp(mem, 0, 500)


# ----------------------------- 1) Data Upload & Training -----------------------------
st.header("1) Upload LoadRunner CSV")

perf_file = st.file_uploader(
    "Upload performance CSV (required columns: TPS, CPU_Cores, Memory_GB, ResponseTime_sec, CPU_Load, Memory_Load)",
    type=["csv"],
)

df = None
cpu_model = mem_model = None
if perf_file:
    try:
        df = load_lr_csv(perf_file)
        cpu_model, mem_model = fit_util_models(df)
        st.success(f"Loaded {len(df)} rows. Trained utilization models.")
        with st.expander("Preview first 10 rows"):
            st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Failed to load/train: {e}")

if not cpu_model or not mem_model:
    # Fallback coefficients if no CSV yet (gentle defaults)
    class _Fallback:
        def __init__(self, a, b): self.coef_, self.intercept_ = np.array([a]), b
        def predict(self, X): return X.ravel() * self.coef_[0] + self.intercept_
    cpu_model = _Fallback(a=0.18, b=40.0)   # ~+18% CPU per +100 TPS
    mem_model = _Fallback(a=0.15, b=35.0)   # ~+15% MEM per +100 TPS
    st.info("Using fallback model (upload a CSV to train from real data).")


# ----------------------------- 2) Forecast Controls -----------------------------
st.header("2) Forecast Inputs")

c1, c2, c3, c4 = st.columns(4)
with c1:
    expected_tps = st.slider("Expected TPS", min_value=1, max_value=2000, value=120, step=1)
with c2:
    cpu_per_pod = st.slider("CPU cores per pod", min_value=1, max_value=8, value=1, step=1)
with c3:
    mem_per_pod = st.slider("Memory per pod (GiB)", min_value=1, max_value=32, value=2, step=1)
with c4:
    target_rt = st.slider("Target Response Time (sec)", min_value=1, max_value=10, value=4, step=1)

st.caption(f"Guardrail: keep per-pod CPU & Memory ≤ {int(MAX_UTIL_RATIO*100)}% on steady load.")

# Predict utilization if all load went to a single pod, then compute pods
pred_cpu_1pod, pred_mem_1pod = utilization_for_tps(cpu_model, mem_model, expected_tps)
pods = pods_from_util(pred_cpu_1pod, pred_mem_1pod, MAX_UTIL_RATIO)

est_cpu_per_pod = pred_cpu_1pod / pods
est_mem_per_pod = pred_mem_1pod / pods

st.subheader(f"Estimated Pods Required: {pods}")
st.write(f"Estimated CPU Utilization per pod: **{est_cpu_per_pod:.2f}%**")
st.write(f"Estimated Memory Utilization per pod: **{est_mem_per_pod:.2f}%**")

if est_cpu_per_pod <= MAX_UTIL_RATIO * 100 and est_mem_per_pod <= MAX_UTIL_RATIO * 100:
    st.info("Configuration is within acceptable limits.")
else:
    st.warning("Configuration exceeds utilization guardrails; increase pods or resources per pod.")

# Download result
result_row = pd.DataFrame(
    [{
        "tps": expected_tps,
        "cpu_per_pod": cpu_per_pod,
        "mem_per_pod_gib": mem_per_pod,
        "target_p95_sec": target_rt,
        "estimated_pods": pods,
        "est_cpu_util_per_pod_pct": round(est_cpu_per_pod, 2),
        "est_mem_util_per_pod_pct": round(est_mem_per_pod, 2),
    }]
)
st.download_button(
    "Download forecast as CSV",
    data=result_row.to_csv(index=False),
    file_name="pod_forecast.csv",
)

st.divider()


# ----------------------------- 3) Knowledge Base (RAG) -----------------------------
st.header("3) Knowledge Base (RAG)")

# Upload KB files (txt/md/csv)
kb_files = st.file_uploader(
    "Upload knowledge files (txt, md, csv). These are indexed for Q&A.",
    type=["txt", "md", "csv"],
    accept_multiple_files=True,
)
if kb_files:
    kb_dir = Path(KB_FOLDER)
    kb_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in kb_files:
        (kb_dir / f.name).write_bytes(f.read())
        saved += 1
    st.success(f"Saved {saved} file(s) to {kb_dir}")

# Build / Rebuild index
colA, colB = st.columns(2)
with colA:
    if st.button("Build / Rebuild FAISS Index"):
        st.session_state.db = ensure_index(KB_FOLDER, FAISS_INDEX_PATH)
        if st.session_state.db and all(st.session_state.db):
            st.toast("Index built.", icon="✅")
        else:
            st.toast("No KB docs found. Add files first.", icon="⚠️")

# Lazy load (or auto-build if index + docs exist)
if "db" not in st.session_state:
    st.session_state.db = ensure_index(KB_FOLDER, FAISS_INDEX_PATH)

if st.session_state.db and all(st.session_state.db):
    st.info("RAG ready ✓ Index found.")
else:
    st.warning("No existing FAISS index found. Upload KB files and click **Build / Rebuild FAISS Index**.")


# ----------------------------- 4) RAG Q&A (Enter-to-submit) -----------------------------
st.subheader("Ask a performance-related question")

# Controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    use_rag = st.checkbox("Use Knowledge Base (RAG)", value=True)
with col2:
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
with col3:
    temp = st.slider("Temp", 0.0, 1.0, 0.2, 0.05)


def run_answer(q: str):
    if not q.strip():
        st.warning("Type a question first.")
        return

    # Build context if RAG is enabled and index exists
    context = ""
    if use_rag:
        db_tuple = st.session_state.get("db", None)
        if not db_tuple or not all(db_tuple):
            st.error("Knowledge index not ready. Upload KB files and click **Build / Rebuild**.")
            return
        context = ask(db_tuple, q, k=8)  # wider retrieval
        context = context[:8000]  # keep prompt safe

    try:
        ans = answer_with_openai(context, q, model=model_choice, temperature=temp)
        st.markdown(ans)
        if use_rag:
            with st.expander("Show retrieved context"):
                st.code(context[:4000])
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")


# Use a form so ENTER submits the question
with st.form("rag_form", clear_on_submit=False):
    user_q = st.text_input(
        "Question",
        placeholder="e.g., What’s the safe CPU and memory utilization target per pod?",
        key="rag_q",
    )
    # Pressing ENTER triggers this submit button automatically
    submitted = st.form_submit_button("Answer with RAG" if use_rag else "Answer (model only)")

if submitted:
    run_answer(st.session_state.get("rag_q", ""))

st.caption("Developed by Devesh Kumar")
