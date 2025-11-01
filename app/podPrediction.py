# app/podPrediction.py

import math
from pathlib import Path
import os

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
    "This app forecasts the optimal number of pods in OpenShift using your LoadRunner CSV "
    "and augments explanations via RAG (FAISS + OpenAI)."
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
def load_lr_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(file_bytes)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # Clean basics
    for c in ["TPS", "CPU_Load", "Memory_Load"]:
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
    Given predicted total utilization for a *single* pod at the target TPS,
    compute how many pods we need to keep both CPU/Mem under max_util_ratio.
    """
    max_util_pct = max_util_ratio * 100.0
    need_cpu = math.ceil(cpu_pct_1pod / max_util_pct)
    need_mem = math.ceil(mem_pct_1pod / max_util_pct)
    return max(1, int(max(need_cpu, need_mem)))


def utilization_for_tps(cpu_model, mem_model, tps: float) -> tuple[float, float]:
    cpu = float(cpu_model.predict(np.array([[tps]]))[0])
    mem = float(mem_model.predict(np.array([[tps]]))[0])
    # Keep within sane bounds
    return clamp(cpu, 0, 500), clamp(mem, 0, 500)


# ----------------------------- Data Upload & Training -----------------------------
st.header("1) Upload LoadRunner CSV")

left, right = st.columns([2, 1])
with left:
    perf_file = st.file_uploader(
        "Upload performance CSV (columns required: TPS, CPU_Cores, Memory_GB, ResponseTime_sec, CPU_Load, Memory_Load)",
        type=["csv"],
    )
with right:
    st.write(" ")
    st.write("Need a sample?")
    st.link_button("Download 150-row sample CSV", url="https://sandbox:/mnt/data/sample_pod_prediction_inputs/loadrunner_performance_large.csv")  # This link works only in ChatGPT preview; ignore in Streamlit Cloud.

df = None
cpu_model = mem_model = None
if perf_file:
    try:
        df = load_lr_csv(perf_file)
        cpu_model, mem_model = fit_util_models(df)
        st.success(f"Loaded {len(df)} rows. Trained simple utilization models.")
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
    st.info("Using fallback utilization model (upload a CSV to train from real data).")


# ----------------------------- Forecast Controls -----------------------------
st.header("2) Forecast Inputs")

c1, c2, c3, c4 = st.columns(4)
with c1:
    expected_tps = st.slider("Expected TPS", min_value=1, max_value=2000, value=120, step=1)
with c2:
    cpu_per_pod = st.slider("CPU cores per pod", min_value=1, max_value=8, value=1, step=1)
with c3:
    mem_per_pod = st.slider("Memory per pod (GiB)", min_value=1, max_value=16, value=2, step=1)
with c4:
    target_rt = st.slider("Target Response Time (sec)", min_value=1, max_value=10, value=4, step=1)

st.caption(f"Guardrail: keep per-pod CPU & Memory ≤ {int(MAX_UTIL_RATIO*100)}% on steady load.")

# Predict utilization if all load goes to a single pod, then compute pods
pred_cpu_1pod, pred_mem_1pod = utilization_for_tps(cpu_model, mem_model, expected_tps)
pods = pods_from_util(pred_cpu_1pod, pred_mem_1pod, MAX_UTIL_RATIO)

est_cpu_per_pod = pred_cpu_1pod / pods
est_mem_per_pod = pred_mem_1pod / pods

st.subheader("Estimated Pods Required: **{}**".format(pods))
st.write(f"Estimated CPU Utilization per pod: **{est_cpu_per_pod:.2f}%**")
st.write(f"Estimated Memory Utilization per pod: **{est_mem_per_pod:.2f}%**")

if est_cpu_per_pod <= MAX_UTIL_RATIO * 100 and est_mem_per_pod <= MAX_UTIL_RATIO * 100:
    st.info("Configuration is within acceptable limits.")
else:
    st.warning("Configuration exceeds utilization guardrails; consider increasing pods, CPU, or memory.")

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


# ----------------------------- RAG Section -----------------------------
st.header("3) Knowledge Base (RAG)")

# Upload KB files (txt/md/csv)
kb_files = st.file_uploader(
    "Upload knowledge files (txt, md, csv). These will be indexed for Q&A.",
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

colA, colB = st.columns(2)
with colA:
    if st.button("Build / Rebuild FAISS Index"):
        st.session_state.db = ensure_index(KB_FOLDER, FAISS_INDEX_PATH)
        if st.session_state.db:
            st.toast("Index built.", icon="✅")
        else:
            st.toast("No KB docs found. Add files first.", icon="⚠️")

# Lazy load (or auto-build if index + docs exist)
if "db" not in st.session_state:
    st.session_state.db = ensure_index(KB_FOLDER, FAISS_INDEX_PATH)

if st.session_state.db:
    st.info("RAG ready ✓ Index found.")
else:
    st.warning("No existing FAISS index found. Upload KB files and click **Build / Rebuild FAISS Index**.")

# Ask a question
st.subheader("Ask a performance-related question")
cA, cB = st.columns([3, 1])
with cA:
    user_q = st.text_input("Question", placeholder="e.g., Why does p95 rise when CPU > 80% but errors are 0%?")
with cB:
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

if st.button("Answer with RAG"):
    if not st.session_state.db:
        st.error("No index yet. Upload KB files and build the index.")
    elif not user_q.strip():
        st.warning("Type a question first.")
    else:
        # Retrieve top chunks
        context = ask(st.session_state.db, user_q, k=6)
        context = context[:8000]  # keep prompt small for reliability
        try:
            answer = answer_with_openai(context, user_q, model=model_choice, temperature=0.2)
            st.markdown(answer)
            with st.expander("Show retrieved context"):
                st.code(context[:4000])
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")

st.caption("Developed by Devesh Kumar")
