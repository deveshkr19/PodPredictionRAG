import streamlit as st

def render_sliders():
    tps = st.slider("Expected TPS", 1, 100, 10, 1)
    cpu = st.slider("CPU Cores per Pod", 1, 2, 1)
    mem = st.slider("Memory per Pod (GB)", 1, 4, 2)
    resp = st.slider("Target Response Time (sec)", 1, 10, 2)
    return tps, cpu, mem, resp

def display_results(pods, cpu, mem, msg):
    st.write(f"### Estimated Pods Required: {pods}")
    st.write(f"Estimated CPU Utilization: {cpu:.2f}%")
    st.write(f"Estimated Memory Utilization: {mem:.2f}%")
    st.info(msg)

def show_footer():
    st.markdown("<br><hr><p style='text-align:center;'>Developed by Devesh Kumar</p>", unsafe_allow_html=True)
