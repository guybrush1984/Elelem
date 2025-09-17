#!/usr/bin/env python3
import streamlit as st
import requests

st.title("🚀 Minimal Elelem Dashboard")

try:
    response = requests.get("http://localhost:9200/health", timeout=5)
    st.success(f"✅ Server responding: {response.json()}")
except Exception as e:
    st.error(f"❌ Server error: {e}")

try:
    response = requests.get("http://localhost:9200/v1/metrics/summary", timeout=5)
    metrics = response.json()
    st.json(metrics)
except Exception as e:
    st.error(f"❌ Metrics error: {e}")

st.write("If you see this, Streamlit is working!")