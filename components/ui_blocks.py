from __future__ import annotations

import random
import time
from typing import Dict

import streamlit as st

from config.settings import UI, APP


def inject_css():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {{
  font-family: {UI.font_stack};
  background-color: {UI.theme_bg};
  color: {UI.text};
}}

.stApp {{
  background: radial-gradient(1200px 800px at 10% 10%, #111a24 0%, {UI.theme_bg} 55%);
}}

h1, h2, h3 {{
  color: {UI.text};
  letter-spacing: 0.5px;
}}

.bloomberg-card {{
  background: {UI.panel_bg};
  border: 1px solid {UI.border};
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.2);
}}

.bloomberg-badge {{
  display: inline-block;
  padding: 4px 8px;
  border: 1px solid {UI.border};
  border-radius: 4px;
  color: {UI.accent};
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
}}

.metric {{
  font-size: 20px;
  font-weight: 600;
}}

.metric-label {{
  color: {UI.text_dim};
  font-size: 12px;
  letter-spacing: 0.8px;
}}

.good {{ color: {UI.good}; }}
.bad {{ color: {UI.bad}; }}
.info {{ color: {UI.info}; }}

.sidebar .sidebar-content {{
  background: {UI.panel_bg};
}}

.stTextInput > div > div > input,
.stTextArea textarea {{
  background: {UI.code_bg};
  color: {UI.text};
  border: 1px solid {UI.border};
}}

.stFileUploader > div {{
  background: {UI.code_bg};
  border: 1px dashed {UI.border};
}}

div[data-testid="stChatMessage"] {{
  background: {UI.panel_bg};
  border: 1px solid {UI.border};
  border-radius: 8px;
  padding: 12px;
}}

small {{
  color: {UI.text_dim};
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(state: Dict):
    with st.sidebar:
        st.markdown("<span class='bloomberg-badge'>Alpha-RAG</span>", unsafe_allow_html=True)
        st.markdown(f"**{APP.app_title}**")
        st.caption(APP.app_subtitle)
        st.divider()

        st.markdown("### Upload PDF")
        file = st.file_uploader(
            "Financial report",
            type=["pdf"],
            key="pdf_uploader",
            accept_multiple_files=True,
        )
        st.caption(f"Max size: {APP.max_upload_mb} MB")

        st.divider()
        st.markdown("### Session")
        if st.button("Clear chat history"):
            state["messages"] = []
            state["vectorstore"] = None
            state["doc_stats"] = None
            state["index_key"] = None
            st.toast("Chat cleared", icon="✅")

        return file


def render_metrics():
    st.markdown("<div class='bloomberg-card'>", unsafe_allow_html=True)
    cols = st.columns(3)
    metrics = _fake_metrics()
    for col, (label, value, delta, klass) in zip(cols, metrics):
        with col:
            st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric {klass}'>{value}</div>", unsafe_allow_html=True)
            st.markdown(f"<small class='{klass}'>{delta}</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _fake_metrics():
    seed = int(time.time()) // 30
    random.seed(seed)
    return [
        ("RISK SCORE", f"{random.randint(61, 89)}", f"{random.choice(['+','-'])}{random.randint(1,4)}%", "info"),
        ("ALPHA SIGNAL", f"{random.uniform(-0.8, 1.6):.2f}", f"{random.choice(['+','-'])}{random.uniform(0.1, 0.6):.2f}", "good"),
        ("VOLATILITY", f"{random.uniform(12.5, 28.4):.2f}%", f"{random.choice(['+','-'])}{random.uniform(0.2, 0.9):.2f}%", "bad"),
    ]
