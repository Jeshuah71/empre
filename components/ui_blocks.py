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

.block-container {{
  max-width: 1180px;
  padding-top: 2rem;
  padding-bottom: 6rem;
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

div[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #23242d 0%, #262731 100%);
}}

div[role="radiogroup"] {{
  gap: 0.4rem;
}}

div[role="radiogroup"] label {{
  background: {UI.code_bg};
  border: 1px solid {UI.border};
  border-radius: 999px;
  padding: 0.35rem 0.85rem;
}}

.metric-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 1rem 0 1.2rem;
}}

.metric-tile {{
  background: linear-gradient(180deg, #111923 0%, #0f1720 100%);
  border: 1px solid {UI.border};
  border-radius: 12px;
  padding: 14px;
  min-height: 138px;
}}

.metric-tile__label {{
  color: {UI.text_dim};
  font-size: 11px;
  letter-spacing: 0.6px;
  text-transform: uppercase;
}}

.metric-tile__value {{
  color: {UI.text};
  font-size: 1.65rem;
  font-weight: 600;
  margin-top: 0.45rem;
  line-height: 1.15;
}}

.metric-tile__trend {{
  font-size: 12px;
  font-weight: 700;
  margin-top: 0.7rem;
}}

.metric-tile__note {{
  color: {UI.text_dim};
  font-size: 12px;
  margin-top: 0.55rem;
}}

.investment-card {{
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 16px;
  background: linear-gradient(180deg, #111923 0%, #0f1720 100%);
  border: 1px solid {UI.border};
  border-radius: 12px;
  padding: 16px;
  margin: 0.75rem 0 1rem;
}}

.investment-card__stance {{
  color: {UI.text_dim};
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}}

.investment-card__value {{
  color: {UI.text};
  font-size: 2rem;
  font-weight: 700;
  margin: 0.3rem 0 0.8rem;
}}

.investment-card__bar {{
  height: 10px;
  width: 100%;
  background: #1b2430;
  border-radius: 999px;
  overflow: hidden;
}}

.investment-card__bar span {{
  display: block;
  height: 100%;
  background: linear-gradient(90deg, {UI.accent} 0%, #3ea6ff 100%);
  border-radius: 999px;
}}

.investment-card__rationale {{
  color: {UI.text};
  line-height: 1.55;
}}

.investment-card__verdict {{
  color: {UI.text_dim};
  margin-top: 0.7rem;
  line-height: 1.45;
}}

small {{
  color: {UI.text_dim};
}}

@media (max-width: 1024px) {{
  .block-container {{
    max-width: 100%;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
  }}

  .metric-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .investment-card {{
    grid-template-columns: 1fr;
  }}
}}

@media (max-width: 640px) {{
  .block-container {{
    padding-top: 1.1rem;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
    padding-bottom: 7rem;
  }}

  h1 {{
    font-size: 2rem;
  }}

  .metric-grid {{
    grid-template-columns: 1fr;
    gap: 10px;
  }}

  .metric-tile {{
    min-height: auto;
  }}

  div[data-testid="column"] {{
    min-width: 0;
  }}
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
