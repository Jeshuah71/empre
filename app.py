from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple
from urllib import error, request

import streamlit as st
import streamlit.components.v1 as components
from langchain.prompts import ChatPromptTemplate

from config.settings import APP, validate_env
from core.document_processor import DocumentProcessor, DocumentProcessorError
from core.llm_engine import LLMEngine


BASE_DIR = Path(__file__).resolve().parent
LANDING_ROUTE = "landing"
DEMO_ROUTE = "demo"
TRIAL_LIMIT = 3
REPORT_OPTIONS = [
    "NVIDIA Corp. (FY2025 10-K)",
    "Microsoft Corp. (Q4 2025 10-Q)",
    "Upload Custom PDF...",
]
REPORT_PATHS = {
    "NVIDIA Corp. (FY2025 10-K)": BASE_DIR / "data" / "uploads" / "NVIDIA_10K_2026.pdf",
    "Microsoft Corp. (Q4 2025 10-Q)": BASE_DIR / "data" / "uploads" / "Microsoft_10Q.pdf",
}
PRICING_URL = "/?view=landing#pricing"
FORMSPREE_ENDPOINT = os.getenv("FORMSPREE_ENDPOINT", "https://formspree.io/f/mdapawak")
WORKSPACE_OPTIONS = ["Chat", "Metrics", "Investment"]
QUICK_QUERY_MAP = {
    "NVIDIA Corp. (FY2025 10-K)": [
        ("quick_yoy_revenue", "📈 YoY Revenue", "Summarize YoY revenue growth across NVIDIA and highlight Data Center and Gaming performance."),
        ("quick_risk_factors", "⚠️ Risk Factors", "Analyze Item 1A: Risk Factors and summarize the most material risks for NVIDIA."),
        ("quick_ai_capex", "🏗️ AI CapEx", "Audit AI infrastructure demand, capital intensity, and supply constraints discussed in the NVIDIA filing."),
    ],
    "Microsoft Corp. (Q4 2025 10-Q)": [
        ("quick_yoy_revenue", "📈 YoY Revenue", "Summarize YoY revenue growth and highlight Azure or Cloud performance where disclosed."),
        ("quick_risk_factors", "⚠️ Risk Factors", "Analyze Item 1A: Risk Factors and summarize the most material risks for Microsoft."),
        ("quick_ai_capex", "🏗️ AI CapEx", "Audit CapEx versus AI infrastructure spend and explain how Microsoft describes data center or cloud investment."),
    ],
    "Upload Custom PDF...": [
        ("quick_yoy_revenue", "📈 YoY Revenue", "Summarize year-over-year revenue growth using the filing selected in this session."),
        ("quick_risk_factors", "⚠️ Risk Factors", "Analyze the most material risk factors discussed in the selected filing."),
        ("quick_ai_capex", "🏗️ AI CapEx", "Audit capital expenditures, infrastructure investment, and AI-related spending in the selected filing."),
    ],
}
METRIC_SPECS = [
    (
        "Revenue",
        "What is the most clearly reported revenue figure in this filing? Prefer the primary consolidated statement. "
        "If possible include the period context and whether the filing indicates year-over-year change.",
    ),
    (
        "Net Income",
        "What is the most clearly reported net income or net earnings figure in this filing? Prefer the primary consolidated statement.",
    ),
    (
        "Operating Cash Flow",
        "What is the reported net cash provided by operating activities in this filing? Prefer the statement of cash flows.",
    ),
    (
        "Cash & Equivalents",
        "What is the reported cash and cash equivalents balance in this filing? Prefer the balance sheet period-end figure.",
    ),
    (
        "Debt",
        "What is the most clearly reported debt figure in this filing? Prefer total debt, long-term debt, or debt obligations from the balance sheet or debt footnotes.",
    ),
    (
        "CapEx",
        "What is the reported capital expenditures figure in this filing? Prefer property and equipment additions or purchases from the cash flow statement or MD&A.",
    ),
]
METRIC_KEYWORDS = {
    "Revenue": ["revenue", "net sales", "total revenue", "statement of income", "income statement"],
    "Net Income": ["net income", "net earnings", "net loss", "statement of income", "income statement"],
    "Operating Cash Flow": ["operating activities", "net cash provided by operating activities", "cash flows"],
    "Cash & Equivalents": ["cash and cash equivalents", "balance sheets", "cash equivalents"],
    "Debt": ["long-term debt", "total debt", "debt", "borrowings", "notes payable"],
    "CapEx": ["capital expenditures", "purchases of property", "property and equipment", "additions", "capex"],
}
RISK_KEYWORDS = ["risk factors", "item 1a", "risk factor", "uncertainties", "material adverse"]
INVESTMENT_PROMPT = """
You are preparing a document-based investment readiness view for one SEC filing.
Use only the provided filing context.
Return valid JSON only with this exact schema:
{
  "stance": "Constructive|Balanced|Cautious",
  "score": 0,
  "rationale": "2 sentence rationale",
  "bullish_signals": ["...", "...", "..."],
  "risk_signals": ["...", "...", "..."],
  "verdict": "short final takeaway"
}
The score must be an integer from 0 to 100 where:
0-39 = Cautious, 40-69 = Balanced, 70-100 = Constructive.
This is not financial advice; base the output only on filing evidence.
Question: Build the investment readiness view for this filing.
""".strip()


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = None
    if "index_key" not in st.session_state:
        st.session_state.index_key = None
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
    if "selected_report" not in st.session_state:
        st.session_state.selected_report = REPORT_OPTIONS[0]
    if "current_route" not in st.session_state:
        st.session_state.current_route = LANDING_ROUTE
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Chat"
    if "insight_cache" not in st.session_state:
        st.session_state.insight_cache = {}
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    if "raw_documents" not in st.session_state:
        st.session_state.raw_documents = []
    if "paywall_triggered" not in st.session_state:
        st.session_state.paywall_triggered = False


def _get_route() -> str:
    view = st.query_params.get("view", LANDING_ROUTE)
    if isinstance(view, list):
        view = view[0] if view else LANDING_ROUTE
    return DEMO_ROUTE if view == DEMO_ROUTE else LANDING_ROUTE


def _set_route(route: str):
    st.session_state.current_route = route
    st.query_params["view"] = route
    st.rerun()


def _browser_shell(route: str):
    browser_path = "/demo" if route == DEMO_ROUTE else "/"
    components.html(
        f"""
        <script>
        const desiredPath = {browser_path!r};
        if (window.location.pathname !== desiredPath) {{
          window.history.replaceState({{}}, "", desiredPath + window.location.search + window.location.hash);
        }}
        const hideChrome = () => {{
          const footer = window.parent.document.querySelector("footer");
          if (footer) footer.style.display = "none";
          const badge = window.parent.document.querySelector('[data-testid="stStatusWidget"]');
          if (badge) badge.style.display = "none";
        }};
        hideChrome();
        const observer = new MutationObserver(hideChrome);
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        </script>
        """,
        height=0,
        width=0,
    )


def _logo_data_uri() -> str:
    logo_path = BASE_DIR / "logo_vultur.png"
    if not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _inject_styles():
    is_light = st.session_state.theme_mode == "Light"
    if is_light:
        theme_vars = """
        --alpha-bg: #f3f7fc;
        --alpha-bg-2: #edf3fb;
        --alpha-panel: rgba(255, 255, 255, 0.9);
        --alpha-panel-strong: rgba(255, 255, 255, 0.96);
        --alpha-panel-soft: rgba(255, 255, 255, 0.78);
        --alpha-surface-1: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 249, 255, 0.96) 100%);
        --alpha-surface-2: linear-gradient(180deg, rgba(248, 251, 255, 0.98) 0%, rgba(239, 245, 252, 0.98) 100%);
        --alpha-surface-3: linear-gradient(180deg, rgba(241, 247, 255, 0.98) 0%, rgba(233, 241, 252, 0.98) 100%);
        --alpha-soft-fill: rgba(17, 40, 68, 0.03);
        --alpha-nav-fill: rgba(255, 255, 255, 0.8);
        --alpha-button-fill: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(242, 247, 255, 0.98) 100%);
        --alpha-shadow: 0 22px 60px rgba(31, 57, 88, 0.14);
        --alpha-border: rgba(28, 56, 91, 0.12);
        --alpha-text: #0d1b2a;
        --alpha-dim: #546579;
        --alpha-blue: #2d6df6;
        --alpha-blue-strong: #2d6df6;
        --alpha-gold: #9a6f2f;
        --alpha-blue-soft: rgba(45, 109, 246, 0.12);
        """
        app_background = """
            radial-gradient(1200px 700px at 8% 0%, rgba(45, 109, 246, 0.1) 0%, rgba(45, 109, 246, 0) 55%),
            radial-gradient(800px 520px at 90% 10%, rgba(202, 168, 106, 0.08) 0%, rgba(202, 168, 106, 0) 56%),
            linear-gradient(180deg, #f3f7fc 0%, #eef4fb 32%, #f7faff 100%)
        """
        sidebar_background = "linear-gradient(180deg, #f4f8fd 0%, #eef4fb 100%)"
    else:
        theme_vars = """
        --alpha-bg: #07111f;
        --alpha-bg-2: #0a1628;
        --alpha-panel: rgba(14, 24, 40, 0.88);
        --alpha-panel-strong: rgba(18, 30, 49, 0.96);
        --alpha-panel-soft: rgba(17, 29, 47, 0.72);
        --alpha-surface-1: linear-gradient(180deg, rgba(13, 22, 37, 0.84) 0%, rgba(12, 22, 36, 0.72) 100%);
        --alpha-surface-2: linear-gradient(180deg, rgba(15, 27, 44, 0.96) 0%, rgba(10, 20, 34, 0.96) 100%);
        --alpha-surface-3: linear-gradient(180deg, rgba(19, 29, 45, 0.94) 0%, rgba(14, 23, 38, 0.94) 100%);
        --alpha-soft-fill: rgba(255, 255, 255, 0.03);
        --alpha-nav-fill: rgba(8, 16, 29, 0.72);
        --alpha-button-fill: linear-gradient(180deg, rgba(27, 41, 65, 0.94) 0%, rgba(18, 29, 47, 0.94) 100%);
        --alpha-shadow: 0 22px 60px rgba(2, 8, 17, 0.34);
        --alpha-border: rgba(118, 155, 203, 0.16);
        --alpha-text: #f4f7fb;
        --alpha-dim: #9aa9bc;
        --alpha-blue: #4a8dff;
        --alpha-blue-strong: #77a8ff;
        --alpha-gold: #caa86a;
        --alpha-blue-soft: rgba(74, 141, 255, 0.18);
        """
        app_background = """
            radial-gradient(1200px 700px at 8% 0%, rgba(74, 141, 255, 0.17) 0%, rgba(74, 141, 255, 0) 55%),
            radial-gradient(800px 520px at 90% 10%, rgba(202, 168, 106, 0.1) 0%, rgba(202, 168, 106, 0) 56%),
            linear-gradient(180deg, #07111f 0%, #0a1320 32%, #09111a 100%)
        """
        sidebar_background = "linear-gradient(180deg, #151a22 0%, #11161e 100%)"

    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;700;800&display=swap');

        :root {
          __THEME_VARS__
        }

        html, body, [class*="css"] {
          font-family: 'Inter', sans-serif;
          background: var(--alpha-bg);
          color: var(--alpha-text);
          scroll-behavior: smooth;
        }

        .stApp {
          background: __APP_BACKGROUND__;
        }

        .block-container {
          max-width: 1260px;
          padding-top: 1.2rem;
          padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
          background: __SIDEBAR_BACKGROUND__;
          border-right: 1px solid rgba(255,255,255,0.05);
        }

        [data-testid="stSidebar"] .block-container {
          padding-top: 1.25rem;
        }

        h1, h2, h3 {
          color: var(--alpha-text);
          letter-spacing: -0.02em;
        }

        .alpha-nav {
          position: sticky;
          top: 0.8rem;
          z-index: 20;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1.4rem;
          padding: 1.15rem 1.45rem;
          margin-bottom: 2.5rem;
          background: var(--alpha-nav-fill);
          border: 1px solid var(--alpha-border);
          border-radius: 26px;
          box-shadow: var(--alpha-shadow);
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
        }

        .alpha-nav__brand {
          display: inline-flex;
          align-items: center;
          gap: 1rem;
          font-family: 'Manrope', sans-serif;
          font-size: 1.3rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--alpha-text);
        }

        .alpha-nav__logo {
          width: 48px;
          height: 48px;
          object-fit: contain;
          display: block;
          filter: drop-shadow(0 8px 18px rgba(45, 109, 246, 0.18));
        }

        .alpha-theme-wrap {
          display: flex;
          justify-content: flex-end;
          margin-bottom: 0.75rem;
        }

        .alpha-nav__links {
          display: flex;
          align-items: center;
          gap: 1.35rem;
          flex-wrap: wrap;
        }

        .alpha-nav__links a,
        .alpha-nav__cta {
          text-decoration: none;
          color: var(--alpha-dim);
          font-size: 1rem;
          font-weight: 600;
          transition: color 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
        }

        .alpha-nav__links a:hover,
        .alpha-nav__cta:hover {
          color: var(--alpha-text);
          transform: translateY(-1px);
        }

        .alpha-nav__cta {
          color: var(--alpha-text);
          padding: 0.9rem 1.25rem;
          border-radius: 999px;
          border: 1px solid var(--alpha-border);
          background: var(--alpha-button-fill);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        .alpha-hero {
          padding: 2.4rem 0 2.2rem;
        }

        .alpha-hero-shell {
          display: grid;
          grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.85fr);
          gap: 1.5rem;
          align-items: stretch;
        }

        .alpha-hero-copy {
          background: var(--alpha-surface-1);
          border: 1px solid var(--alpha-border);
          border-radius: 30px;
          padding: 2.35rem;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-hero-panel {
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 30px;
          padding: 1.85rem;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-eyebrow {
          color: var(--alpha-blue-strong);
          font-size: 0.86rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 1rem;
        }

        .alpha-title {
          font-family: 'Manrope', sans-serif;
          font-size: clamp(2.9rem, 6vw, 5.1rem);
          line-height: 0.95;
          font-weight: 800;
          max-width: 760px;
          margin: 0;
          color: var(--alpha-text) !important;
        }

        .alpha-subtitle {
          max-width: 650px;
          color: var(--alpha-dim);
          font-size: 1.08rem;
          line-height: 1.8;
          margin-top: 1.25rem;
        }

        .alpha-actions {
          display: flex;
          gap: 0.9rem;
          align-items: center;
          flex-wrap: wrap;
          margin-top: 1.7rem;
        }

        .alpha-secondary-link {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 2.9rem;
          padding: 0.82rem 1.1rem;
          border-radius: 999px;
          color: var(--alpha-text);
          text-decoration: none;
          font-weight: 700;
          border: 1px solid rgba(118, 155, 203, 0.18);
          background: rgba(255, 255, 255, 0.02);
        }

        .alpha-proof {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.8rem;
          margin-top: 1.7rem;
        }

        .alpha-proof__item {
          background: var(--alpha-soft-fill);
          border: 1px solid var(--alpha-border);
          border-radius: 18px;
          padding: 1rem;
        }

        .alpha-proof__value {
          font-family: 'Manrope', sans-serif;
          font-size: 1.55rem;
          font-weight: 800;
          color: var(--alpha-text);
        }

        .alpha-proof__label {
          color: var(--alpha-dim);
          font-size: 0.88rem;
          line-height: 1.5;
          margin-top: 0.3rem;
        }

        .alpha-panel-label {
          color: var(--alpha-gold);
          font-size: 0.8rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 1rem;
        }

        .alpha-panel-title {
          font-family: 'Manrope', sans-serif;
          font-size: 1.55rem;
          font-weight: 800;
          line-height: 1.2;
          color: var(--alpha-text);
        }

        .alpha-panel-copy {
          color: var(--alpha-dim);
          line-height: 1.75;
          margin-top: 0.85rem;
          font-size: 0.97rem;
        }

        .alpha-list {
          margin: 1.2rem 0 0;
          padding: 0;
          list-style: none;
        }

        .alpha-list li {
          color: var(--alpha-text);
          line-height: 1.65;
          padding: 0.85rem 0;
          border-top: 1px solid rgba(118, 155, 203, 0.12);
        }

        .alpha-list li:first-child {
          border-top: 0;
          padding-top: 0;
        }

        .alpha-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 18px;
          margin-top: 1.35rem;
        }

        .alpha-card {
          background: var(--alpha-surface-3);
          border: 1px solid var(--alpha-border);
          border-radius: 24px;
          padding: 1.35rem;
          min-height: 210px;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-card__kicker {
          color: var(--alpha-blue-strong);
          font-size: 0.8rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .alpha-card__title {
          font-family: 'Manrope', sans-serif;
          font-size: 1.22rem;
          font-weight: 700;
          margin-top: 0.85rem;
          color: var(--alpha-text);
        }

        .alpha-card__copy {
          color: var(--alpha-dim);
          line-height: 1.7;
          margin-top: 0.85rem;
          font-size: 0.96rem;
        }

        .alpha-section {
          margin-top: 4.5rem;
        }

        .alpha-section-heading {
          display: flex;
          align-items: end;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: 1.15rem;
        }

        .alpha-section-heading h2 {
          font-family: 'Manrope', sans-serif;
          font-size: clamp(1.9rem, 3vw, 2.6rem);
          margin: 0.2rem 0 0;
        }

        .alpha-section-intro {
          max-width: 640px;
          color: var(--alpha-dim);
          line-height: 1.75;
        }

        .alpha-feature-band {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          margin-top: 1.2rem;
        }

        .alpha-band-card {
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 24px;
          padding: 1.5rem;
        }

        .alpha-band-card h3 {
          font-family: 'Manrope', sans-serif;
          font-size: 1.25rem;
          margin: 0 0 0.7rem;
        }

        .alpha-band-card p {
          color: var(--alpha-dim);
          line-height: 1.75;
          margin: 0;
        }

        .alpha-pricing {
          display: grid;
          grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
          gap: 1.2rem;
          align-items: stretch;
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 28px;
          padding: 1.7rem;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-pricing__price {
          color: var(--alpha-blue-strong);
          font-family: 'Manrope', sans-serif;
          font-size: 2.8rem;
          font-weight: 800;
        }

        .alpha-pricing__card {
          background: var(--alpha-soft-fill);
          border: 1px solid var(--alpha-border);
          border-radius: 22px;
          padding: 1.35rem;
        }

        .alpha-pricing__list {
          list-style: none;
          padding: 0;
          margin: 1rem 0 0;
        }

        .alpha-pricing__list li {
          color: var(--alpha-text);
          padding: 0.7rem 0;
          border-top: 1px solid rgba(118, 155, 203, 0.1);
        }

        .alpha-pricing__list li:first-child {
          border-top: 0;
          padding-top: 0;
        }

        .alpha-contact {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
          gap: 18px;
          align-items: stretch;
        }

        .alpha-contact-card {
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 28px;
          padding: 1.7rem;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-contact-box {
          background: var(--alpha-soft-fill);
          border: 1px solid var(--alpha-border);
          border-radius: 22px;
          padding: 1.2rem;
          margin-top: 1rem;
        }

        .alpha-contact-label {
          color: var(--alpha-dim);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .alpha-contact-value {
          font-family: 'Manrope', sans-serif;
          font-size: 1.15rem;
          font-weight: 700;
          color: var(--alpha-text);
          margin-top: 0.3rem;
        }

        .alpha-contact-copy {
          color: var(--alpha-dim);
          line-height: 1.75;
          margin-top: 0.55rem;
        }

        .alpha-contact-form {
          background: var(--alpha-surface-1);
          border: 1px solid var(--alpha-border);
          border-radius: 28px;
          padding: 1.7rem;
          box-shadow: var(--alpha-shadow);
        }

        .alpha-demo-shell {
          padding-top: 0.25rem;
        }

        .alpha-demo-header {
          margin-bottom: 1.25rem;
        }

        .alpha-demo-header h1 {
          font-size: clamp(2.4rem, 5vw, 3.6rem);
          line-height: 1;
        }

        .alpha-demo-label {
          color: var(--alpha-dim);
          font-size: 0.88rem;
          margin-top: 0.35rem;
        }

        .alpha-demo-panel {
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 18px;
          padding: 1rem;
          overflow: hidden;
        }

        .alpha-demo-empty {
          border: 1px dashed var(--alpha-border);
          background: var(--alpha-surface-1);
          border-radius: 16px;
          padding: 1rem;
          margin-bottom: 1rem;
        }

        .alpha-demo-empty__title {
          font-size: 1rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .alpha-demo-empty__copy {
          color: var(--alpha-dim);
          line-height: 1.65;
          font-size: 0.92rem;
        }

        div[data-testid="stChatMessage"] {
          background: var(--alpha-panel);
          border: 1px solid var(--alpha-border);
          border-radius: 14px;
          padding: 12px;
        }

        .alpha-sidebar-title {
          font-size: 1.35rem;
          font-weight: 700;
          margin-top: 0.9rem;
        }

        .alpha-sidebar-copy {
          color: var(--alpha-dim);
          font-size: 0.92rem;
        }

        div[role="radiogroup"] {
          gap: 0.5rem;
          margin-bottom: 0.35rem;
        }

        div[role="radiogroup"] label {
          background: var(--alpha-soft-fill);
          border: 1px solid var(--alpha-border);
          border-radius: 999px;
          padding: 0.3rem 0.8rem;
        }

        .alpha-metric-tile {
          background: var(--alpha-surface-2);
          border: 1px solid var(--alpha-border);
          border-radius: 16px;
          padding: 1rem;
          min-height: 136px;
          margin-bottom: 0.9rem;
        }

        .alpha-metric-label {
          color: var(--alpha-dim);
          font-size: 0.75rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .alpha-metric-value {
          color: var(--alpha-text);
          font-size: 1.6rem;
          font-weight: 700;
          margin-top: 0.45rem;
          line-height: 1.15;
        }

        .alpha-metric-trend {
          font-size: 0.78rem;
          font-weight: 700;
          margin-top: 0.6rem;
        }

        .alpha-metric-note {
          color: var(--alpha-dim);
          font-size: 0.82rem;
          line-height: 1.5;
          margin-top: 0.55rem;
        }

        .alpha-overlay {
          position: fixed;
          top: 0;
          right: 0;
          bottom: 0;
          left: 21rem;
          background: rgba(8, 12, 18, 0.62);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          z-index: 999;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1rem;
        }

        .alpha-overlay__card {
          width: min(640px, 100%);
          background: rgba(18, 24, 33, 0.94);
          border: 1px solid rgba(0, 123, 255, 0.24);
          border-radius: 22px;
          box-shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
          padding: 1.6rem;
          text-align: center;
        }

        .alpha-overlay__title {
          font-size: 1.4rem;
          font-weight: 800;
          margin-bottom: 0.75rem;
        }

        .alpha-overlay__copy {
          color: var(--alpha-dim);
          line-height: 1.75;
          margin-bottom: 1.2rem;
        }

        .alpha-overlay__button {
          display: inline-block;
          text-decoration: none;
          background: linear-gradient(90deg, #4a8dff 0%, #79aaff 100%);
          color: #ffffff !important;
          padding: 0.85rem 1.2rem;
          border-radius: 999px;
          font-weight: 700;
          border: 1px solid rgba(255,255,255,0.16);
          box-shadow: 0 14px 26px rgba(23, 86, 196, 0.28);
        }

        .alpha-overlay__button:hover,
        .alpha-overlay__button:visited,
        .alpha-overlay__button:active {
          color: #ffffff !important;
          text-decoration: none;
        }

        .alpha-citation-note {
          color: var(--alpha-dim);
          font-size: 0.92rem;
        }

        .stButton > button, .stLinkButton > a {
          border-radius: 999px !important;
          min-height: 3.2rem;
          font-size: 1rem;
        }

        [data-testid="stSegmentedControl"] {
          background: transparent;
        }

        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #4a8dff 0%, #6ea4ff 100%);
          color: #fff;
          border: 1px solid rgba(255,255,255,0.08);
          font-weight: 700;
        }

        @media (max-width: 1024px) {
          .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
          }

          .alpha-hero-shell,
          .alpha-pricing,
          .alpha-contact,
          .alpha-feature-band {
            grid-template-columns: 1fr;
          }

          .alpha-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }

          .alpha-overlay {
            left: 0;
            padding: 1.25rem;
          }

          .alpha-overlay__card {
            width: min(560px, 100%);
          }
        }

        @media (max-width: 640px) {
          .block-container {
            padding-top: 1rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-bottom: 6rem;
          }

          .alpha-hero {
            padding-top: 2.2rem;
          }

          .alpha-title {
            font-size: 2.25rem;
          }

          .alpha-nav {
            top: 0.3rem;
            align-items: flex-start;
            flex-direction: column;
            padding: 1rem;
          }

          .alpha-proof,
          .alpha-grid {
            grid-template-columns: 1fr;
          }

          .alpha-section {
            margin-top: 3rem;
          }

          .alpha-section-heading {
            flex-direction: column;
            align-items: flex-start;
          }

          [data-testid="stHorizontalBlock"] {
            flex-direction: column;
            gap: 0.85rem;
          }

          [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }

          .alpha-hero-copy,
          .alpha-hero-panel,
          .alpha-band-card,
          .alpha-pricing,
          .alpha-contact-card,
          .alpha-contact-form {
            padding: 1.25rem;
            border-radius: 22px;
          }

          .alpha-pricing__price {
            font-size: 2.2rem;
          }

          .alpha-demo-header h1 {
            font-size: 2.1rem;
          }

          .alpha-demo-label,
          .alpha-demo-empty__copy,
          .alpha-card__copy,
          .alpha-panel-copy,
          .alpha-contact-copy,
          .alpha-section-intro {
            font-size: 0.94rem;
            line-height: 1.6;
          }

          .alpha-demo-panel,
          .alpha-demo-empty,
          div[data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.9rem;
          }

          .alpha-metric-tile {
            min-height: auto;
            padding: 0.95rem;
          }

          .alpha-metric-value {
            font-size: 1.32rem;
          }

          .alpha-overlay__card {
            padding: 1.2rem;
            border-radius: 18px;
          }

          .alpha-overlay__title {
            font-size: 1.15rem;
          }

          .alpha-overlay__copy {
            font-size: 0.94rem;
            line-height: 1.6;
          }

          .stButton > button,
          .stLinkButton > a {
            min-height: 3rem;
            font-size: 0.96rem;
            white-space: normal;
          }
        }
        </style>
        """
    css = css.replace("__THEME_VARS__", theme_vars.strip())
    css = css.replace("__APP_BACKGROUND__", app_background.strip())
    css = css.replace("__SIDEBAR_BACKGROUND__", sidebar_background)
    st.markdown(css, unsafe_allow_html=True)


def _persist_upload(uploaded_file) -> Tuple[List[str], str]:
    upload_root = Path(APP.upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    safe_name = f"{file_hash[:12]}_{uploaded_file.name}"
    path = upload_root / safe_name
    if not path.exists():
        path.write_bytes(data)
    return [str(path)], file_hash


def _build_index_key(file_paths: List[str]) -> str:
    normalized = []
    for file_path in file_paths:
        path = Path(file_path)
        stat = path.stat()
        normalized.append(f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}")
    return hashlib.sha256("|".join(sorted(normalized)).encode("utf-8")).hexdigest()


def _build_audit_trail(source_documents: List) -> List[str]:
    seen = set()
    lines = []
    for doc in source_documents or []:
        meta = doc.metadata or {}
        source = meta.get("document") or meta.get("source", "unknown")
        page = meta.get("page", "?")
        if isinstance(page, int):
            page = page + 1
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {source} (page {page})")
    return lines


def _render_sources(audit_trail: List[str]):
    if not audit_trail:
        return
    with st.expander("View Audit Trail"):
        for line in audit_trail:
            st.markdown(line)


def _render_chat_intro():
    st.markdown(
        """
        <div class="alpha-demo-empty">
          <div class="alpha-demo-empty__title">Institutional Audit Chat</div>
          <div class="alpha-demo-empty__copy">
            Ask a filing-specific question or trigger a predefined prompt to inspect revenue, risk factors, or AI infrastructure spend with source-backed retrieval.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_prompt_dock():
    quick_queries = QUICK_QUERY_MAP.get(
        st.session_state.selected_report,
        QUICK_QUERY_MAP["Upload Custom PDF..."],
    )
    cols = st.columns(3)
    for col, (key, label, query) in zip(cols, quick_queries):
        with col:
            if st.button(label, key=key, use_container_width=True):
                st.session_state.current_query = query


def _clear_demo_state():
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.doc_stats = None
    st.session_state.index_key = None
    st.session_state.usage_count = 0
    st.session_state.raw_documents = []
    st.session_state.paywall_triggered = False


def _submit_contact_form(name: str, email: str, company: str, message: str) -> Tuple[bool, str]:
    payload = {
        "name": name,
        "email": email,
        "company": company,
        "message": message,
        "source": "alpha-rag-landing",
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        FORMSPREE_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8") if resp.length != 0 else "{}"
            parsed = json.loads(body or "{}")
            if 200 <= resp.status < 300:
                return True, parsed.get("next", "")
            return False, "Form submission failed."
    except error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read().decode("utf-8"))
            errors = parsed.get("errors") or []
            if errors:
                return False, errors[0].get("message", "Form submission failed.")
        except Exception:
            pass
        return False, "Form submission failed."
    except Exception:
        return False, "Form submission failed. Please try again."


def _render_landing():
    theme_cols = st.columns([5, 1.2])
    with theme_cols[1]:
        st.segmented_control(
            "Theme",
            ["Dark", "Light"],
            key="theme_mode",
            label_visibility="collapsed",
        )

    logo_uri = _logo_data_uri()
    brand_markup = (
        f"<div class='alpha-nav__brand'><img class='alpha-nav__logo' src='{logo_uri}' alt='Alpha-RAG logo' /><span>Alpha-RAG</span></div>"
        if logo_uri
        else "<div class='alpha-nav__brand'>Alpha-RAG</div>"
    )
    st.markdown(
        f"""
        <nav class="alpha-nav">
          {brand_markup}
          <div class="alpha-nav__links">
            <a href="#overview">Overview</a>
            <a href="#capabilities">Capabilities</a>
            <a href="#pricing">Pricing</a>
            <a href="#contact">Contact</a>
            <a class="alpha-nav__cta" href="/?view=demo">Open Demo</a>
          </div>
        </nav>
        <section class="alpha-hero" id="overview">
          <div class="alpha-hero-shell">
            <div class="alpha-hero-copy">
              <div class="alpha-eyebrow">Institutional Audit Infrastructure</div>
              <h1 class="alpha-title">Evidence-first financial review for serious teams.</h1>
              <p class="alpha-subtitle">Alpha-RAG gives investment, diligence, and finance operators a clean path from question to filing evidence. The product is built to be professional, easy to navigate, and reliable under review.</p>
              <div class="alpha-actions">
                <a class="alpha-secondary-link" href="#contact">Talk to Sales</a>
              </div>
              <div class="alpha-proof">
                <div class="alpha-proof__item">
                  <div class="alpha-proof__value">No Source</div>
                  <div class="alpha-proof__label">Unsupported claims are blocked instead of presented as answers.</div>
                </div>
                <div class="alpha-proof__item">
                  <div class="alpha-proof__value">Page-Level</div>
                  <div class="alpha-proof__label">Every response can point analysts back to filing evidence.</div>
                </div>
                <div class="alpha-proof__item">
                  <div class="alpha-proof__value">Fast Review</div>
                  <div class="alpha-proof__label">The interface reduces time spent digging through long SEC documents.</div>
                </div>
              </div>
            </div>
            <div class="alpha-hero-panel">
              <div class="alpha-panel-label">Why Teams Buy</div>
              <div class="alpha-panel-title">From document overload to auditable answers.</div>
              <div class="alpha-panel-copy">Replace scattered reading workflows with a single workspace designed for document-backed decisions, quick verification, and executive-ready outputs.</div>
              <ul class="alpha-list">
                <li>Built for diligence teams, equity research, investor relations, and institutional finance operations.</li>
                <li>Clear navigation, clean information density, and sections that are easy to scan on desktop and mobile.</li>
                <li>Professional presentation for prospects who need trust before they request a live product walkthrough.</li>
              </ul>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cta_cols = st.columns([1, 1])
    with cta_cols[0]:
        if st.button("Try Alpha-RAG Live", type="primary", use_container_width=True):
            _set_route(DEMO_ROUTE)
    with cta_cols[1]:
        st.markdown(
            "<div class='alpha-citation-note'>A professional landing page should make the value, the workflow, and the next step obvious within a few seconds.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <section class="alpha-section" id="capabilities">
          <div class="alpha-section-heading">
            <div>
              <div class="alpha-eyebrow" style="margin-bottom:0.4rem;">Core Capabilities</div>
              <h2>Designed for trust, speed, and reviewability.</h2>
            </div>
            <div class="alpha-section-intro">The site now reads like a real product company: one clear menu, structured feature sections, stronger pricing visibility, and a direct contact path for prospects.</div>
          </div>
          <div class="alpha-grid">
            <div class="alpha-card">
              <div class="alpha-card__kicker">Pillar 01</div>
              <div class="alpha-card__title">Zero Hallucination</div>
              <div class="alpha-card__copy">No source means no answer. That standard protects teams from unsupported financial claims entering memos, notes, or recommendations.</div>
            </div>
            <div class="alpha-card">
              <div class="alpha-card__kicker">Pillar 02</div>
              <div class="alpha-card__title">Clickable Citations</div>
              <div class="alpha-card__copy">Answers map back to filing evidence so analysts can validate the exact page context before sharing the output.</div>
            </div>
            <div class="alpha-card">
              <div class="alpha-card__kicker">Pillar 03</div>
              <div class="alpha-card__title">Controlled Security</div>
              <div class="alpha-card__copy">The workflow is framed for sensitive document handling, professional review standards, and institutional expectations.</div>
            </div>
          </div>
          <div class="alpha-feature-band">
            <div class="alpha-band-card">
              <h3>Clean navigation</h3>
              <p>Visitors can move directly to overview, capabilities, pricing, or contact without hunting through the page. That is a basic requirement for a product site that needs to convert interest into conversations.</p>
            </div>
            <div class="alpha-band-card">
              <h3>Better visual balance</h3>
              <p>The color system now uses a refined navy and steel-blue palette with warmer accents, which feels more credible and less generic than a flat dark page with one bright button.</p>
            </div>
          </div>
        </section>
        <section class="alpha-section" id="pricing">
          <div class="alpha-pricing">
            <div>
              <div class="alpha-eyebrow" style="margin-bottom:0.4rem;">Pricing</div>
              <h2 style="margin-top:0;">Professional License</h2>
              <div class="alpha-pricing__price">$149<span style="font-size:1rem;color:#97a6ba;">/month</span></div>
              <p class="alpha-card__copy">A clear pricing block gives prospects enough confidence to qualify themselves before they contact you. This version is easier to scan and ties the fee to the product outcome.</p>
            </div>
            <div class="alpha-pricing__card">
              <div class="alpha-card__kicker">Included</div>
              <ul class="alpha-pricing__list">
                <li>Unlimited SEC document analysis</li>
                <li>Citation-backed audit trail workflow</li>
                <li>Metrics and investment review views</li>
                <li>Professional institutional interface</li>
              </ul>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="alpha-section" id="contact">
          <div class="alpha-section-heading">
            <div>
              <div class="alpha-eyebrow" style="margin-bottom:0.4rem;">Contact</div>
              <h2>Give interested buyers a direct next step.</h2>
            </div>
            <div class="alpha-section-intro">The menu now includes a contact entry, and this section makes it easy for a visitor to request access, book a demo, or start a sales conversation.</div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    contact_cols = st.columns([1.05, 0.95], gap="large")
    with contact_cols[0]:
        st.markdown(
            """
            <div class="alpha-contact-card">
              <div class="alpha-panel-title">Contact the Alpha-RAG team</div>
              <div class="alpha-panel-copy">Use the form to collect inbound interest directly on the site. If you later connect email or a CRM, this section can send leads automatically without redesigning the page.</div>
              <div class="alpha-contact-box">
                <div class="alpha-contact-label">Best For</div>
                <div class="alpha-contact-value">Private demos, pricing questions, and pilot requests</div>
                <div class="alpha-contact-copy">This keeps the site conversion path simple: understand the product, review the price, and contact the team.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with contact_cols[1]:
        st.markdown("<div class='alpha-contact-form'>", unsafe_allow_html=True)
        st.markdown("#### Request Product Access")
        st.caption("Collect interest from visitors directly in the website navigation flow.")
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Full name")
            email = st.text_input("Work email")
            company = st.text_input("Company")
            message = st.text_area(
                "What do you want to know?",
                placeholder="Tell us whether you want a demo, pricing details, or a pilot for your team.",
                height=140,
            )
            submitted = st.form_submit_button("Send Request", use_container_width=True)

        if submitted:
            if name and email and message:
                ok, error_message = _submit_contact_form(name, email, company, message)
                if ok:
                    st.success(f"Thanks {name}. Your request has been submitted.")
                else:
                    st.error(error_message)
            else:
                st.error("Enter your name, work email, and message so the contact request is complete.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_demo_sidebar() -> Tuple[List[str], str | None]:
    with st.sidebar:
        logo_path = BASE_DIR / "logo_vultur.png"
        if logo_path.exists():
            st.image(str(logo_path), width=60)
        st.markdown("<div class='alpha-sidebar-title'>Alpha-RAG</div>", unsafe_allow_html=True)
        st.markdown("<div class='alpha-sidebar-copy'>Institutional Demo Workspace</div>", unsafe_allow_html=True)
        st.divider()

        st.radio(
            "Workspace",
            WORKSPACE_OPTIONS,
            key="current_view",
            horizontal=True,
        )

        st.selectbox(
            "Select Filing",
            REPORT_OPTIONS,
            key="selected_report",
        )

        selected_report = st.session_state.selected_report
        if selected_report == "Upload Custom PDF...":
            uploaded_file = st.file_uploader(
                "Upload Custom PDF",
                type=["pdf"],
                accept_multiple_files=False,
            )
            if uploaded_file is None:
                return [], None
            file_paths, _ = _persist_upload(uploaded_file)
            return file_paths, _build_index_key(file_paths)

        report_path = REPORT_PATHS[selected_report]
        if not report_path.exists():
            st.error(f"Missing demo filing: {report_path.name}")
            return [], None
        file_paths = [str(report_path)]
        return file_paths, _build_index_key(file_paths)


def _ensure_demo_chain(file_paths: List[str], index_key: str | None):
    chain = None
    if file_paths and index_key:
        try:
            processor = DocumentProcessor()
            index_dir = str(Path(APP.index_root) / index_key)
            if st.session_state.index_key != index_key:
                result = processor.process_pdfs(file_paths, index_dir=index_dir)
                st.session_state.vectorstore = result.vectorstore
                st.session_state.doc_stats = (result.doc_count, result.chunk_count)
                st.session_state.index_key = index_key
                st.session_state.raw_documents = result.documents or processor.load_pdfs(file_paths)
                st.session_state.messages = []
                st.session_state.paywall_triggered = False
            elif not st.session_state.raw_documents:
                st.session_state.raw_documents = processor.load_pdfs(file_paths)
            engine = LLMEngine()
            chain = engine.build_qa_chain(st.session_state.vectorstore)
        except DocumentProcessorError as exc:
            st.error(str(exc))
    else:
        st.session_state.vectorstore = None
        st.session_state.doc_stats = None
        st.session_state.index_key = None
        st.session_state.raw_documents = []
    return chain


def _extract_json_block(raw_text: str) -> Dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))


def _run_structured_prompt(chain: object, prompt: str) -> Tuple[Dict, List[str]]:
    result = chain({"query": prompt})
    payload = _extract_json_block(result.get("result", "{}"))
    sources = _build_audit_trail(result.get("source_documents", []))
    return payload, sources


def _metric_prompt(label: str, question: str) -> str:
    return f"""
Use only the filing context to answer this metric question.
Return valid JSON only with this exact schema:
{{
  "label": "{label}",
  "value": "...",
  "trend": "improving|stable|watch",
  "note": "one short sentence"
}}
Rules:
- Prefer official statement tables over narrative text.
- If the figure is ambiguous, set value to "Not clearly disclosed".
- Set trend to "watch" whenever the number is missing, ambiguous, or deteriorating.
- Set trend to "improving" only when the filing context clearly supports improvement.
- Otherwise set trend to "stable".
Question: {question}
""".strip()


def _build_structured_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Alpha-RAG, a senior financial analysis assistant. Use only the provided context. "
                "Be precise, conservative, and prefer official statement tables.",
            ),
            ("human", "{question}\n\nContext:\n{context}"),
        ]
    )


def _run_metric_prompt(vectorstore, label: str, question: str) -> Tuple[Dict, List[str]]:
    engine = LLMEngine()
    metric_chain = engine.build_qa_chain(vectorstore, prompt=_build_structured_prompt(), search_k=14)
    try:
        payload, sources = _run_structured_prompt(metric_chain, _metric_prompt(label, question))
    except Exception:
        payload = {
            "label": label,
            "value": "Not clearly disclosed",
            "trend": "watch",
            "note": "Alpha-RAG could not confidently extract this metric from the retrieved context.",
        }
        sources = []
    payload["label"] = label
    return payload, sources


def _page_num(doc) -> int:
    meta = doc.metadata or {}
    page = meta.get("page", 0)
    return (page + 1) if isinstance(page, int) else 0


def _source_line(doc) -> str:
    meta = doc.metadata or {}
    source = meta.get("document") or meta.get("source", "unknown")
    return f"- {source} (page {_page_num(doc)})"


def _collect_candidate_pages(documents: List, keywords: List[str], limit: int = 4) -> List:
    scored = []
    keywords_lower = [k.lower() for k in keywords]
    for doc in documents or []:
        text = (doc.page_content or "").lower()
        score = sum(1 for keyword in keywords_lower if keyword in text)
        if score > 0:
            scored.append((score, _page_num(doc), doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _render_context_block(documents: List) -> str:
    blocks = []
    for doc in documents:
        blocks.append(f"[Page {_page_num(doc)}]\n{doc.page_content[:4500]}")
    return "\n\n".join(blocks)


def _build_metric_context(documents: List, label: str) -> Tuple[str, List[str]]:
    candidates = _collect_candidate_pages(documents, METRIC_KEYWORDS.get(label, [label]), limit=4)
    if not candidates:
        return "", []
    return _render_context_block(candidates), [_source_line(doc) for doc in candidates]


def _build_risk_context(documents: List) -> Tuple[str, List[str]]:
    candidates = _collect_candidate_pages(documents, RISK_KEYWORDS, limit=4)
    if not candidates:
        return "", []
    return _render_context_block(candidates), [_source_line(doc) for doc in candidates]


def _run_direct_prompt(prompt: str) -> Dict:
    engine = LLMEngine()
    llm = engine.primary_llm or engine.fallback_llm
    if llm is None:
        raise RuntimeError("No LLM provider is configured.")
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_json_block(content)


def _run_metric_from_documents(documents: List, label: str, question: str) -> Tuple[Dict, List[str]]:
    context, sources = _build_metric_context(documents, label)
    if not context:
        return {
            "label": label,
            "value": "Not clearly disclosed",
            "trend": "watch",
            "note": "Alpha-RAG could not find a relevant statement page for this metric.",
        }, []
    prompt = f"""
Use only the filing pages below to answer this metric question.
Return valid JSON only with this exact schema:
{{
  "label": "{label}",
  "value": "...",
  "trend": "improving|stable|watch",
  "note": "one short sentence"
}}
Rules:
- Prefer exact figures from statement tables.
- Include units if shown, such as million or billion.
- If the figure is unclear, set value to "Not clearly disclosed".
- Use "watch" for missing, deteriorating, or materially concerning signals.
- Use "improving" only when the pages clearly support that conclusion.
- Otherwise use "stable".
Question: {question}

Filing pages:
{context}
""".strip()
    try:
        payload = _run_direct_prompt(prompt)
    except Exception:
        payload = {
            "label": label,
            "value": "Not clearly disclosed",
            "trend": "watch",
            "note": "Alpha-RAG could not confidently extract this metric from the selected filing pages.",
        }
    payload["label"] = label
    return payload, sources


def _compute_investment_from_metrics(metric_payloads: List[Dict], documents: List) -> Tuple[Dict, List[str]]:
    bullish = []
    risks = []
    score = 50

    for metric in metric_payloads:
        label = metric.get("label", "Metric")
        trend = (metric.get("trend") or "").lower()
        note = metric.get("note", "")
        value = metric.get("value", "Not clearly disclosed")
        if value != "Not clearly disclosed" and trend == "improving":
            bullish.append(f"{label}: {value}. {note}".strip())
            score += 8
        elif value != "Not clearly disclosed" and trend == "stable":
            bullish.append(f"{label}: {value}.")
            score += 3
        else:
            risks.append(f"{label}: {note or 'Not clearly disclosed from the filing pages reviewed.'}")
            score -= 6

    risk_context, risk_sources = _build_risk_context(documents)
    risk_summary = ""
    if risk_context:
        risk_prompt = f"""
Use only the filing pages below.
Return valid JSON only:
{{
  "summary": "one or two sentences",
  "risk_signals": ["...", "...", "..."]
}}
Summarize the most important risk factor signals for an institutional analyst.

Filing pages:
{risk_context}
""".strip()
        try:
            risk_payload = _run_direct_prompt(risk_prompt)
            risk_summary = risk_payload.get("summary", "")
            risks.extend(risk_payload.get("risk_signals", []))
        except Exception:
            pass

    score = max(15, min(85, score))
    if score >= 70:
        stance = "Constructive"
    elif score >= 40:
        stance = "Balanced"
    else:
        stance = "Cautious"

    if not bullish:
        bullish = ["The filing does not yet show enough clearly favorable evidence across the extracted metrics."]
    if not risks:
        risks = ["Risk disclosure did not surface a concentrated concern in the reviewed pages."]

    rationale = risk_summary or "This view is based on extracted filing metrics, cash generation, balance sheet strength, and document-level risk disclosure."
    verdict = (
        "The filing supports constructive follow-up work."
        if stance == "Constructive"
        else "The filing looks investable but still needs analyst follow-up."
        if stance == "Balanced"
        else "The filing presents enough uncertainty that further diligence is required before a constructive view."
    )
    payload = {
        "stance": stance,
        "score": score,
        "rationale": rationale,
        "bullish_signals": bullish[:4],
        "risk_signals": risks[:4],
        "verdict": verdict,
    }
    return payload, risk_sources


def _compute_document_insights(index_key: str, vectorstore, documents: List) -> Dict:
    cached = st.session_state.insight_cache.get(index_key)
    if cached:
        return cached

    metric_payloads = []
    metric_sources = []
    for label, question in METRIC_SPECS:
        payload, sources = _run_metric_from_documents(documents, label, question)
        metric_payloads.append(payload)
        metric_sources.extend(sources)

    confirmed_metrics = [m for m in metric_payloads if m.get("value") != "Not clearly disclosed"]
    improving = [m.get("label", "Metric") for m in metric_payloads if m.get("trend") == "improving"]
    watch_items = [m.get("note", "") for m in metric_payloads if m.get("trend") == "watch" and m.get("note")]
    if not watch_items and not confirmed_metrics:
        watch_items = ["Metrics extraction needs a narrower retrieval context."]

    metrics_payload = {
        "headline": f"{len(confirmed_metrics)} key figures extracted from the filing" if confirmed_metrics else "Metrics unavailable",
        "summary": "Alpha-RAG assembled this panel from filing statement pages and targeted document extraction." if confirmed_metrics else "Alpha-RAG could not confidently assemble a metrics dashboard from the filing pages reviewed.",
        "metrics": metric_payloads,
        "strengths": [f"{label} appears favorable based on the retrieved filing context." for label in improving[:3]],
        "watch_items": watch_items[:3],
    }

    try:
        investment_payload, investment_sources = _compute_investment_from_metrics(metric_payloads, documents)
    except Exception:
        investment_payload = {
            "stance": "Balanced",
            "score": 50,
            "rationale": "Alpha-RAG could not confidently derive a document-based investment signal from the filing pages reviewed.",
            "bullish_signals": [],
            "risk_signals": ["Review the filing with narrower follow-up prompts."],
            "verdict": "Insufficient evidence from the current filing review.",
        }
        investment_sources = []

    insight = {
        "metrics": metrics_payload,
        "metrics_sources": list(dict.fromkeys(metric_sources)),
        "investment": investment_payload,
        "investment_sources": investment_sources,
    }
    st.session_state.insight_cache[index_key] = insight
    return insight


def _trend_color(trend: str) -> str:
    return {
        "improving": "#2fd07d",
        "stable": "#3ea6ff",
        "watch": "#ffb01f",
    }.get((trend or "").lower(), "#97a6ba")


def _render_metrics_view(insight: Dict):
    metrics_data = insight.get("metrics", {})
    st.markdown("### Filing Metrics")
    st.caption(metrics_data.get("headline", "Document metrics"))
    st.markdown(metrics_data.get("summary", ""))

    metrics = metrics_data.get("metrics", [])
    rows = [metrics[i : i + 3] for i in range(0, min(len(metrics), 6), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, metric in zip(cols, row):
            color = _trend_color(metric.get("trend", "stable"))
            with col:
                st.markdown(
                    f"""
                    <div class="alpha-metric-tile">
                      <div class="alpha-metric-label">{metric.get('label', 'Metric')}</div>
                      <div class="alpha-metric-value">{metric.get('value', 'N/A')}</div>
                      <div class="alpha-metric-trend" style="color:{color};">{metric.get('trend', 'stable').upper()}</div>
                      <div class="alpha-metric-note">{metric.get('note', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Strengths**")
        strengths = metrics_data.get("strengths", [])
        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.caption("No strengths were confidently extracted.")
    with detail_cols[1]:
        st.markdown("**Watch Items**")
        watch_items = metrics_data.get("watch_items", [])
        if watch_items:
            for item in watch_items:
                st.markdown(f"- {item}")
        else:
            st.caption("No watch items were confidently extracted.")

    _render_sources(insight.get("metrics_sources", []))


def _render_investment_view(insight: Dict):
    investment = insight.get("investment", {})
    score = max(0, min(100, int(investment.get("score", 50))))
    st.markdown("### Investment Readiness")
    st.markdown(
        f"""
        <div class="alpha-demo-panel" style="margin-bottom:1rem;">
          <div style="color:#97a6ba;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;">{investment.get('stance', 'Balanced')}</div>
          <div style="font-size:2rem;font-weight:800;margin-top:0.35rem;">{score}/100</div>
          <div style="height:10px;background:#1b2430;border-radius:999px;overflow:hidden;margin:0.8rem 0 1rem;">
            <div style="width:{score}%;height:100%;background:linear-gradient(90deg,#007BFF 0%,#42a5ff 100%);"></div>
          </div>
          <div style="color:#eef3fb;line-height:1.65;">{investment.get('rationale', '')}</div>
          <div style="color:#97a6ba;margin-top:0.7rem;">{investment.get('verdict', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Bullish Signals**")
        bullish = investment.get("bullish_signals", [])
        if bullish:
            for item in bullish:
                st.markdown(f"- {item}")
        else:
            st.caption("No positive signals were confidently extracted.")
    with cols[1]:
        st.markdown("**Risk Signals**")
        risks = investment.get("risk_signals", [])
        if risks:
            for item in risks:
                st.markdown(f"- {item}")
        else:
            st.caption("No risk signals were confidently extracted.")

    _render_sources(insight.get("investment_sources", []))


def _render_paywall_overlay():
    st.markdown(
        f"""
        <div class="alpha-overlay">
          <div class="alpha-overlay__card">
            <div class="alpha-overlay__title">Institutional Trial Limit Reached.</div>
            <div class="alpha-overlay__copy">
              To continue auditing with Absolute Traceability, upgrade to a Professional License.
            </div>
            <a class="alpha-overlay__button" href="{PRICING_URL}">View Pricing &amp; Plans</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_demo(chain: object | None):
    st.markdown("<div class='alpha-demo-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='alpha-demo-header'><h1 style='margin-bottom:0;'>Alpha-RAG</h1></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='alpha-demo-label'>Demo route: /demo | Trial prompts used: {st.session_state.usage_count}/{TRIAL_LIMIT}</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.doc_stats:
        doc_count, chunk_count = st.session_state.doc_stats
        st.caption(f"Loaded: {doc_count} pages | {chunk_count} chunks")

    insights = None
    if st.session_state.vectorstore is not None and st.session_state.index_key and st.session_state.current_view in {"Metrics", "Investment"}:
        insights = _compute_document_insights(
            st.session_state.index_key,
            st.session_state.vectorstore,
            st.session_state.raw_documents,
        )

    with st.container():
        st.markdown("<div class='alpha-demo-panel'>", unsafe_allow_html=True)
        if st.session_state.current_view == "Metrics":
            if insights is not None:
                _render_metrics_view(insights)
            else:
                st.info("Select a filing in the sidebar to view extracted metrics.")
        elif st.session_state.current_view == "Investment":
            if insights is not None:
                _render_investment_view(insights)
            else:
                st.info("Select a filing in the sidebar to view the investment signal.")
        else:
            _render_chat_intro()
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    _render_sources(message.get("audit_trail", []))

            if not st.session_state.messages:
                st.info("The chat is ready. Use a quick prompt below or ask your own question.")

            _render_prompt_dock()

            prompt = None
            if not st.session_state.paywall_triggered:
                prompt = st.session_state.current_query or st.chat_input("Ask the filing anything...")

            if st.session_state.current_query:
                st.session_state.current_query = None

            if prompt:
                if st.session_state.usage_count >= TRIAL_LIMIT:
                    st.session_state.paywall_triggered = True
                    st.rerun()
                st.session_state.usage_count += 1
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    if chain is None:
                        response = "Select a filing in the sidebar to start the live demo."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        result = chain({"query": prompt})
                        answer = result.get("result", "")
                        audit_trail = _build_audit_trail(result.get("source_documents", []))
                        st.markdown(answer)
                        _render_sources(audit_trail)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "audit_trail": audit_trail,
                            }
                        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.paywall_triggered:
        _render_paywall_overlay()


st.set_page_config(
    page_title="Alpha-RAG",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)
_init_state()
_inject_styles()

route = _get_route()
st.session_state.current_route = route
_browser_shell(route)

missing = validate_env()
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

if route == LANDING_ROUTE:
    _render_landing()
else:
    file_paths, index_key = _render_demo_sidebar()
    chain = _ensure_demo_chain(file_paths, index_key)
    _render_demo(chain)
