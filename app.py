from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

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
WORKSPACE_OPTIONS = ["Chat", "Metrics", "Investment"]
QUICK_QUERIES = [
    ("quick_yoy_revenue", "📈 YoY Revenue", "Summarize YoY Revenue Growth (Azure/Cloud)"),
    ("quick_risk_factors", "⚠️ Risk Factors", "Analyze Item 1A: Risk Factors"),
    ("quick_ai_capex", "🏗️ AI CapEx", "Audit CapEx vs. AI Infrastructure Spend"),
]
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


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto:wght@400;500;700&display=swap');

        :root {
          --alpha-bg: #0b0f14;
          --alpha-panel: #121821;
          --alpha-panel-2: #171f2b;
          --alpha-border: #1f2a37;
          --alpha-text: #eef3fb;
          --alpha-dim: #97a6ba;
          --alpha-blue: #007BFF;
          --alpha-blue-soft: rgba(0, 123, 255, 0.18);
        }

        html, body, [class*="css"] {
          font-family: 'Inter', 'Roboto', sans-serif;
          background: var(--alpha-bg);
          color: var(--alpha-text);
        }

        .stApp {
          background:
            radial-gradient(1100px 700px at 10% 0%, rgba(0, 123, 255, 0.12) 0%, rgba(0, 123, 255, 0) 52%),
            linear-gradient(180deg, #0b0f14 0%, #0c1118 100%);
        }

        .block-container {
          max-width: 1180px;
          padding-top: 2rem;
          padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #151a22 0%, #11161e 100%);
          border-right: 1px solid rgba(255,255,255,0.05);
        }

        [data-testid="stSidebar"] .block-container {
          padding-top: 1.25rem;
        }

        h1, h2, h3 {
          color: var(--alpha-text);
          letter-spacing: -0.02em;
        }

        .alpha-hero {
          padding: 4rem 0 3rem;
        }

        .alpha-eyebrow {
          color: var(--alpha-blue);
          font-size: 0.86rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 1rem;
        }

        .alpha-title {
          font-size: clamp(2.6rem, 6vw, 4.8rem);
          line-height: 0.96;
          font-weight: 800;
          max-width: 820px;
          margin: 0;
        }

        .alpha-subtitle {
          max-width: 680px;
          color: var(--alpha-dim);
          font-size: 1.12rem;
          line-height: 1.7;
          margin-top: 1.25rem;
        }

        .alpha-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          margin-top: 2rem;
        }

        .alpha-card {
          background: linear-gradient(180deg, rgba(22,29,40,0.96) 0%, rgba(17,23,32,0.96) 100%);
          border: 1px solid var(--alpha-border);
          border-radius: 18px;
          padding: 1.2rem;
          min-height: 190px;
          box-shadow: 0 22px 50px rgba(0, 0, 0, 0.26);
        }

        .alpha-card__kicker {
          color: var(--alpha-blue);
          font-size: 0.8rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .alpha-card__title {
          font-size: 1.2rem;
          font-weight: 700;
          margin-top: 0.85rem;
        }

        .alpha-card__copy {
          color: var(--alpha-dim);
          line-height: 1.7;
          margin-top: 0.85rem;
          font-size: 0.96rem;
        }

        .alpha-section {
          margin-top: 4rem;
        }

        .alpha-pricing {
          background: linear-gradient(180deg, rgba(18,24,33,0.96) 0%, rgba(14,19,27,0.96) 100%);
          border: 1px solid var(--alpha-border);
          border-radius: 22px;
          padding: 1.5rem;
        }

        .alpha-pricing__price {
          color: var(--alpha-blue);
          font-size: 2.4rem;
          font-weight: 800;
        }

        .alpha-demo-shell {
          padding-top: 0.25rem;
        }

        .alpha-demo-header {
          margin-bottom: 1.25rem;
        }

        .alpha-demo-label {
          color: var(--alpha-dim);
          font-size: 0.88rem;
          margin-top: 0.35rem;
        }

        .alpha-demo-panel {
          background: linear-gradient(180deg, rgba(18,24,33,0.96) 0%, rgba(16,21,30,0.96) 100%);
          border: 1px solid var(--alpha-border);
          border-radius: 18px;
          padding: 1rem;
        }

        .alpha-demo-empty {
          border: 1px dashed rgba(0,123,255,0.22);
          background: linear-gradient(180deg, rgba(12,17,24,0.9) 0%, rgba(10,15,21,0.9) 100%);
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
          background: rgba(18,24,33,0.88);
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
          background: rgba(255,255,255,0.04);
          border: 1px solid var(--alpha-border);
          border-radius: 999px;
          padding: 0.3rem 0.8rem;
        }

        .alpha-metric-tile {
          background: linear-gradient(180deg, rgba(18,24,33,0.96) 0%, rgba(16,21,30,0.96) 100%);
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
          background: var(--alpha-blue);
          color: white;
          padding: 0.85rem 1.2rem;
          border-radius: 999px;
          font-weight: 700;
        }

        .alpha-citation-note {
          color: var(--alpha-dim);
          font-size: 0.92rem;
        }

        .stButton > button, .stLinkButton > a {
          border-radius: 999px !important;
          min-height: 2.9rem;
        }

        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #007BFF 0%, #1d93ff 100%);
          color: #fff;
          border: 1px solid rgba(255,255,255,0.08);
        }

        @media (max-width: 1024px) {
          .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
          }

          .alpha-grid {
            grid-template-columns: 1fr;
          }

          .alpha-overlay {
            left: 0;
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    cols = st.columns(3)
    for col, (key, label, query) in zip(cols, QUICK_QUERIES):
        with col:
            if st.button(label, key=key, use_container_width=True):
                st.session_state.current_query = query


def _clear_demo_state():
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.doc_stats = None
    st.session_state.index_key = None
    st.session_state.usage_count = 0


def _render_landing():
    st.markdown(
        """
        <section class="alpha-hero">
          <div class="alpha-eyebrow">Institutional Dark Audit Layer</div>
          <h1 class="alpha-title">Absolute Traceability for Institutional Finance.</h1>
          <p class="alpha-subtitle">Stop reading 200-page 10-Ks. Start auditing with zero hallucinations.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cta_cols = st.columns([1, 2])
    with cta_cols[0]:
        if st.button("Try Alpha-RAG Live", type="primary", use_container_width=True):
            _set_route(DEMO_ROUTE)
    with cta_cols[1]:
        st.markdown(
            "<div class='alpha-citation-note'>Built for diligence teams, equity research, and institutional finance operations that require evidence first.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="alpha-grid">
          <div class="alpha-card">
            <div class="alpha-card__kicker">Pillar 01</div>
            <div class="alpha-card__title">Zero Hallucination</div>
            <div class="alpha-card__copy">No Source = No Answer. Alpha-RAG is designed to stop unsupported financial claims before they enter your workflow.</div>
          </div>
          <div class="alpha-card">
            <div class="alpha-card__kicker">Pillar 02</div>
            <div class="alpha-card__title">Clickable Citations</div>
            <div class="alpha-card__copy">Every answer carries an auditable trail back to the filing context so analysts can verify the exact page-level evidence.</div>
          </div>
          <div class="alpha-card">
            <div class="alpha-card__kicker">Pillar 03</div>
            <div class="alpha-card__title">Bank-Grade Security</div>
            <div class="alpha-card__copy">A controlled, isolated pipeline architecture built for sensitive document workflows and professional review standards.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="alpha-section" id="pricing">
          <div class="alpha-pricing">
            <div class="alpha-card__kicker">Pricing</div>
            <h2 style="margin-top:0.6rem;">Professional License</h2>
            <div class="alpha-pricing__price">$149<span style="font-size:1rem;color:#97a6ba;">/month</span></div>
            <p class="alpha-card__copy">Unlimited SEC document analysis, full citation-backed audit trails, and a professional-grade institutional review workflow.</p>
          </div>
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
                st.session_state.messages = []
            engine = LLMEngine()
            chain = engine.build_qa_chain(st.session_state.vectorstore)
        except DocumentProcessorError as exc:
            st.error(str(exc))
    else:
        st.session_state.vectorstore = None
        st.session_state.doc_stats = None
        st.session_state.index_key = None
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


def _compute_document_insights(index_key: str, vectorstore) -> Dict:
    cached = st.session_state.insight_cache.get(index_key)
    if cached:
        return cached

    metric_payloads = []
    metric_sources = []
    for label, question in METRIC_SPECS:
        payload, sources = _run_metric_prompt(vectorstore, label, question)
        metric_payloads.append(payload)
        metric_sources.extend(sources)

    confirmed_metrics = [m for m in metric_payloads if m.get("value") != "Not clearly disclosed"]
    improving = [m.get("label", "Metric") for m in metric_payloads if m.get("trend") == "improving"]
    watch_items = [m.get("note", "") for m in metric_payloads if m.get("trend") == "watch" and m.get("note")]
    if not watch_items and not confirmed_metrics:
        watch_items = ["Metrics extraction needs a narrower retrieval context."]

    metrics_payload = {
        "headline": f"{len(confirmed_metrics)} key figures extracted from the filing" if confirmed_metrics else "Metrics unavailable",
        "summary": "Alpha-RAG assembled this panel from filing-specific retrieval across financial statements." if confirmed_metrics else "Alpha-RAG could not confidently assemble a metrics dashboard from the retrieved filing context.",
        "metrics": metric_payloads,
        "strengths": [f"{label} appears favorable based on the retrieved filing context." for label in improving[:3]],
        "watch_items": watch_items[:3],
    }

    engine = LLMEngine()
    investment_chain = engine.build_qa_chain(vectorstore, prompt=_build_structured_prompt(), search_k=14)
    try:
        investment_payload, investment_sources = _run_structured_prompt(investment_chain, INVESTMENT_PROMPT)
    except Exception:
        investment_payload = {
            "stance": "Balanced",
            "score": 50,
            "rationale": "Alpha-RAG could not confidently derive a document-based investment signal from the retrieved context.",
            "bullish_signals": [],
            "risk_signals": ["Review the filing with narrower follow-up prompts."],
            "verdict": "Insufficient evidence from the current retrieval context.",
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
        insights = _compute_document_insights(st.session_state.index_key, st.session_state.vectorstore)

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
            if st.session_state.usage_count < TRIAL_LIMIT:
                prompt = st.session_state.current_query or st.chat_input("Ask the filing anything...")

            if st.session_state.current_query:
                st.session_state.current_query = None

            if prompt:
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

    if st.session_state.usage_count >= TRIAL_LIMIT:
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
