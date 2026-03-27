from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from config.settings import APP, validate_env
from core.document_processor import DocumentProcessor, DocumentProcessorError
from core.llm_engine import LLMEngine
from components.ui_blocks import inject_css, render_sidebar, render_metrics


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = None
    if "index_key" not in st.session_state:
        st.session_state.index_key = None


def _persist_uploads(uploaded_files) -> Tuple[List[str], str]:
    upload_root = Path(APP.upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    file_paths = []
    hashes = []
    for uf in uploaded_files:
        data = uf.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()
        hashes.append(file_hash)
        safe_name = f"{file_hash[:12]}_{uf.name}"
        path = upload_root / safe_name
        if not path.exists():
            path.write_bytes(data)
        file_paths.append(str(path))

    combined = hashlib.sha256(("|".join(hashes)).encode("utf-8")).hexdigest()
    return file_paths, combined


def _render_chat(state: Dict, chain: object | None):
    for message in state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources**")
                st.markdown(message["sources"])

    if prompt := st.chat_input("Ask about the financials..."):
        state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if chain is None:
                response = "Upload a PDF to begin analysis."
                st.markdown(response)
                state["messages"].append({"role": "assistant", "content": response})
                return

            result = chain({"query": prompt})
            answer = result.get("result", "")
            sources = LLMEngine.format_sources(result.get("source_documents", []))
            st.markdown(answer)
            if sources:
                st.markdown("**Sources**")
                st.markdown(sources)
            state["messages"].append({"role": "assistant", "content": answer, "sources": sources})


st.set_page_config(page_title=APP.app_title, layout="wide")
_init_state()

inject_css()

st.markdown(f"# {APP.app_title}")
st.caption(APP.app_subtitle)

render_metrics()

missing = validate_env()
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

uploaded = render_sidebar(st.session_state)

chain = None
if uploaded:
    try:
        processor = DocumentProcessor()
        if isinstance(uploaded, list):
            file_paths, index_key = _persist_uploads(uploaded)
        else:
            file_paths, index_key = _persist_uploads([uploaded])

        index_dir = str(Path(APP.index_root) / index_key)
        if st.session_state.index_key != index_key:
            result = processor.process_pdfs(file_paths, index_dir=index_dir)
            st.session_state.vectorstore = result.vectorstore
            st.session_state.doc_stats = (result.doc_count, result.chunk_count)
            st.session_state.index_key = index_key
    except DocumentProcessorError as exc:
        st.error(str(exc))

if st.session_state.vectorstore is not None:
    engine = LLMEngine()
    chain = engine.build_qa_chain(st.session_state.vectorstore)

if st.session_state.doc_stats:
    doc_count, chunk_count = st.session_state.doc_stats
    st.markdown(f"**Loaded**: {doc_count} pages | {chunk_count} chunks")

st.divider()
_render_chat(st.session_state, chain)
