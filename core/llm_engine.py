from __future__ import annotations

import os
from typing import List

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config.settings import LLM, APP


SYSTEM_PROMPT = """
You are Alpha-RAG, a senior financial analysis assistant.
Use only the provided context to answer. If the context is insufficient, say so.
Always be precise, conservative, and cite sources by filename and page.
If a numeric value appears truncated or ambiguous, explicitly say you cannot confirm it.
Prefer the official statement tables (e.g., income statement, balance sheet, cash flows).
""".strip()

USER_PROMPT = """
Question: {question}

Context:
{context}

Return a clear answer with a short citation list.
Format money values with full digits and units (e.g., "$38,458 million").
""".strip()


class FallbackRetrievalQA:
    def __init__(self, primary_chain, fallback_chain=None):
        self.primary_chain = primary_chain
        self.fallback_chain = fallback_chain

    def __call__(self, inputs):
        try:
            return self.primary_chain(inputs)
        except Exception:
            if self.fallback_chain is None:
                raise
            return self.fallback_chain(inputs)


class LLMEngine:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )
        self.primary_llm = self._build_primary_llm()
        self.fallback_llm = self._build_fallback_llm()

    def _build_primary_llm(self):
        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(
                model=LLM.model,
                temperature=0.0,
                max_output_tokens=LLM.max_tokens,
                top_p=LLM.top_p,
            )
        return self._build_fallback_llm()

    def _build_fallback_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                model=LLM.openai_model,
                temperature=0.0,
                max_tokens=LLM.max_tokens,
            )
        return None

    def _build_chain(self, llm, retriever):
        if llm is None:
            return None
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def build_qa_chain(self, vectorstore, prompt=None, search_k=None):
        retriever = vectorstore.as_retriever(search_kwargs={"k": search_k or APP.k_retrieval})
        chain_prompt = prompt or self.prompt
        primary_chain = self._build_chain(self.primary_llm, retriever)
        fallback_chain = self._build_chain(self.fallback_llm, retriever)
        if primary_chain is not None:
            primary_chain.combine_documents_chain.llm_chain.prompt = chain_prompt
        if fallback_chain is not None:
            fallback_chain.combine_documents_chain.llm_chain.prompt = chain_prompt
        if primary_chain is None and fallback_chain is None:
            raise RuntimeError("No LLM provider is configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
        if primary_chain is None:
            return fallback_chain
        if fallback_chain is None or fallback_chain is primary_chain:
            return primary_chain
        return FallbackRetrievalQA(primary_chain=primary_chain, fallback_chain=fallback_chain)

    @staticmethod
    def format_sources(source_documents: List) -> str:
        if not source_documents:
            return ""
        seen = set()
        lines = []
        for doc in source_documents:
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
        return "\n".join(lines)
