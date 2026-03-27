from __future__ import annotations

from typing import List

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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


class LLMEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM.model,
            temperature=0.0,
            max_output_tokens=LLM.max_tokens,
            top_p=LLM.top_p,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )

    def build_qa_chain(self, vectorstore):
        retriever = vectorstore.as_retriever(search_kwargs={"k": APP.k_retrieval})
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )
        return chain

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
