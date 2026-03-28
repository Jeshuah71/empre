from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import APP, LLM


@dataclass
class VectorStoreResult:
    vectorstore: FAISS
    documents: List
    chunks: List
    doc_count: int
    chunk_count: int


class DocumentProcessorError(RuntimeError):
    pass


class DocumentProcessor:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or self._build_embeddings()

    def _build_openai_embeddings(self):
        try:
            return OpenAIEmbeddings(model=LLM.openai_embedding_model)
        except Exception as exc:
            raise DocumentProcessorError(
                "OpenAI embeddings are unavailable. Set OPENAI_API_KEY or configure another embeddings provider."
            ) from exc

    def _build_local_embeddings(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as exc:
            raise DocumentProcessorError(
                "Local embeddings are unavailable. Install sentence-transformers or set EMBEDDINGS_PROVIDER=google."
            ) from exc
        return HuggingFaceEmbeddings(model_name=LLM.local_embedding_model)

    def _build_embeddings(self):
        provider = (LLM.embeddings_provider or "").lower()
        if provider == "openai":
            return self._build_openai_embeddings()
        if provider == "local":
            return self._build_local_embeddings()
        try:
            return GoogleGenerativeAIEmbeddings(model=LLM.embedding_model)
        except Exception:
            try:
                return self._build_openai_embeddings()
            except Exception:
                return self._build_local_embeddings()

    def load_pdf(self, file_path: str):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            filename = Path(file_path).name
            match = re.match(r"^[0-9a-f]{12}_(.+)$", filename)
            display_name = match.group(1) if match else filename
            for doc in documents:
                meta = doc.metadata or {}
                meta["document"] = display_name
                meta["source"] = display_name
                doc.metadata = meta
            return documents
        except Exception as exc:
            raise DocumentProcessorError(f"Failed to load PDF: {exc}") from exc

    def load_pdfs(self, file_paths: List[str]):
        documents = []
        for path in file_paths:
            documents.extend(self.load_pdf(path))
        return documents

    def split_documents(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=APP.chunk_size,
                chunk_overlap=APP.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            return splitter.split_documents(documents)
        except Exception as exc:
            raise DocumentProcessorError(f"Failed to split document: {exc}") from exc

    def build_vectorstore(self, chunks) -> FAISS:
        try:
            return FAISS.from_documents(chunks, self.embeddings)
        except Exception:
            try:
                openai_embeddings = self._build_openai_embeddings()
                return FAISS.from_documents(chunks, openai_embeddings)
            except Exception:
                try:
                    local_embeddings = self._build_local_embeddings()
                    return FAISS.from_documents(chunks, local_embeddings)
                except Exception as exc:
                    raise DocumentProcessorError(f"Failed to build vector store: {exc}") from exc

    def save_vectorstore(self, vectorstore: FAISS, index_dir: str, doc_count: int, chunk_count: int):
        try:
            Path(index_dir).mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(index_dir)
            stats_path = Path(index_dir) / "stats.json"
            stats_path.write_text(json.dumps({"doc_count": doc_count, "chunk_count": chunk_count}))
        except Exception as exc:
            raise DocumentProcessorError(f"Failed to save vector store: {exc}") from exc

    def load_vectorstore(self, index_dir: str) -> Tuple[FAISS, int, int]:
        try:
            vectorstore = FAISS.load_local(
                index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            stats_path = Path(index_dir) / "stats.json"
            doc_count = 0
            chunk_count = 0
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
                doc_count = int(stats.get("doc_count", 0))
                chunk_count = int(stats.get("chunk_count", 0))
            return vectorstore, doc_count, chunk_count
        except Exception as exc:
            raise DocumentProcessorError(f"Failed to load vector store: {exc}") from exc

    def process_pdfs(self, file_paths: List[str], index_dir: str | None = None) -> VectorStoreResult:
        if index_dir:
            index_path = Path(index_dir) / "index.faiss"
            if index_path.exists():
                vectorstore, doc_count, chunk_count = self.load_vectorstore(index_dir)
                return VectorStoreResult(
                    vectorstore=vectorstore,
                    documents=[],
                    chunks=[],
                    doc_count=doc_count,
                    chunk_count=chunk_count,
                )

        documents = self.load_pdfs(file_paths)
        chunks = self.split_documents(documents)
        vectorstore = self.build_vectorstore(chunks)
        doc_count, chunk_count = self.preview_stats(documents, chunks)
        if index_dir:
            self.save_vectorstore(vectorstore, index_dir, doc_count, chunk_count)
        return VectorStoreResult(
            vectorstore=vectorstore,
            documents=documents,
            chunks=chunks,
            doc_count=doc_count,
            chunk_count=chunk_count,
        )

    def preview_stats(self, documents, chunks) -> Tuple[int, int]:
        return (len(documents), len(chunks))
