import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class LLMConfig:
    model: str = _get_env("GEMINI_MODEL", "gemini-2.5-flash")
    temperature: float = 0.0
    max_tokens: int = int(_get_env("GEMINI_MAX_TOKENS", "1024"))
    top_p: float = float(_get_env("GEMINI_TOP_P", "1.0"))
    embedding_model: str = _get_env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    local_embedding_model: str = _get_env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en")
    embeddings_provider: str = _get_env("EMBEDDINGS_PROVIDER", "local")


@dataclass(frozen=True)
class AppConfig:
    app_title: str = _get_env("APP_TITLE", "Alpha-RAG")
    app_subtitle: str = _get_env("APP_SUBTITLE", "Enterprise Financial Document Analysis")
    max_upload_mb: int = int(_get_env("MAX_UPLOAD_MB", "50"))
    chunk_size: int = int(_get_env("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(_get_env("CHUNK_OVERLAP", "150"))
    k_retrieval: int = int(_get_env("K_RETRIEVAL", "6"))
    index_root: str = _get_env("INDEX_ROOT", "data/indices")
    upload_root: str = _get_env("UPLOAD_ROOT", "data/uploads")


def validate_env() -> list[str]:
    missing = []
    if not _get_env("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    return missing


@dataclass(frozen=True)
class UIConfig:
    theme_bg: str = "#0b0f14"
    panel_bg: str = "#0f1720"
    border: str = "#202a33"
    accent: str = "#f7a600"
    accent_dim: str = "#c47f00"
    text: str = "#e6edf3"
    text_dim: str = "#94a3b8"
    good: str = "#26d07c"
    bad: str = "#ff5c5c"
    info: str = "#3ea6ff"
    code_bg: str = "#111821"
    font_stack: str = "'IBM Plex Mono', 'JetBrains Mono', 'SFMono-Regular', Menlo, Monaco, Consolas, monospace"


LLM = LLMConfig()
APP = AppConfig()
UI = UIConfig()
