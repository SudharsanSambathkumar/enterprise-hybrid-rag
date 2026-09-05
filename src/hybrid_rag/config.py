"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    provider: str = "openai"  # "openai" or "fake" (offline, deterministic; used in tests/CI)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    temperature: float | None = 0.0
    data_dir: Path = Path("data")
    index_dir: Path = Path(".index")
    chunk_size: int = 600
    chunk_overlap: int = 120
    top_k: int = 4
    graph_hops: int = 3
    max_graph_facts: int = 12
    max_context_items: int = 10

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"


def load_settings(**overrides) -> Settings:
    """Build Settings from the environment. Keyword overrides win over env vars."""
    temp = _env("OPENAI_TEMPERATURE", "0")
    values = dict(
        provider=(_env("LLM_PROVIDER", "openai") or "openai").lower(),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        # Some reasoning models reject `temperature`; set OPENAI_TEMPERATURE=none to omit it.
        temperature=None if str(temp).lower() == "none" else float(temp),
        data_dir=Path(_env("DATA_DIR", "data")),
        index_dir=Path(_env("INDEX_DIR", ".index")),
        chunk_size=int(_env("CHUNK_SIZE", "600")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "120")),
        top_k=int(_env("TOP_K", "4")),
        graph_hops=int(_env("GRAPH_HOPS", "3")),
        max_graph_facts=int(_env("MAX_GRAPH_FACTS", "12")),
        max_context_items=int(_env("MAX_CONTEXT_ITEMS", "10")),
    )
    values.update(overrides)
    return Settings(**values)
