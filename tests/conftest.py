from pathlib import Path

import pytest

from hybrid_rag import HybridRAG, load_settings
from hybrid_rag.llm import FakeLLM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rag(tmp_path_factory) -> HybridRAG:
    settings = load_settings(
        provider="fake",
        data_dir=ROOT / "data",
        index_dir=tmp_path_factory.mktemp("index"),
    )
    pipeline = HybridRAG(settings, llm=FakeLLM())
    pipeline.ingest()
    return pipeline
