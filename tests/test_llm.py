from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_rag import load_settings
from hybrid_rag.llm import FakeLLM, OpenAILLM, get_llm


def test_openai_requires_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm(load_settings(provider="openai", openai_api_key=None))


def test_openai_client_calls(monkeypatch):
    llm = OpenAILLM(load_settings(provider="openai", openai_api_key="sk-test", chat_model="gpt-test"))
    calls = {}

    def fake_create(**kwargs):
        calls["chat"] = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"route": "graph"}'))])

    def fake_embed(**kwargs):
        calls.setdefault("embed", []).append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in kwargs["input"]])

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm.client.embeddings, "create", fake_embed)

    assert llm.chat("sys", "user", json_mode=True) == '{"route": "graph"}'
    assert calls["chat"]["model"] == "gpt-test"
    assert calls["chat"]["response_format"] == {"type": "json_object"}
    assert calls["chat"]["temperature"] == 0.0

    vecs = llm.embed([f"t{i}" for i in range(150)], batch_size=96)
    assert vecs.shape == (150, 2)
    assert [len(c["input"]) for c in calls["embed"]] == [96, 54]


def test_temperature_can_be_omitted(monkeypatch):
    llm = OpenAILLM(load_settings(provider="openai", openai_api_key="sk-test", temperature=None))
    seen = {}
    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        lambda **kw: (
            seen.update(kw) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        ),
    )
    llm.chat("s", "u")
    assert "temperature" not in seen


def test_fake_embeddings_are_normalized_and_deterministic():
    llm = FakeLLM()
    a, b = llm.embed(["sick leave policy", "sick leave policy"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
