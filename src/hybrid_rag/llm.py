"""LLM + embedding clients.

`OpenAILLM` talks to the OpenAI API (GPT chat model + text-embedding model).
`FakeLLM` is an offline, deterministic stand-in so the whole pipeline, the tests
and CI run without an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

import numpy as np

from .config import Settings


class LLM(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def chat(self, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAILLM:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file, or run offline with LLM_PROVIDER=fake."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.chat_model = settings.chat_model
        self.embed_model = settings.embed_model
        self.temperature = settings.temperature
        self.name = f"openai:{self.embed_model}"

    def embed(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.embed_model, input=texts[start : start + batch_size])
            vectors.extend(item.embedding for item in resp.data)
        return np.asarray(vectors, dtype=np.float32)

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "do",
    "does",
    "did",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "what",
    "how",
    "who",
    "which",
    "can",
    "i",
    "we",
    "my",
    "our",
    "be",
    "with",
    "by",
    "at",
    "it",
    "this",
    "that",
    "get",
    "if",
    "per",
    "from",
    "as",
    "any",
}


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


class FakeLLM:
    """Offline stand-in: hashed bag-of-words embeddings + rule-based 'chat'."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"fake:hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                mat[row, h % self.dim] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        if "query router" in system.lower():
            from .router import heuristic_route

            route, reason = heuristic_route(user)
            return json.dumps({"route": route, "reason": f"{reason} (offline heuristic)"})
        if json_mode:
            return json.dumps({"entities": [], "triples": []})
        # Grounded "answer": surface the top evidence with its references.
        blocks = re.findall(r"^\[(\w+)\] \(([^)]*)\)\n(.+?)(?=\n\n\[|\Z)", user, re.S | re.M)
        if not blocks:
            return "I don't know based on the available context."
        lines = ["(Offline mode: set LLM_PROVIDER=openai for GPT answers.) Top evidence:"]
        for ref, _source, text in blocks[:4]:
            snippet = " ".join(text.split())[:220]
            lines.append(f"- {snippet} [{ref}]")
        return "\n".join(lines)


def get_llm(settings: Settings) -> LLM:
    if settings.provider == "fake":
        return FakeLLM()
    if settings.provider == "openai":
        return OpenAILLM(settings)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.provider}'. Use 'openai' or 'fake'.")


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ```json fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(text[start : end + 1])
