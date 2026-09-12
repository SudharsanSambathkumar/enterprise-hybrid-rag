"""A small, dependency-light vector index (NumPy cosine similarity, persisted to disk).

Good for thousands of chunks. For millions, swap this class for pgvector,
Azure AI Search, Qdrant or Chroma: only `build`, `search`, `save` and `load`
are used by the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .acl import can_access
from .chunking import Chunk


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))

    @classmethod
    def build(cls, chunks: list[Chunk], llm) -> VectorStore:
        if not chunks:
            return cls([], np.zeros((0, 1), dtype=np.float32))
        return cls(chunks, llm.embed([c.embedding_text() for c in chunks]))

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self, query_vector: np.ndarray, k: int = 4, roles: Iterable[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, after the role-based access filter."""
        if not self.chunks:
            return []
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(-1))
        scores = self.embeddings @ query
        results: list[tuple[Chunk, float]] = []
        for idx in np.argsort(-scores):
            chunk = self.chunks[int(idx)]
            if not can_access(chunk.access, roles):
                continue
            results.append((chunk, float(scores[idx])))
            if len(results) >= k:
                break
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.embeddings)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        directory = Path(directory)
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text("utf-8"))]
        return cls(chunks, np.load(directory / "vectors.npy"))
