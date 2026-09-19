"""Reciprocal Rank Fusion (RRF) for merging ranked lists from different retrievers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ContextItem:
    key: str  # stable id (chunk id or graph path)
    kind: str  # "doc" or "graph"
    text: str
    citation: str
    raw_score: float = 0.0  # retriever's own score (cosine similarity for docs)
    score: float = 0.0  # fused RRF score
    ref: str = ""  # prompt reference, e.g. D1 / G2

    def to_dict(self) -> dict:
        return asdict(self)


def reciprocal_rank_fusion(ranked_lists: list[list[ContextItem]], k: int = 60, top_n: int = 10) -> list[ContextItem]:
    """score(item) = Σ 1 / (k + rank). Robust because it ignores incomparable raw scores."""
    scores: dict[str, float] = {}
    items: dict[str, ContextItem] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item.key] = scores.get(item.key, 0.0) + 1.0 / (k + rank)
            items.setdefault(item.key, item)
    fused = sorted(items.values(), key=lambda it: -scores[it.key])[:top_n]
    for item in fused:
        item.score = round(scores[item.key], 6)
    return fused


def assign_refs(items: list[ContextItem]) -> list[ContextItem]:
    counters = {"doc": 0, "graph": 0}
    for item in items:
        counters[item.kind] += 1
        item.ref = f"{'D' if item.kind == 'doc' else 'G'}{counters[item.kind]}"
    return items
