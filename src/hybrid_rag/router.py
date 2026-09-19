"""Query router: decide whether a question needs vector search, graph traversal, or both."""

from __future__ import annotations

import re

from .llm import parse_json

ROUTES = ("vector", "graph", "hybrid")

ROUTER_SYSTEM = """You are a query router for an enterprise RAG system over a supermarket chain.
Pick the retrieval strategy for the user's question:
- "vector": the answer is written in a policy, SOP, manual or FAQ (rules, limits, procedures, "how many", "what is").
- "graph": the answer needs relationships between entities (suppliers, products, stores, managers,
  regions): "which stores", "who manages", "what is affected", "who supplies", multi-hop questions.
- "hybrid": it needs both, e.g. who is affected AND what they must do.
Return ONLY JSON: {"route": "vector|graph|hybrid", "reason": "<one short sentence>"}"""

_GRAPH_CUES = re.compile(
    r"\b(which|who|whom|affected|impact\w*|supplie[sd]?|supplier|stocked|stock(s)? (of|in)|"
    r"manage[sd]?|manager|reports? to|linked|connected|related|depend\w*|where is)\b",
    re.I,
)
_VECTOR_CUES = re.compile(
    r"\b(policy|policies|sop|procedure|process|rule|rules|how many|how much|how long|allowed|"
    r"must|should|steps?|what (is|are) the|limit|eligible|entitled|refund|leave|leaves|checklist|do)\b",
    re.I,
)


def heuristic_route(question: str) -> tuple[str, str]:
    graph, vector = bool(_GRAPH_CUES.search(question)), bool(_VECTOR_CUES.search(question))
    if graph and vector:
        return "hybrid", "Asks about related entities and a procedure/policy"
    if graph:
        return "graph", "Asks about relationships between entities"
    if vector:
        return "vector", "Asks about a policy or procedure"
    return "hybrid", "Unclear intent, so search both"


def route_query(question: str, llm) -> tuple[str, str]:
    try:
        data = parse_json(llm.chat(ROUTER_SYSTEM, question, json_mode=True))
        route = str(data.get("route", "")).lower().strip()
        if route in ROUTES:
            return route, str(data.get("reason", ""))
    except Exception:
        pass
    return heuristic_route(question)
