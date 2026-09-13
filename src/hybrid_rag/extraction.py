"""Optional: use GPT to extract (entity, relation, entity) triples from documents
and merge them into the knowledge graph ("Extract Graph" lane)."""

from __future__ import annotations

import logging

from .chunking import Document
from .graph_store import KnowledgeGraph
from .llm import parse_json

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = """You extract a knowledge graph from enterprise documents.
Return ONLY a JSON object:
{{"triples": [{{"source": "...", "source_type": "...", "relation": "UPPER_SNAKE_CASE",
               "target": "...", "target_type": "..."}}]}}
Rules:
- Only include facts explicitly stated in the document. No guessing.
- Prefer types: Supplier, Product, Category, Store, Manager, RegionalManager, Role, Policy, System, Process.
- Reuse these existing entity names exactly when they refer to the same thing: {names}
- At most 25 triples."""

ENTITY_SYSTEM = """Extract the named business entities (suppliers, products, stores, people,
categories, regions) mentioned in the user's question.
Return ONLY JSON: {"entities": ["...", "..."]}"""


def extract_graph_from_documents(docs: list[Document], llm, kg: KnowledgeGraph, max_chars: int = 6000) -> int:
    known = sorted({data["name"] for _, data in kg.g.nodes(data=True)})
    system = EXTRACT_SYSTEM.format(names=", ".join(known[:150]))
    added = 0
    for doc in docs:
        try:
            reply = llm.chat(system, f"Document: {doc.title}\n\n{doc.body[:max_chars]}", json_mode=True)
            triples = parse_json(reply).get("triples", [])
        except Exception as exc:  # keep ingest going if one document fails
            log.warning("Graph extraction failed for %s: %s", doc.doc_id, exc)
            continue
        for t in triples:
            try:
                src = kg.resolve_or_create(t["source"], t.get("source_type", "Entity"))
                tgt = kg.resolve_or_create(t["target"], t.get("target_type", "Entity"))
                added += kg.add_edge(src, t["relation"], tgt, origin=f"doc:{doc.doc_id}")
            except (KeyError, AttributeError):
                continue
    return added


def extract_query_entities(question: str, llm) -> list[str]:
    try:
        return [str(e) for e in parse_json(llm.chat(ENTITY_SYSTEM, question, json_mode=True)).get("entities", [])]
    except Exception:
        return []
