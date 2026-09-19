"""The end-to-end Hybrid RAG pipeline: ingest → route → retrieve (vector + graph) → fuse → generate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .acl import normalize_roles
from .chunking import chunk_document, load_documents
from .config import Settings, load_settings
from .extraction import extract_graph_from_documents, extract_query_entities
from .fusion import ContextItem, assign_refs, reciprocal_rank_fusion
from .graph_store import KnowledgeGraph
from .llm import get_llm
from .router import ROUTES, route_query
from .vector_store import VectorStore

log = logging.getLogger(__name__)

ANSWER_SYSTEM = """You are the internal knowledge assistant for DailyBasket, a supermarket chain.
Answer the question using ONLY the context provided.
- [D#] items are excerpts from company documents (policies, SOPs, manuals).
- [G#] items are relationship paths from the company knowledge graph, written as
  Entity (Type) -[RELATION]-> Entity (Type).
Rules:
1. Cite every factual statement with its reference, e.g. [D1] or [G2].
2. Combine graph facts (WHO / WHAT is connected) with document facts (WHAT TO DO / the RULE) when both are relevant.
3. If the context does not contain the answer, say "I don't know based on the available sources." Never invent facts.
4. Be concise: a short direct answer first, then bullet points if useful."""


@dataclass
class Answer:
    question: str
    route: str
    route_reason: str
    answer: str
    contexts: list[ContextItem] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "route": self.route,
            "route_reason": self.route_reason,
            "answer": self.answer,
            "entities": self.entities,
            "roles": self.roles,
            "contexts": [c.to_dict() for c in self.contexts],
        }


class HybridRAG:
    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = settings or load_settings()
        self.llm = llm or get_llm(self.settings)
        self.vectors: VectorStore | None = None
        self.graph: KnowledgeGraph | None = None

    # ---------------------------------------------------------------- index
    @property
    def _manifest_path(self) -> Path:
        return self.settings.index_dir / "manifest.json"

    def ingest(self, extract_graph: bool = False) -> dict:
        s = self.settings
        docs = load_documents(s.documents_dir)
        if not docs:
            raise FileNotFoundError(f"No .md documents found in {s.documents_dir}")
        chunks = [c for d in docs for c in chunk_document(d, s.chunk_size, s.chunk_overlap)]
        log.info("Embedding %d chunks from %d documents", len(chunks), len(docs))
        self.vectors = VectorStore.build(chunks, self.llm)

        self.graph = KnowledgeGraph.from_csv(s.graph_dir / "nodes.csv", s.graph_dir / "edges.csv")
        extracted = extract_graph_from_documents(docs, self.llm, self.graph) if extract_graph else 0

        s.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors.save(s.index_dir)
        self.graph.save(s.index_dir / "graph.json")
        stats = {
            "documents": len(docs),
            "chunks": len(chunks),
            "graph": self.graph.stats(),
            "extracted_edges": extracted,
            "embedder": self.llm.name,
        }
        self._manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    def load(self) -> HybridRAG:
        """Load the index from disk; (re)build it if missing or built with another embedder."""
        manifest = {}
        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedder") != self.llm.name:
            log.info("Index missing or built with a different embedder: re-ingesting")
            self.ingest()
            return self
        self.vectors = VectorStore.load(self.settings.index_dir)
        self.graph = KnowledgeGraph.load(self.settings.index_dir / "graph.json")
        return self

    def _ensure_loaded(self):
        if self.vectors is None or self.graph is None:
            self.load()

    # ------------------------------------------------------------ retrieval
    def retrieve_vector(self, question: str, roles=None) -> list[ContextItem]:
        self._ensure_loaded()
        qvec = self.llm.embed([question])[0]
        return [
            ContextItem(key=f"doc:{c.id}", kind="doc", text=c.text, citation=c.citation, raw_score=round(s, 4))
            for c, s in self.vectors.search(qvec, self.settings.top_k, roles)
        ]

    def retrieve_graph(self, question: str) -> tuple[list[ContextItem], list[str]]:
        self._ensure_loaded()
        seeds = self.graph.link_entities(question)
        if not seeds:  # fall back to GPT entity extraction + fuzzy matching
            seeds = self.graph.fuzzy_link(extract_query_entities(question, self.llm))
        facts = self.graph.traverse(seeds, self.settings.graph_hops, self.settings.max_graph_facts, question=question)
        items = [ContextItem(key=f.key, kind="graph", text=f.text, citation=f.citation) for f in facts]
        return items, [self.graph.g.nodes[s]["name"] for s in seeds]

    # --------------------------------------------------------------- answer
    def query(self, question: str, roles=None, route: str | None = None) -> Answer:
        self._ensure_loaded()
        roles = sorted(normalize_roles(roles))
        if route in ROUTES:
            reason = "Route forced by caller"
        else:
            route, reason = route_query(question, self.llm)

        lists: list[list[ContextItem]] = []
        entities: list[str] = []
        if route in ("vector", "hybrid"):
            lists.append(self.retrieve_vector(question, roles))
        if route in ("graph", "hybrid"):
            graph_items, entities = self.retrieve_graph(question)
            lists.append(graph_items)
            if route == "graph" and not graph_items:  # nothing linked: fall back to documents
                lists.append(self.retrieve_vector(question, roles))
                reason += " (no graph entities found, fell back to documents)"

        contexts = assign_refs(reciprocal_rank_fusion(lists, top_n=self.settings.max_context_items))
        if not contexts:
            text = "I don't know based on the available sources."
        else:
            text = self.llm.chat(ANSWER_SYSTEM, self._build_prompt(question, contexts))
        return Answer(question, route, reason, text.strip(), contexts, entities, roles)

    @staticmethod
    def _build_prompt(question: str, contexts: list[ContextItem]) -> str:
        blocks = [f"[{c.ref}] ({c.citation})\n{c.text}" for c in contexts]
        return f"Question: {question}\n\nContext:\n\n" + "\n\n".join(blocks)
