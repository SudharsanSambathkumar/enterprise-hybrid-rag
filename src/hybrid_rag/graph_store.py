"""Knowledge graph built on NetworkX: entity linking + multi-hop path retrieval.

Swap for Neo4j / Cosmos DB Gremlin in production; the pipeline only needs
`link_entities`, `fuzzy_link` and `traverse`.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

_SEP = "\x1f"


@dataclass
class GraphFact:
    node_ids: list[str]
    labels: list[str]
    relations: list[str]
    seeds_hit: int = 0

    @property
    def key(self) -> str:
        return "graph:" + "|".join(self.node_ids)

    @property
    def text(self) -> str:
        parts = [self.labels[0]]
        for rel, label in zip(self.relations, self.labels[1:], strict=False):
            parts.append(f"-[{rel}]-> {label}")
        return " ".join(parts)

    @property
    def citation(self) -> str:
        names = [lbl.split(" (")[0] for lbl in self.labels]
        return "Graph: " + " → ".join(names)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "entity"


class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()
        self._alias_index: dict[str, str] | None = None

    # ------------------------------------------------------------------ build
    def add_node(self, node_id: str, type: str, name: str, aliases=(), description: str = ""):
        existing = self.g.nodes[node_id] if node_id in self.g else {}
        merged_aliases = sorted(set(existing.get("aliases", [])) | {a for a in aliases if a})
        self.g.add_node(
            node_id,
            type=type or existing.get("type", "Entity"),
            name=name or existing.get("name", node_id),
            aliases=merged_aliases,
            description=description or existing.get("description", ""),
        )
        self._alias_index = None

    def add_edge(self, source: str, relation: str, target: str, origin: str = "csv"):
        relation = relation.strip().upper().replace(" ", "_")
        for _, tgt, data in self.g.out_edges(source, data=True):
            if tgt == target and data.get("relation") == relation:
                return False
        self.g.add_edge(source, target, relation=relation, origin=origin)
        return True

    @classmethod
    def from_csv(cls, nodes_csv: str | Path, edges_csv: str | Path) -> KnowledgeGraph:
        kg = cls()
        with open(nodes_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                aliases = [a.strip() for a in (row.get("aliases") or "").split("|") if a.strip()]
                kg.add_node(row["id"], row["type"], row["name"], aliases, row.get("description", ""))
        with open(edges_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for end in (row["source"], row["target"]):
                    if end not in kg.g:
                        raise ValueError(f"Edge references unknown node '{end}'")
                kg.add_edge(row["source"], row["relation"], row["target"])
        return kg

    def resolve_or_create(self, name: str, type: str = "Entity") -> str:
        """Return an existing node id for `name` (by name/alias), or create a new node."""
        hit = self._index().get(name.strip().lower())
        if hit:
            return hit
        node_id = slugify(name)
        while node_id in self.g and self.g.nodes[node_id]["name"].lower() != name.lower():
            node_id += "_x"
        self.add_node(node_id, type, name.strip())
        return node_id

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        data = {
            "nodes": [{"id": n, **attrs} for n, attrs in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
        }
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        for node in data["nodes"]:
            node_id = node.pop("id")
            kg.g.add_node(node_id, **node)
        for edge in data["edges"]:
            kg.g.add_edge(edge.pop("source"), edge.pop("target"), **edge)
        return kg

    # ------------------------------------------------------------- lookup
    def label(self, node_id: str) -> str:
        data = self.g.nodes[node_id]
        return f"{data['name']} ({data['type']})"

    def _index(self) -> dict[str, str]:
        if self._alias_index is None:
            index: dict[str, str] = {}
            for node_id, data in self.g.nodes(data=True):
                for alias in [data.get("name", ""), *data.get("aliases", [])]:
                    if alias:
                        index.setdefault(alias.lower(), node_id)
            self._alias_index = index
        return self._alias_index

    def link_entities(self, text: str) -> list[str]:
        """Find graph entities mentioned in free text (longest alias match first)."""
        lowered = text.lower()
        found: list[str] = []
        spans: list[tuple[int, int]] = []
        for alias in sorted(self._index(), key=len, reverse=True):
            for m in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered):
                if any(s < m.end() and m.start() < e for s, e in spans):
                    continue
                spans.append((m.start(), m.end()))
                node_id = self._index()[alias]
                if node_id not in found:
                    found.append(node_id)
        return found

    def fuzzy_link(self, names: list[str], cutoff: float = 0.8) -> list[str]:
        """Map loosely-spelled names (e.g. from an LLM) onto graph nodes."""
        found: list[str] = []
        keys = list(self._index())
        for name in names:
            match = difflib.get_close_matches(name.lower().strip(), keys, n=1, cutoff=cutoff)
            if match and self._index()[match[0]] not in found:
                found.append(self._index()[match[0]])
        return found

    # ----------------------------------------------------------- retrieval
    def traverse(self, seeds: list[str], hops: int = 3, max_paths: int = 12) -> list[GraphFact]:
        """Collect relationship paths around the seed entities.

        * Downstream: follow outgoing edges up to `hops` (supplier → product → store → manager).
        * Upstream: follow incoming edges up to `hops - 1` (who supplies / manages this?).
        Paths that are fully contained in a longer path are dropped, and paths that
        connect several seed entities are ranked first.
        """
        seeds = [s for s in seeds if s in self.g]
        raw: list[tuple[list[str], list[str]]] = []
        budget = max_paths * 20

        def walk(path: list[str], rels: list[str], forward: bool, limit: int):
            if len(raw) >= budget:
                return
            node, extended = path[-1], False
            if len(rels) < limit:
                edges = self.g.out_edges(node, data=True) if forward else self.g.in_edges(node, data=True)
                for u, v, data in edges:
                    nxt = v if forward else u
                    if nxt in path:
                        continue
                    extended = True
                    walk(path + [nxt], rels + [data["relation"]], forward, limit)
            if not extended and rels:
                raw.append((path, rels) if forward else (path[::-1], rels[::-1]))

        for seed in seeds:
            walk([seed], [], True, hops)
            walk([seed], [], False, max(1, hops - 1))

        def seq(p):
            nodes, rels = p
            out = [nodes[0]]
            for r, n in zip(rels, nodes[1:], strict=False):
                out += [r, n]
            return _SEP + _SEP.join(out) + _SEP

        unique = {seq(p): p for p in raw}
        keys = sorted(unique, key=len, reverse=True)
        kept = [k for i, k in enumerate(keys) if not any(k in longer for longer in keys[:i])]

        seed_set = set(seeds)
        facts = []
        for k in kept:
            nodes, rels = unique[k]
            facts.append(
                GraphFact(
                    node_ids=nodes,
                    labels=[self.label(n) for n in nodes],
                    relations=rels,
                    seeds_hit=len(seed_set & set(nodes)),
                )
            )
        facts.sort(key=lambda f: (-f.seeds_hit, len(f.relations), f.text))
        return facts[:max_paths]

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            by_type[data["type"]] = by_type.get(data["type"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), "by_type": by_type}
