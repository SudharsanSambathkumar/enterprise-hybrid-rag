"""Load markdown documents and split them into section-aware chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    access: list[str]
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    access: list[str] = field(default_factory=lambda: ["all"])

    @property
    def citation(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title

    def embedding_text(self) -> str:
        # Prepending title + section ("contextual chunk headers") makes short chunks
        # far easier to retrieve than embedding the bare paragraph.
        return f"{self.title} — {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse a minimal YAML-like front matter block (key: value / key: [a, b])."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value.strip("'\"")
    return meta, raw[match.end() :]


def _first_h1(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def load_documents(doc_dir: str | Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(doc_dir).glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta.get("title") or _first_h1(body) or path.stem.replace("_", " ").title()
        access = meta.get("access") or ["all"]
        if isinstance(access, str):
            access = [access]
        docs.append(
            Document(
                doc_id=path.stem,
                title=title,
                body=body,
                access=[a.lower() for a in access],
                source=str(path),
                meta=meta,
            )
        )
    return docs


def _sections(body: str):
    """Yield (section_heading, text) pairs. The H1 is treated as the document title."""
    section, buf = "", []
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match:
            if "".join(buf).strip():
                yield section, "\n".join(buf).strip()
            buf = []
            if len(match.group(1)) > 1:
                section = match.group(2).strip()
            continue
        buf.append(line)
    if "".join(buf).strip():
        yield section, "\n".join(buf).strip()


def chunk_document(doc: Document, size: int = 600, overlap: int = 120) -> list[Chunk]:
    """Split a document by section, then pack paragraphs up to `size` characters.

    When a section overflows, the last paragraph of the previous chunk is carried
    over (if it is shorter than `overlap`) so context is not cut mid-thought.
    """
    pieces: list[tuple[str, str]] = []
    for section, text in _sections(doc.body):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current: list[str] = []
        fresh = 0  # paragraphs in `current` that were not carried over
        for para in paragraphs:
            if fresh and len("\n\n".join(current + [para])) > size:
                pieces.append((section, "\n\n".join(current)))
                tail = current[-1]
                current = [tail] if len(tail) <= overlap else []
                fresh = 0
            current.append(para)
            fresh += 1
        if fresh:
            pieces.append((section, "\n\n".join(current)))

    return [
        Chunk(
            id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            section=section,
            text=text,
            access=list(doc.access),
        )
        for i, (section, text) in enumerate(pieces)
    ]
