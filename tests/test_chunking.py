from hybrid_rag.acl import can_access
from hybrid_rag.chunking import Document, chunk_document, load_documents, parse_front_matter

from .conftest import ROOT


def test_front_matter_is_parsed():
    meta, body = parse_front_matter("---\ntitle: X\naccess: [manager, qa]\n---\n# X\nbody")
    assert meta == {"title": "X", "access": ["manager", "qa"]}
    assert body.startswith("# X")


def test_documents_load_with_access_lists():
    docs = {d.doc_id: d for d in load_documents(ROOT / "data" / "documents")}
    assert docs["hr_policy"].access == ["all"]
    assert "manager" in docs["product_recall_sop"].access


def test_chunks_keep_section_and_title():
    doc = Document("d", "Doc", "# Doc\n\n## Alpha\none\n\n## Beta\ntwo", ["all"], "d.md")
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["Alpha", "Beta"]
    assert chunks[0].citation == "Doc › Alpha"
    assert chunks[0].embedding_text().startswith("Doc — Alpha")


def test_long_sections_are_split():
    body = "## S\n\n" + "\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(6))
    chunks = chunk_document(Document("d", "Doc", body, ["all"], "d.md"), size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_acl():
    assert can_access(["all"], ["staff"])
    assert not can_access(["manager"], ["staff"])
    assert can_access(["manager"], ["staff", "manager"])
    assert can_access(["manager"], ["admin"])
