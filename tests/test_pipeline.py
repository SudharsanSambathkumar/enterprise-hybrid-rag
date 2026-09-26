from fastapi.testclient import TestClient


def test_vector_route_finds_hr_policy(rag):
    ans = rag.query("How many sick leaves do store staff get?")
    assert ans.route == "vector"
    assert ans.contexts[0].citation == "HR Policy › Sick Leave"
    assert "[D1]" in ans.answer


def test_graph_route_finds_affected_stores(rag):
    ans = rag.query("FreshDairy milk is recalled. Which stores and managers are affected?", route="graph")
    assert ans.entities == ["FreshDairy", "Milk 1L"]
    graph_text = " ".join(c.text for c in ans.contexts if c.kind == "graph")
    assert "Ravi" in graph_text and "Priya" in graph_text
    assert all(c.kind == "graph" for c in ans.contexts)


def test_hybrid_combines_graph_and_documents_for_managers(rag):
    ans = rag.query("Milk recall: which stores are affected and what must the manager do?", roles=["manager"])
    assert ans.route == "hybrid"
    kinds = {c.kind for c in ans.contexts}
    assert kinds == {"doc", "graph"}
    assert any(c.citation.startswith("Product Recall SOP") for c in ans.contexts)


def test_acl_hides_manager_documents_from_staff(rag):
    ans = rag.query("Product recall SOP: steps to pull stock and quarantine", roles=["staff"], route="vector")
    assert not any(c.citation.startswith(("Product Recall SOP", "Store Operations")) for c in ans.contexts)
    ans_mgr = rag.query("Product recall SOP: steps to pull stock and quarantine", roles=["manager"], route="vector")
    assert ans_mgr.contexts[0].citation.startswith("Product Recall SOP")


def test_index_reloads_from_disk(rag):
    from hybrid_rag import HybridRAG

    reloaded = HybridRAG(rag.settings, llm=rag.llm).load()
    assert len(reloaded.vectors) == len(rag.vectors)
    assert reloaded.graph.stats() == rag.graph.stats()


def test_api(rag, monkeypatch):
    from hybrid_rag import api

    monkeypatch.setattr(api, "get_rag", lambda: rag)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    res = client.post("/ask", json={"question": "Who supplies bread?", "route": "graph"})
    assert res.status_code == 200
    assert any("BakeCo" in c["text"] for c in res.json()["contexts"])
