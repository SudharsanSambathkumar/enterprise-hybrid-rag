from hybrid_rag.graph_store import KnowledgeGraph

from .conftest import ROOT


def kg() -> KnowledgeGraph:
    g = ROOT / "data" / "graph"
    return KnowledgeGraph.from_csv(g / "nodes.csv", g / "edges.csv")


def test_entity_linking_uses_names_and_aliases():
    graph = kg()
    assert graph.link_entities("FreshDairy milk is recalled") == ["freshdairy", "milk_1l"]
    assert "hsr" in graph.link_entities("what does the HSR store sell?")


def test_fuzzy_linking():
    assert kg().fuzzy_link(["Fresh Diary"]) == ["freshdairy"]


def test_recall_traversal_finds_affected_stores_and_managers():
    graph = kg()
    facts = graph.traverse(graph.link_entities("FreshDairy milk recall"), hops=3)
    text = "\n".join(f.text for f in facts)
    assert "Milk 1L (Product) -[STOCKED_IN]-> Koramangala (Store) -[MANAGED_BY]-> Ravi (Manager)" in text
    assert "Whitefield (Store) -[MANAGED_BY]-> Priya (Manager)" in text
    # Bread is not a FreshDairy product, so Indiranagar/Suresh must not appear via milk
    assert "Milk 1L (Product) -[STOCKED_IN]-> Indiranagar" not in text
    # Paths that connect both seeds rank first
    assert facts[0].text.startswith("FreshDairy (Supplier) -[SUPPLIES]-> Milk 1L")


def test_upstream_traversal_finds_suppliers():
    graph = kg()
    text = "\n".join(f.text for f in graph.traverse(["hsr"], hops=3))
    assert "GreenFarms (Supplier) -[SUPPLIES]-> Spinach (Product) -[STOCKED_IN]-> HSR Layout" in text


def test_paths_contained_in_longer_paths_are_dropped():
    graph = kg()
    texts = [f.text for f in graph.traverse(["freshdairy", "milk_1l"], hops=3, max_paths=100)]
    for t in texts:
        assert not any(t != other and t in other for other in texts)


def test_save_and_load_roundtrip(tmp_path):
    graph = kg()
    graph.save(tmp_path / "g.json")
    loaded = KnowledgeGraph.load(tmp_path / "g.json")
    assert loaded.stats() == graph.stats()
    assert loaded.link_entities("bread") == ["bread"]

