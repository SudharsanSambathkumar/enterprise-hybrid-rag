from hybrid_rag.fusion import ContextItem, assign_refs, reciprocal_rank_fusion
from hybrid_rag.llm import parse_json
from hybrid_rag.router import heuristic_route


def test_heuristic_router():
    assert heuristic_route("How many sick leaves do store staff get?")[0] == "vector"
    assert heuristic_route("Which stores stock FreshDairy milk?")[0] == "graph"
    assert heuristic_route("Milk recall: which managers are affected and what must they do?")[0] == "hybrid"


def item(key, kind="doc"):
    return ContextItem(key=key, kind=kind, text=key, citation=key)


def test_rrf_rewards_items_ranked_high_in_multiple_lists():
    fused = reciprocal_rank_fusion([[item("a"), item("b"), item("c")], [item("c"), item("a")]])
    assert [f.key for f in fused] == ["a", "c", "b"]
    assert fused[0].score > fused[1].score > fused[2].score


def test_rrf_interleaves_independent_lanes():
    fused = reciprocal_rank_fusion([[item("d1"), item("d2")], [item("g1", "graph"), item("g2", "graph")]])
    assert {fused[0].key, fused[1].key} == {"d1", "g1"}
    refs = [i.ref for i in assign_refs(fused)]
    assert sorted(refs) == ["D1", "D2", "G1", "G2"]


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"route": "graph"}\n```') == {"route": "graph"}
    assert parse_json('Sure! {"a": 1}') == {"a": 1}
