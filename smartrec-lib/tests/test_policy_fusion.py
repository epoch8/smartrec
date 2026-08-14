from smartrec_lib.policy.fusion import rrf_fuse, session_weight


def test_single_source_preserves_order():
    fused = rrf_fuse({"a": ["x", "y", "z"]}, {"a": 1.0})
    assert [item for item, _ in fused] == ["x", "y", "z"]


def test_agreement_beats_single_source():
    # "y" is ranked by both sources, "x" and "z" by one each -> "y" wins.
    fused = rrf_fuse({"a": ["x", "y"], "b": ["z", "y"]}, {"a": 1.0, "b": 1.0})
    assert fused[0][0] == "y"


def test_weight_dominance():
    # With a heavily weighted source "b", its top item must win.
    fused = rrf_fuse({"a": ["x"], "b": ["z"]}, {"a": 1.0, "b": 10.0})
    assert fused[0][0] == "z"


def test_zero_weight_source_is_ignored():
    fused = rrf_fuse({"a": ["x"], "b": ["z"]}, {"a": 1.0, "b": 0.0})
    assert [item for item, _ in fused] == ["x"]


def test_deterministic_tie_break():
    fused = rrf_fuse({"a": ["b_item", "a_item"], "b": ["a_item", "b_item"]}, {"a": 1.0, "b": 1.0})
    # Equal scores -> lexicographic by str(item) for determinism.
    assert [item for item, _ in fused] == ["a_item", "b_item"]


def test_session_weight_tiers():
    tiers = [(1, 0.3), (3, 1.0)]
    assert session_weight(0, tiers) == 0.0
    assert session_weight(1, tiers) == 0.3
    assert session_weight(2, tiers) == 0.3
    assert session_weight(3, tiers) == 1.0
    assert session_weight(10, tiers) == 1.0
