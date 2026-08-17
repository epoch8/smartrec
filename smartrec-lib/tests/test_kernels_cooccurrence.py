"""Unit tests for the L1 co-visitation kernel.

The kernel is the one implementation behind both the serving shell
(`recommenders/recommender_covis.py`) and the research model
(`research/covis.py`), so its four parameters are tested here directly rather
than only through whichever caller happens to exercise them.
"""

import pytest

from smartrec_lib.kernels.cooccurrence import build_neighbor_map, score_session


def test_min_cooc_drops_edges_below_the_threshold():
    baskets = [["a", "b"], ["a", "b"], ["a", "c"]]

    kept = build_neighbor_map(baskets, min_cooc=2, top_k=10, deterministic_ties=True)
    assert kept == {"a": [("b", 2.0)], "b": [("a", 2.0)]}

    loose = build_neighbor_map(baskets, min_cooc=1, top_k=10, deterministic_ties=True)
    assert dict(loose["a"]) == {"b": 2.0, "c": 1.0}


def test_a_pair_counts_once_per_basket_however_often_it_repeats():
    """Baskets are deduplicated before pairing, so a user re-viewing the same
    tour ten times still contributes a single co-occurrence."""
    repeated = build_neighbor_map([["a", "b", "a", "b", "a"]], min_cooc=1, top_k=10, deterministic_ties=True)
    assert repeated == {"a": [("b", 1.0)], "b": [("a", 1.0)]}


def test_single_item_baskets_produce_no_edges():
    assert build_neighbor_map([["a"], ["b"], []], min_cooc=1, top_k=10, deterministic_ties=True) == {}


def test_fit_basket_size_keeps_the_most_recent_entries():
    """Baskets arrive most-recent-first, so the cap keeps the head."""
    baskets = [["new1", "new2", "old1", "old2"]]

    capped = build_neighbor_map(baskets, min_cooc=1, top_k=10, deterministic_ties=True, fit_basket_size=2)
    assert set(capped) == {"new1", "new2"}

    uncapped = build_neighbor_map(baskets, min_cooc=1, top_k=10, deterministic_ties=True)
    assert set(uncapped) == {"new1", "new2", "old1", "old2"}


def test_deterministic_ties_resolve_top_k_truncation_by_id():
    """Every candidate co-occurs with the hub exactly once, so `top_k` has to
    cut a tie. With deterministic_ties the survivors are the id-ascending ones,
    reproducibly - this is the regime where the two callers can diverge, and it
    was previously untested."""
    hub_basket = ["hub"] + [f"cand{i:02d}" for i in range(12)]

    first = build_neighbor_map([hub_basket], min_cooc=1, top_k=3, deterministic_ties=True)
    second = build_neighbor_map([hub_basket], min_cooc=1, top_k=3, deterministic_ties=True)

    assert [item for item, _ in first["hub"]] == ["cand00", "cand01", "cand02"]
    assert first == second


def test_counts_outrank_ids_when_ties_are_deterministic():
    """Id order is only the tie-break: a higher count always wins first."""
    baskets = [["hub", "zzz"], ["hub", "zzz"], ["hub", "aaa"]]

    neighbors = build_neighbor_map(baskets, min_cooc=1, top_k=1, deterministic_ties=True)
    assert neighbors["hub"] == [("zzz", 2.0)]


def test_score_session_weights_seeds_by_recency():
    """Seed at position `pos` of `n` contributes (n - pos) / n."""
    neighbors = {"s1": [("x", 1.0)], "s2": [("y", 1.0)]}

    ranked = score_session(neighbors, ["s1", "s2"], k=10, deterministic_ties=True)
    assert ranked == [("x", 1.0), ("y", 0.5)]


def test_session_size_changes_scores_not_just_membership():
    """The cap changes `n`, so it rescales every remaining seed's contribution."""
    neighbors = {"s1": [("x", 1.0)], "s2": [("y", 1.0)]}

    full = score_session(neighbors, ["s1", "s2"], k=10, deterministic_ties=True)
    capped = score_session(neighbors, ["s1", "s2"], k=10, deterministic_ties=True, session_size=1)

    assert dict(full)["x"] == 0.5 * 2
    assert dict(capped) == {"x": 1.0}


def test_seed_weights_multiply_recency():
    """A heavier seed can outrank a more recent lighter one."""
    neighbors = {"s1": [("x", 1.0)], "s2": [("y", 1.0)]}

    plain = score_session(neighbors, ["s1", "s2"], k=10, deterministic_ties=True)
    assert [item for item, _ in plain] == ["x", "y"]

    weighted = score_session(neighbors, ["s1", "s2"], k=10, deterministic_ties=True, seed_weights=[1.0, 3.0])
    assert weighted == [("y", 1.5), ("x", 1.0)]


def test_seed_weights_shorter_than_the_session_is_an_error():
    with pytest.raises(ValueError, match="shorter than the session"):
        score_session({}, ["s1", "s2"], k=10, deterministic_ties=True, seed_weights=[1.0])


def test_exclude_and_allowed_restrict_candidates():
    neighbors = {"s1": [("x", 1.0), ("y", 1.0), ("z", 1.0)]}

    assert dict(score_session(neighbors, ["s1"], k=10, deterministic_ties=True, exclude={"x"})) == {
        "y": 1.0,
        "z": 1.0,
    }
    assert dict(score_session(neighbors, ["s1"], k=10, deterministic_ties=True, allowed={"z"})) == {"z": 1.0}


def test_unknown_seeds_and_empty_sessions_contribute_nothing():
    assert score_session({"s1": [("x", 1.0)]}, ["missing"], k=10, deterministic_ties=True) == []
    assert score_session({"s1": [("x", 1.0)]}, [], k=10, deterministic_ties=True) == []
