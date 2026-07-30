import pandas as pd
from rectools import Columns
from smartrec_lib.models import CoVisModel


def _internal(dataset, external_id):
    return int(dataset.item_id_map.convert_to_internal([external_id])[0])


def test_fit_builds_symmetric_neighbors(dataset):
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    m1, m2 = _internal(dataset, "m1"), _internal(dataset, "m2")
    m1_neighbors = dict(model.neighbors[m1])
    m2_neighbors = dict(model.neighbors[m2])
    assert m1_neighbors[m2] == 2.0
    assert m2_neighbors[m1] == 2.0


def test_min_cooc_filters_weak_edges(dataset):
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    m1, m3 = _internal(dataset, "m1"), _internal(dataset, "m3")
    # (m1, m3) co-occur only once (u2) -> filtered out
    assert m3 not in dict(model.neighbors.get(m1, []))


def test_top_k_truncates(dataset):
    model = CoVisModel(min_cooc=1, top_k=1)
    model.fit(dataset)
    assert all(len(nbrs) == 1 for nbrs in model.neighbors.values())


def test_recommend_scores_by_cooccurrence(dataset):
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    # u1 history (recent first): pop1, m2, m1. All seen items are excluded.
    # Candidate scores come from neighbors of the seed items; m3 must appear
    # (neighbor of m2 and pop1), t-items only via pop1.
    reco = model.recommend(users=["u1"], dataset=dataset, k=3, filter_viewed=True)
    items = reco[Columns.Item].tolist()
    assert "m3" in items
    assert set(items).isdisjoint({"m1", "m2", "pop1"})  # filter_viewed works


def test_recommend_respects_whitelist(dataset):
    model = CoVisModel(min_cooc=1, top_k=100)
    model.fit(dataset)
    reco = model.recommend(
        users=["u1"], dataset=dataset, k=5, filter_viewed=True, items_to_recommend=["m3", "t1"]
    )
    assert set(reco[Columns.Item]) <= {"m3", "t1"}


def test_recency_weighting_prefers_recent_seed_neighbors(dataset):
    model = CoVisModel(min_cooc=1, top_k=100)
    model.fit(dataset)
    # Session of one item = only its neighbors are scored.
    m2_int = _internal(dataset, "m2")
    exclude = {m2_int}
    ranked = model._score_session([m2_int], exclude, None, k=10)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    top_item, top_score = ranked[0]
    # pop1 co-occurs with m2 three times - the strongest neighbor.
    assert top_item == _internal(dataset, "pop1")
    assert top_score == 3.0
