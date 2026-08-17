import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import model_from_config
from smartrec_lib.research import CoVisModel


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
    reco = model.recommend(users=["u1"], dataset=dataset, k=5, filter_viewed=True, items_to_recommend=["m3", "t1"])
    assert "m3" in reco[Columns.Item].tolist()  # non-vacuous: whitelist actually admitted something
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


def test_online_offline_parity(dataset):
    """The core parity guarantee: identical session -> identical ranking,
    whether the history comes from the dataset (offline u2i) or is passed
    explicitly (online path)."""
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    # u2 full history, most recent first: pop1(d4), m3(d3), m2(d2), m1(d1)
    offline = model.recommend(users=["u2"], dataset=dataset, k=5, filter_viewed=True)
    online = model.recommend_for_session(["pop1", "m3", "m2", "m1"], dataset, k=5)
    assert offline[Columns.Item].tolist() == [item for item, _ in online]


def test_fit_basket_size_caps_basket_to_most_recent_interactions():
    # u1: "old" is the earliest event, then two more recent ones. With
    # fit_basket_size=2 the basket for _fit is capped to the 2 most recent
    # items, so "old" never enters any basket and therefore has no neighbors.
    rows = [("u1", "old", 1), ("u1", "recent_a", 2), ("u1", "recent_b", 3)]
    df = pd.DataFrame(rows, columns=[Columns.User, Columns.Item, "day"])
    df[Columns.Weight] = 1.0
    df[Columns.Datetime] = df["day"].map(lambda d: pd.Timestamp("2026-07-01") + pd.Timedelta(days=int(d)))
    df = df[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
    small_dataset = Dataset.construct(interactions_df=df)

    model = CoVisModel(min_cooc=1, top_k=100, fit_basket_size=2)
    model.fit(small_dataset)

    old_internal = _internal(small_dataset, "old")
    recent_a_internal = _internal(small_dataset, "recent_a")
    recent_b_internal = _internal(small_dataset, "recent_b")

    assert old_internal not in model.neighbors  # dropped from the basket entirely
    assert recent_b_internal in dict(model.neighbors[recent_a_internal])


def test_config_roundtrip(dataset):
    model = CoVisModel(top_k=7, min_cooc=3, session_size=5, fit_basket_size=42)
    config = model.get_config()
    restored = model_from_config(config)
    assert isinstance(restored, CoVisModel)
    assert (restored.top_k, restored.min_cooc, restored.session_size, restored.fit_basket_size) == (7, 3, 5, 42)


def test_config_serializes_to_plain_dict():
    model = CoVisModel(top_k=7)
    simple = model.get_config(simple_types=True)
    assert simple["top_k"] == 7
    assert simple["cls"] == "smartrec_lib.research.covis.CoVisModel"
