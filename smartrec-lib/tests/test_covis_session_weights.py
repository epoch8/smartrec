"""Tests for the "sw" scoring variant (COVIS_SESSION_WEIGHTS): seed recency is
multiplied by the API event weight carried in "tour_id:weight" history entries.

Uses the shared synthetic dataset from conftest.py. With COVIS_MIN_COOC=2 the
relevant neighbor counts are: m1 -> m2(2), pop1(2); t1 -> t2(2), pop1(2).
"""

from smartrec_lib.model import ALSSettings, CoVisSettings, ModelSetSettings
from smartrec_lib.recommenders import RecommenderALS, RecommenderCoVis


def _fitted_covis(dataset, **overrides):
    config = CoVisSettings(RECOMMENDER_DAYS_THRESHOLD=30, COVIS_MIN_COOC=2, **overrides)
    model = RecommenderCoVis(recsys_config=config, model_name="covis_test", model_version="1")
    model.train(dataset)
    return model


def test_parse_history_returns_weighted_pairs():
    parsed = RecommenderCoVis._parse_history(["m1:2.0", b"m2", "m3:oops", 42])
    assert parsed == [("m1", 2.0), ("m2", 1.0), ("m3", 1.0), ("42", 1.0)]


def test_flag_off_ignores_weights(dataset):
    model = _fitted_covis(dataset)
    assert model.session_weights_enabled is False
    light = model.recommend("u-x", top_n=5, filter_viewed=True, history=["m1:1.0", "t1:1.0"])
    heavy = model.recommend("u-x", top_n=5, filter_viewed=True, history=["m1:1.0", "t1:3.0"])
    assert light.item_ids == heavy.item_ids
    assert light.scores == heavy.scores


def test_flag_on_heavy_seed_outranks_recent_light_seed(dataset):
    """History is most-recent-first: m1 (recency 1.0) vs t1 (recency 0.5).
    Unweighted, m1's neighbor m2 outranks t1's neighbor t2 (2.0 vs 1.0).
    With weight 3.0 on t1, t2 flips ahead of m2 (2 x 0.5 x 3 = 3.0 vs 2.0)."""
    model_off = _fitted_covis(dataset)
    off = model_off.recommend("u-x", top_n=5, filter_viewed=True, history=["m1:1.0", "t1:3.0"])
    assert off.item_ids.index("m2") < off.item_ids.index("t2")

    model_on = _fitted_covis(dataset, COVIS_SESSION_WEIGHTS=True)
    assert model_on.session_weights_enabled is True
    on = model_on.recommend("u-x", top_n=5, filter_viewed=True, history=["m1:1.0", "t1:3.0"])
    assert on.item_ids.index("t2") < on.item_ids.index("m2")


def test_flag_on_without_weights_in_history_matches_flag_off(dataset):
    """Plain "tour_id" entries default to weight 1.0 -> identical scoring."""
    model_off = _fitted_covis(dataset)
    model_on = _fitted_covis(dataset, COVIS_SESSION_WEIGHTS=True)
    off = model_off.recommend("u-x", top_n=5, filter_viewed=True, history=["m1", "t1"])
    on = model_on.recommend("u-x", top_n=5, filter_viewed=True, history=["m1", "t1"])
    assert off.item_ids == on.item_ids
    assert off.scores == on.scores


def test_als_covis_layer_inherits_the_flag(dataset):
    config = ModelSetSettings(
        als=ALSSettings(
            ALS_ITERATIONS=2,
            ALS_REGULARIZATION_FACTOR=0.01,
            ALS_FACTORS=8,
            ALS_ALPHA=10,
            RECOMMENDER_DAYS_THRESHOLD=30,
        ),
        covis=CoVisSettings(COVIS_MIN_COOC=2, COVIS_SESSION_WEIGHTS=True),
    )
    model = RecommenderALS(recsys_config=config, model_name="als_test", model_version="1")
    model.train(dataset)
    assert model.covis is not None
    assert model.covis.session_weights_enabled is True
