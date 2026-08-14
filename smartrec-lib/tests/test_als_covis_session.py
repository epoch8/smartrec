"""Tests for the covis session layer inside RecommenderALS (recsys_config.covis).

Uses the shared synthetic dataset from conftest.py: maldives cluster (u1-u3,
m1-m3), turkey cluster (u4-u6, t1-t3), globally popular pop1.
"""

from smartrec_lib.model import ALSSettings, CoVisSettings, Strategy
from smartrec_lib.recommenders import RecommenderALS

BASE = dict(
    ALS_ITERATIONS=2,
    ALS_REGULARIZATION_FACTOR=0.01,
    ALS_FACTORS=8,
    ALS_ALPHA=10,
    RECOMMENDER_DAYS_THRESHOLD=30,
)


def _fitted(dataset, **overrides):
    config = ALSSettings(**{**BASE, **overrides})
    model = RecommenderALS(recsys_config=config, model_name="als_test", model_version="1")
    model.train(dataset)
    return model


# --- no covis sub-config (default): prod behavior byte-for-byte --------------


def test_no_covis_config_hot_user_with_session_keeps_realtime_strategy(dataset):
    model = _fitted(dataset)
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"])
    assert result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value


def test_no_covis_config_trains_no_covis_layer(dataset):
    model = _fitted(dataset)
    assert model.recsys_config.covis is None
    assert model.covis is None


# --- covis sub-config present: session paths replaced by covis ----------------


def test_hot_user_without_session_serves_pure_als(dataset):
    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=None)
    assert result.strategy == Strategy.MODEL_HOT_USERS.value
    assert result.item_ids


def test_hot_user_with_session_serves_blend(dataset):
    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"])
    assert result.strategy == Strategy.MODEL_ALS_COVIS_BLEND.value
    assert result.item_ids
    # filter_viewed must hold across BOTH sources: u1 saw m1, m2, pop1 in training.
    assert set(result.item_ids).isdisjoint({"m1", "m2", "pop1"})


def test_unknown_user_with_session_serves_covis(dataset):
    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("ghost-user", top_n=3, filter_viewed=True, history=["m1", "m2"])
    assert result.strategy == Strategy.MODEL_COVIS_SESSION.value
    assert result.item_ids
    assert set(result.item_ids).isdisjoint({"m1", "m2"})  # session seed excluded


def test_unknown_user_with_unknown_session_falls_back(dataset):
    # History covis has never seen -> covis empty -> parent item-sim path answers.
    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("ghost-user", top_n=3, filter_viewed=True, history=["nope-1", "nope-2"])
    assert result.strategy != Strategy.MODEL_COVIS_SESSION.value


def test_cold_user_without_session_serves_popular(dataset):
    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("ghost-user", top_n=3, filter_viewed=False, history=None)
    assert result.strategy == Strategy.MODEL_COLD_USERS.value
    assert result.item_ids


# --- Triton serving input shapes (regression for the live dev incident) ------


def test_session_paths_accept_numpy_history(dataset):
    """Triton serving passes history as a numpy array of (bytes) strings;
    `if not history` on a multi-element ndarray raises ValueError."""
    import numpy as np

    model = _fitted(dataset, covis=CoVisSettings(COVIS_MIN_COOC=2))
    history_np = np.array(["m1", "m2"], dtype=object)
    result = model.recommend("ghost-user", top_n=3, filter_viewed=True, history=history_np)
    assert result.strategy == Strategy.MODEL_COVIS_SESSION.value
    assert result.item_ids

    history_bytes = np.array([b"m2:1.0", b"m1:2.0"], dtype=object)
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=history_bytes)
    assert result.strategy == Strategy.MODEL_ALS_COVIS_BLEND.value
    assert set(result.item_ids).isdisjoint({"m1", "m2", "pop1"})
