"""Tests for the covis session layer of the ALS model set (ModelSetSettings.covis).

Uses the shared synthetic dataset from conftest.py: maldives cluster (u1-u3,
m1-m3), turkey cluster (u4-u6, t1-t3), globally popular pop1.
"""

from smartrec_lib.model import ALSSettings, CoVisSettings, ModelSetSettings, Strategy
from smartrec_lib.recommenders import RecommenderModelSet

BASE_ALS = dict(
    ALS_ITERATIONS=2,
    ALS_REGULARIZATION_FACTOR=0.01,
    ALS_FACTORS=8,
    ALS_ALPHA=10,
    RECOMMENDER_DAYS_THRESHOLD=30,
)


def _fitted(dataset, **overrides):
    """Overrides apply to the model set (covis=, blend=), not to the ALS ranker."""
    config = ModelSetSettings(main=ALSSettings(**BASE_ALS), **overrides)
    model = RecommenderModelSet(recsys_config=config, model_name="als_test", model_version="1")
    model.train(dataset)
    return model


# --- no covis sub-config (default): prod behavior byte-for-byte --------------


def test_no_covis_config_hot_user_with_session_keeps_realtime_strategy(dataset):
    model = _fitted(dataset)
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"])
    assert result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value


def test_no_covis_config_trains_no_covis_layer(dataset):
    model = _fitted(dataset)
    assert model.recsys_config.session is None
    assert model.session is None


# --- covis sub-config present: session paths replaced by covis ----------------


def test_hot_user_without_session_serves_pure_als(dataset):
    model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=None)
    assert result.strategy == Strategy.MODEL_HOT_USERS.value
    assert result.item_ids


def test_hot_user_with_session_reports_the_realtime_segment(dataset):
    """The blend keeps the label the item-sim session path always used: the
    strategy names the segment (hot user, live session), not the algorithm."""
    model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"])
    assert result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value
    assert result.item_ids
    # filter_viewed must hold across BOTH sources: u1 saw m1, m2, pop1 in training.
    assert set(result.item_ids).isdisjoint({"m1", "m2", "pop1"})


def test_hot_user_with_session_actually_goes_through_covis(dataset):
    """Since both paths now report the same strategy, the label can no longer
    prove which one ran - so compare against the same model without the covis
    sub-config. Identical output would mean the blend never engaged."""
    blended = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    item_sim = _fitted(dataset)
    session = ["m2", "m1"]

    with_covis = blended.recommend("u1", top_n=3, filter_viewed=True, history=session)
    without = item_sim.recommend("u1", top_n=3, filter_viewed=True, history=session)

    assert with_covis.strategy == without.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value
    assert (with_covis.item_ids, with_covis.scores) != (without.item_ids, without.scores)


def test_unknown_user_with_session_reports_the_realtime_segment(dataset):
    model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("ghost-user", top_n=3, filter_viewed=True, history=["m1", "m2"])
    assert result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert result.item_ids
    assert set(result.item_ids).isdisjoint({"m1", "m2"})  # session seed excluded


def test_unknown_user_with_session_actually_goes_through_covis(dataset):
    covis_model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    item_sim = _fitted(dataset)
    session = ["m1", "m2"]

    with_covis = covis_model.recommend("ghost-user", top_n=3, filter_viewed=True, history=session)
    without = item_sim.recommend("ghost-user", top_n=3, filter_viewed=True, history=session)

    assert with_covis.item_ids != without.item_ids


def test_unknown_user_with_unknown_session_falls_back_to_item_sim(dataset):
    """History covis has never seen -> covis returns nothing -> the parent
    item-sim path answers. Both report the same strategy now, so the fallback is
    pinned by output equality with the no-covis model instead."""
    covis_model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    item_sim = _fitted(dataset)
    session = ["nope-1", "nope-2"]

    fell_back = covis_model.recommend("ghost-user", top_n=3, filter_viewed=True, history=session)
    direct = item_sim.recommend("ghost-user", top_n=3, filter_viewed=True, history=session)

    assert fell_back.item_ids == direct.item_ids


def test_cold_user_without_session_serves_popular(dataset):
    model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    result = model.recommend("ghost-user", top_n=3, filter_viewed=False, history=None)
    assert result.strategy == Strategy.MODEL_COLD_USERS.value
    assert result.item_ids


# --- Triton serving input shapes (regression for the live dev incident) ------


def test_session_paths_accept_numpy_history(dataset):
    """Triton serving passes history as a numpy array of (bytes) strings;
    `if not history` on a multi-element ndarray raises ValueError."""
    import numpy as np

    model = _fitted(dataset, session=CoVisSettings(COVIS_MIN_COOC=2))
    history_np = np.array(["m1", "m2"], dtype=object)
    result = model.recommend("ghost-user", top_n=3, filter_viewed=True, history=history_np)
    assert result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert result.item_ids

    history_bytes = np.array([b"m2:1.0", b"m1:2.0"], dtype=object)
    result = model.recommend("u1", top_n=3, filter_viewed=True, history=history_bytes)
    assert result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value
    assert set(result.item_ids).isdisjoint({"m1", "m2", "pop1"})
