"""Tests for the model-set config shape and for loading artifacts pickled with
either of the two older shapes.

`recsys_config` is pickled inside every model.pkl in S3, so an artifact trained
before composition moved to `ModelSetSettings` must still unpickle and serve.
Two older shapes exist and both are covered here:

1. flat - one ALSSettings carrying POPULARITY_*/COVIS_*/BLEND_* fields directly;
2. nested - one ALSSettings carrying `popular`/`covis`/`blend` sub-configs.

The legacy pickles are reproduced by a stand-in class whose `__reduce__` rebuilds
a real ALSSettings from a legacy `__getstate__` payload - exactly the path CPython
takes when it unpickles a pre-refactor artifact.
"""

import pickle
from datetime import timedelta

import dill
import pytest

from smartrec_lib.model import ALSSettings, BlendSettings, CoVisSettings, ModelSetSettings, PopularSettings, Strategy
from smartrec_lib.recommenders import RecommenderALS, RecommenderCoVis

BASE = dict(
    ALS_ITERATIONS=2,
    ALS_REGULARIZATION_FACTOR=0.01,
    ALS_FACTORS=8,
    ALS_ALPHA=10,
    RECOMMENDER_DAYS_THRESHOLD=30,
)

# The exact flat payload a pre-refactor als_covis_youtravel artifact carries.
LEGACY_ALS_COVIS_DICT = {
    "RECOMMENDER_DAYS_THRESHOLD": 30,
    "RECOMMENDER_RANDOM_STATE": 42,
    "ALS_ITERATIONS": 2,
    "ALS_REGULARIZATION_FACTOR": 0.01,
    "ALS_FACTORS": 8,
    "ALS_ALPHA": 10,
    "POPULARITY_STRATEGY": "n_users",
    "POPULARITY_PERIOD": timedelta(days=14),
    "SESSION_COVIS_ENABLED": True,
    "COVIS_TOP_K": 100,
    "COVIS_MIN_COOC": 2,
    "COVIS_SESSION_WEIGHTS": False,
    "BLEND_ALS_WEIGHT": 1.0,
    "BLEND_COVIS_WEIGHT": 1.0,
    "BLEND_RRF_K": 60,
}


def _rebuild_legacy_als_settings(state):
    """Unpickle hook: what CPython does for a pickled pydantic model."""
    obj = ALSSettings.__new__(ALSSettings)
    obj.__setstate__(state)
    return obj


class LegacyALSSettings:
    """Stands in for the pre-refactor ALSSettings class.

    Pickles into a smartrec_lib.model.ALSSettings carrying the OLD flat
    __dict__, so loading it with the current code exercises the real migration.
    """

    def __init__(self, flat_fields):
        self.state = {
            "__dict__": dict(flat_fields),
            "__pydantic_extra__": None,
            "__pydantic_fields_set__": set(flat_fields),
            "__pydantic_private__": None,
        }

    def __reduce__(self):
        return (_rebuild_legacy_als_settings, (self.state,))


def _legacy_config(**overrides):
    """A real ALSSettings produced by unpickling legacy bytes."""
    flat = {**LEGACY_ALS_COVIS_DICT, **overrides}
    return pickle.loads(pickle.dumps(LegacyALSSettings(flat)))


# --- the new shape ------------------------------------------------------------


def test_composition_defaults_to_no_session_layer():
    config = ModelSetSettings(als=ALSSettings(**BASE))
    assert config.covis is None
    assert isinstance(config.popular, PopularSettings)
    assert isinstance(config.blend, BlendSettings)


def test_covis_presence_is_the_switch():
    config = ModelSetSettings(als=ALSSettings(**BASE), covis=CoVisSettings(COVIS_MIN_COOC=3, COVIS_TOP_K=7))
    assert config.covis.COVIS_MIN_COOC == 3
    assert config.covis.COVIS_TOP_K == 7


def test_sub_model_fields_are_no_longer_on_the_parent():
    """The copy-pasted flat fields are gone: passing them is now an error."""
    for legacy in ("POPULARITY_STRATEGY", "SESSION_COVIS_ENABLED", "COVIS_TOP_K", "BLEND_RRF_K"):
        with pytest.raises(Exception):
            ALSSettings(**BASE, **{legacy: 1})


def test_composition_is_not_on_the_als_ranker_config():
    """ALSSettings is a leaf: composing a model set through it is an error.

    This is the whole point of the split - "who serves cold users" and "how are
    two rankers fused" are not hyperparameters of ALS.
    """
    for field, value in (
        ("popular", PopularSettings()),
        ("covis", CoVisSettings()),
        ("blend", BlendSettings()),
    ):
        with pytest.raises(Exception):
            ALSSettings(**BASE, **{field: value})


def test_als_passes_the_model_sets_covis_config_to_the_sub_model(dataset):
    config = ModelSetSettings(
        als=ALSSettings(**BASE),
        covis=CoVisSettings(COVIS_MIN_COOC=2, COVIS_SESSION_WEIGHTS=True),
    )
    model = RecommenderALS(recsys_config=config, model_name="als_test", model_version="1")
    model.train(dataset)
    assert model.covis.recsys_config is config.covis


# --- back-compat: artifacts pickled with the old flat shape -------------------


def test_legacy_pickle_migrates_flat_fields_into_sub_configs():
    config = _legacy_config()
    assert isinstance(config, ALSSettings)
    assert config.popular.POPULARITY_STRATEGY == "n_users"
    assert config.popular.POPULARITY_PERIOD == timedelta(days=14)
    assert config.covis is not None
    assert config.covis.COVIS_TOP_K == 100
    assert config.covis.COVIS_MIN_COOC == 2
    assert config.covis.COVIS_SESSION_WEIGHTS is False
    assert config.blend.ALS_WEIGHT == 1.0
    assert config.blend.COVIS_WEIGHT == 1.0
    assert config.blend.RRF_K == 60


def test_legacy_pickle_without_the_flag_has_no_session_layer():
    config = _legacy_config(SESSION_COVIS_ENABLED=False)
    assert config.covis is None


def test_legacy_pickle_predating_covis_fields_entirely():
    """An als_youtravel artifact from before the covis layer shipped."""
    flat = {k: v for k, v in LEGACY_ALS_COVIS_DICT.items() if not k.startswith(("SESSION_", "COVIS_", "BLEND_"))}
    config = pickle.loads(pickle.dumps(LegacyALSSettings(flat)))
    assert config.covis is None
    assert config.popular.POPULARITY_STRATEGY == "n_users"
    assert config.blend.RRF_K == 60


def test_legacy_pickle_keeps_the_old_attributes_readable():
    """Stale flat keys stay in __dict__ so straggler readers do not crash."""
    config = _legacy_config()
    assert config.POPULARITY_STRATEGY == "n_users"
    assert config.BLEND_RRF_K == 60


def test_legacy_blend_weights_are_actually_used():
    config = _legacy_config(BLEND_ALS_WEIGHT=3.5, BLEND_COVIS_WEIGHT=0.25, BLEND_RRF_K=11)
    assert config.blend.ALS_WEIGHT == 3.5
    assert config.blend.COVIS_WEIGHT == 0.25
    assert config.blend.RRF_K == 11


# --- back-compat, end to end: a whole model.pkl written with the old shape ----


def test_legacy_artifact_loads_and_recommends_identically(dataset, tmp_path):
    """Train, write model.pkl with a LEGACY-shaped recsys_config, reload it with
    the current code, and assert the recommender builds and serves the same
    recommendations and strategies as an equivalent new-shape model."""
    new_config = ModelSetSettings(
        als=ALSSettings(**BASE),
        popular=PopularSettings(POPULARITY_STRATEGY="n_users", POPULARITY_PERIOD=timedelta(days=14)),
        covis=CoVisSettings(RECOMMENDER_DAYS_THRESHOLD=30, COVIS_TOP_K=100, COVIS_MIN_COOC=2),
    )
    reference = RecommenderALS(recsys_config=new_config, model_name="als_covis_youtravel", model_version="1")
    reference.train(dataset)

    # Same trained artifact, but with the pre-refactor config object inside.
    legacy_state = dict(reference.__dict__)
    legacy_state["recsys_config"] = LegacyALSSettings(LEGACY_ALS_COVIS_DICT)
    save_dir = tmp_path / "als_covis_youtravel"
    save_dir.mkdir()
    with open(save_dir / "model.pkl", "wb") as file:
        dill.dump(legacy_state, file)

    loaded = RecommenderALS.load_model(load_dir=str(save_dir))
    assert isinstance(loaded.recsys_config, ALSSettings)
    # The recommender does not read composition off recsys_config any more; the
    # normalising view is what has to see the session layer.
    assert loaded.model_set.covis is not None

    cases = [
        dict(user_ids="u1", history=["m2", "m1"]),  # hot + session -> ALS x CoVis blend
        dict(user_ids="ghost-user", history=["m1", "m2"]),  # unknown + session -> CoVis
        dict(user_ids="u1", history=None),  # hot, no session -> pure ALS
        dict(user_ids="ghost-user", history=None),  # cold -> popular
    ]
    for case in cases:
        got = loaded.recommend(top_n=3, filter_viewed=True, **case)
        want = reference.recommend(top_n=3, filter_viewed=True, **case)
        assert got.strategy == want.strategy, case
        assert got.item_ids == want.item_ids, case
        assert got.scores == want.scores, case

    # And the blend path really did run off the migrated config: the covis layer
    # is present and the hot-plus-session request reports that segment.
    assert loaded.covis is not None
    assert loaded.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"]).strategy == (
        Strategy.MODEL_REALTIME_HOT_USERS.value
    )


def test_legacy_nested_config_is_read_as_a_model_set():
    """The shape shipped between the two refactors: composition on ALSSettings.

    Nothing rebuilds these by migration - they arrive as an ALSSettings with the
    sub-configs already in __dict__, and `model_set` has to read them from there.
    """
    config = ALSSettings(**BASE)
    config.__dict__["popular"] = PopularSettings(POPULARITY_STRATEGY="n_interactions")
    config.__dict__["covis"] = CoVisSettings(COVIS_TOP_K=33)
    config.__dict__["blend"] = BlendSettings(ALS_WEIGHT=2.5, COVIS_WEIGHT=0.5, RRF_K=7)

    model_set = ModelSetSettings.from_legacy_als_settings(config)
    assert model_set.als.ALS_FACTORS == BASE["ALS_FACTORS"]
    assert model_set.popular.POPULARITY_STRATEGY == "n_interactions"
    assert model_set.covis.COVIS_TOP_K == 33
    assert model_set.blend.ALS_WEIGHT == 2.5
    assert model_set.blend.RRF_K == 7


def test_legacy_artifact_blends_with_its_own_weights_not_defaults(dataset):
    """The silent failure the `model_set` shim exists to prevent.

    `blend` is the only composition field read on the serving path. If a legacy
    artifact were read as a bare ModelSetSettings, nothing would raise - the RRF
    fusion would just run with default 1.0/1.0/60 instead of the trained weights,
    and the feed would quietly change ranking with no error anywhere.
    """
    trained = _legacy_config(BLEND_ALS_WEIGHT=4.0, BLEND_COVIS_WEIGHT=0.1, BLEND_RRF_K=13)
    model = RecommenderALS(recsys_config=trained, model_name="als_covis_youtravel", model_version="1")

    blend = model.model_set.blend
    assert (blend.ALS_WEIGHT, blend.COVIS_WEIGHT, blend.RRF_K) == (4.0, 0.1, 13)
    assert blend != BlendSettings(), "read the trained weights, not the defaults"


def test_legacy_covis_artifact_loads_and_recommends(dataset, tmp_path):
    """The standalone covis artifact pickles a CoVisSettings; its fields are
    unchanged by this refactor, so a plain round-trip must still serve."""
    model = RecommenderCoVis(
        recsys_config=CoVisSettings(RECOMMENDER_DAYS_THRESHOLD=30, COVIS_MIN_COOC=2),
        model_name="covis_youtravel",
        model_version="1",
    )
    model.train(dataset)

    save_dir = tmp_path / "covis_youtravel"
    save_dir.mkdir()
    with open(save_dir / "model.pkl", "wb") as file:
        dill.dump(model.__dict__, file)

    loaded = RecommenderCoVis.load_model(load_dir=str(save_dir))
    assert set(loaded.__dict__) >= {"neighbors", "top_k", "session_weights_enabled", "recsys_config"}
    assert isinstance(loaded.recsys_config, CoVisSettings)
    got = loaded.recommend("u-x", top_n=3, filter_viewed=True, history=["m1", "t1"])
    want = model.recommend("u-x", top_n=3, filter_viewed=True, history=["m1", "t1"])
    assert got.item_ids == want.item_ids
    assert got.scores == want.scores
