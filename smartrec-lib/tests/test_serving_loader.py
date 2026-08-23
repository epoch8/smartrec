"""What Triton actually loads, per pickle shape.

`serving/model.py` is the deployed file - the trainer overwrites it in S3 on every
run - and its `_load_model` picks the class that will answer every request. It
cannot go by artifact name: `als_covis_youtravel` was a RecommenderALS __dict__
before 2026-08-22 and is a RecommenderModelSet __dict__ after, under the SAME
name, because the name is a public query parameter (CLAUDE.md §5.9). Choosing
wrong here does not raise, it serves the wrong thing.

The module imports Triton's C-extension helper, which does not exist off-cluster,
so it is stubbed. Everything else is the real code path.
"""

import sys
import types
from datetime import timedelta

import dill
import pytest

from smartrec_lib.model import ALSSettings, CoVisSettings, ModelSetSettings, PopularSettings
from smartrec_lib.recommenders import RecommenderCoVis, RecommenderModelSet

BASE = dict(
    ALS_ITERATIONS=2,
    ALS_REGULARIZATION_FACTOR=0.01,
    ALS_FACTORS=8,
    ALS_ALPHA=10,
    RECOMMENDER_DAYS_THRESHOLD=30,
)


@pytest.fixture(scope="module")
def triton_model_module():
    """Import serving/model.py with Triton's backend utils stubbed out."""
    stub = types.ModuleType("triton_python_backend_utils")

    class _Logger:
        @staticmethod
        def log_info(_message):
            return None

        @staticmethod
        def log_error(_message):
            return None

    stub.Logger = _Logger
    sys.modules.setdefault("triton_python_backend_utils", stub)

    from smartrec_lib.serving import model as serving_model

    return serving_model


def _trained_set(dataset, *, with_session):
    config = ModelSetSettings(
        main=ALSSettings(**BASE),
        fallback=PopularSettings(POPULARITY_STRATEGY="n_users", POPULARITY_PERIOD=timedelta(days=14)),
        session=CoVisSettings(COVIS_MIN_COOC=2) if with_session else None,
    )
    model_set = RecommenderModelSet(
        recsys_config=config,
        model_name="als_covis_youtravel" if with_session else "als_youtravel",
        model_version="1",
    )
    model_set.train(dataset)
    return model_set


def _write(tmp_path, name, state):
    save_dir = tmp_path / name
    save_dir.mkdir()
    with open(save_dir / "model.pkl", "wb") as file:
        dill.dump(state, file)
    return str(save_dir)


def test_a_model_set_pickle_loads_as_a_model_set(triton_model_module, dataset, tmp_path):
    model_set = _trained_set(dataset, with_session=True)
    load_dir = _write(tmp_path, "als_covis_youtravel", model_set.__dict__)

    loaded = triton_model_module.TritonPythonModel._load_model(model_name="als_covis_youtravel", load_dir=load_dir)

    assert isinstance(loaded, RecommenderModelSet)
    assert loaded.session is not None
    assert loaded.recommend("u1", top_n=3, filter_viewed=True).item_ids


def test_a_legacy_als_pickle_loads_as_an_adapted_model_set(triton_model_module, dataset, tmp_path):
    """The shape every artifact in both buckets has RIGHT NOW."""
    model_set = _trained_set(dataset, with_session=True)
    legacy_state = {
        "model_name": "als_covis_youtravel",
        "model_version": "1",
        "recsys_config": model_set.recsys_config.main,
        "model_hot_users": model_set.main.model_hot_users,
        "model_cold_users": model_set.fallback.model,
        "covis": model_set.session,
        "dataset": model_set.main.dataset,
        "item_id_map": model_set.main.item_id_map,
        "user_id_map": model_set.main.user_id_map,
        "item_similarity": model_set.main.item_similarity,
        "user_ids_hot": model_set.main.user_ids_hot,
        "item_ids_hot": model_set.main.item_ids_hot,
        "implicit_neginf_score": model_set.main.implicit_neginf_score,
        "user_item_matrix_binary": None,
    }
    load_dir = _write(tmp_path, "legacy_als_covis", legacy_state)

    loaded = triton_model_module.TritonPythonModel._load_model(model_name="als_covis_youtravel", load_dir=load_dir)

    # Adapted, not loaded as the old class: one routing path in the library.
    assert isinstance(loaded, RecommenderModelSet)
    assert loaded.main is not None and loaded.fallback is not None
    assert loaded.session is not None
    assert loaded.recommend("u1", top_n=3, filter_viewed=True, history=["m2", "m1"]).item_ids


def test_a_standalone_pickle_still_resolves_by_name(triton_model_module, dataset, tmp_path):
    """No members and no ALS model inside - fall through to the name ladder."""
    covis = RecommenderCoVis(
        recsys_config=CoVisSettings(COVIS_MIN_COOC=2), model_name="covis_youtravel", model_version="1"
    )
    covis.train(dataset)
    load_dir = _write(tmp_path, "covis_youtravel", covis.__dict__)

    loaded = triton_model_module.TritonPythonModel._load_model(model_name="covis_youtravel", load_dir=load_dir)

    assert isinstance(loaded, RecommenderCoVis)


def test_the_ladder_keeps_als_covis_away_from_the_bare_covis_branch(triton_model_module, dataset, tmp_path):
    """A regression that once served empty feeds.

    "als_covis_youtravel" contains "covis" as a substring. With independent ifs
    instead of an ordered chain, the covis branch overwrote the als one and the
    artifact loaded as an empty-neighbors RecommenderCoVis.
    """
    model_set = _trained_set(dataset, with_session=True)
    load_dir = _write(tmp_path, "als_covis_youtravel", model_set.__dict__)

    loaded = triton_model_module.TritonPythonModel._load_model(model_name="als_covis_youtravel", load_dir=load_dir)

    assert not isinstance(loaded, RecommenderCoVis)
    assert isinstance(loaded, RecommenderModelSet)
