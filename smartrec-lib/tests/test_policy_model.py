import pytest
from rectools import Columns
from rectools.models import PopularModel, model_from_config

from smartrec_lib.models import CoVisModel
from smartrec_lib.policy import PolicyModel, PolicyModelConfig, SourceSpec

POLICY_CONFIG = {
    "cls": "smartrec_lib.policy.model.PolicyModel",
    "sources": {
        "popular": {"model": {"cls": "PopularModel"}, "weight": 1.0},
        "covis": {
            "model": {"cls": "smartrec_lib.models.covis.CoVisModel", "min_cooc": 2},
            "weight": 1.0,
            "is_session": True,
        },
    },
    "fallback_source": "popular",
    "category_feature": "tour_country_ru",
    "category_share_cap": 1.0,
    "session_weight_tiers": [(1, 1.0)],
}


def test_fit_trains_all_sources(dataset):
    policy = model_from_config(POLICY_CONFIG)
    policy.fit(dataset)
    assert isinstance(policy.models["popular"], PopularModel)
    assert isinstance(policy.models["covis"], CoVisModel)
    assert policy.models["covis"].neighbors  # actually fitted


def test_fit_builds_external_item_categories(dataset_with_features):
    policy = model_from_config(POLICY_CONFIG)
    policy.fit(dataset_with_features)
    assert policy.item_category["m1"] == "maldives"
    assert policy.item_category["t2"] == "turkey"


def test_fallback_must_be_a_source(dataset):
    config = dict(POLICY_CONFIG, fallback_source="missing")
    policy = model_from_config(config)
    with pytest.raises(ValueError, match="fallback_source"):
        policy.fit(dataset)


def test_config_roundtrip():
    policy = model_from_config(POLICY_CONFIG)
    simple = policy.get_config(simple_types=True)
    restored = model_from_config(simple)
    assert isinstance(restored, PolicyModel)
    assert restored.source_specs["covis"].is_session is True
