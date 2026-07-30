import pytest
from rectools import Columns
from rectools.models import PopularModel, model_from_config

from smartrec_lib.models import CoVisModel
from smartrec_lib.policy import PolicyModel

ITEM_COUNTRY_LOCAL = {
    "m1": "maldives", "m2": "maldives", "m3": "maldives",
    "t1": "turkey", "t2": "turkey", "t3": "turkey",
    "pop1": "france",
}

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


def test_cold_user_gets_popular_fallback(dataset):
    policy = model_from_config(POLICY_CONFIG)
    policy.fit(dataset)
    reco = policy.recommend(users=["ghost-user"], dataset=dataset, k=3, filter_viewed=False)
    assert len(reco) == 3
    assert reco[Columns.Item].iloc[0] == "pop1"  # most popular item overall


def test_hot_user_fusion_includes_covis_signal(dataset):
    policy = model_from_config(POLICY_CONFIG)
    policy.fit(dataset)
    reco = policy.recommend(users=["u1"], dataset=dataset, k=3, filter_viewed=True)
    items = reco[Columns.Item].tolist()
    assert "m3" in items          # covis: strong maldives co-occurrence for u1
    assert "m1" not in items      # filter_viewed honoured through sources


def test_session_weight_zero_turns_covis_off(dataset):
    config = dict(POLICY_CONFIG, session_weight_tiers=[(1, 0.0)])
    with_covis_off = model_from_config(config)
    with_covis_off.fit(dataset)
    popular_only_cfg = {
        "cls": "smartrec_lib.policy.model.PolicyModel",
        "sources": {"popular": {"model": {"cls": "PopularModel"}, "weight": 1.0}},
        "fallback_source": "popular",
    }
    popular_only = model_from_config(popular_only_cfg)
    popular_only.fit(dataset)
    reco_a = with_covis_off.recommend(users=["u1"], dataset=dataset, k=3, filter_viewed=True)
    reco_b = popular_only.recommend(users=["u1"], dataset=dataset, k=3, filter_viewed=True)
    assert reco_a[Columns.Item].tolist() == reco_b[Columns.Item].tolist()


def test_share_cap_prevents_mono_feed(dataset_with_features):
    config = dict(POLICY_CONFIG, category_share_cap=0.34)  # at k=3: ceil(0.34 * 3) = 2 slots per country
    policy = model_from_config(config)
    policy.fit(dataset_with_features)
    reco = policy.recommend(users=["u2"], dataset=dataset_with_features, k=3, filter_viewed=False)
    countries = [ITEM_COUNTRY_LOCAL.get(item) for item in reco[Columns.Item]]
    assert countries.count("maldives") <= 2
