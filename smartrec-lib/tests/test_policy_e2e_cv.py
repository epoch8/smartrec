from rectools.metrics import MAP, CoveredUsers, SufficientReco, UnrepeatedReco
from rectools.model_selection import TimeRangeSplitter, cross_validate
from rectools.models import model_from_config

POLICY_CONFIG = {
    "cls": "smartrec_lib.research.policy.PolicyModel",
    "sources": {
        "popular": {"model": {"cls": "PopularModel"}, "weight": 1.0},
        "covis": {
            "model": {"cls": "smartrec_lib.research.covis.CoVisModel", "min_cooc": 2},
            "weight": 1.0,
            "is_session": True,
        },
    },
    "fallback_source": "popular",
}


def test_policy_runs_under_standard_cross_validate_with_cold_users(dataset):
    splitter = TimeRangeSplitter(
        test_size="2D",
        n_splits=2,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=False,  # e2e: keep cold users in test
    )
    k = 3
    result = cross_validate(
        dataset=dataset,
        splitter=splitter,
        metrics={
            f"map@{k}": MAP(k=k),
            f"covered@{k}": CoveredUsers(k=k),
            f"sufficient@{k}": SufficientReco(k=k),
            f"unrepeated@{k}": UnrepeatedReco(k=k),
        },
        models={"policy": model_from_config(POLICY_CONFIG)},
        k=k,
        filter_viewed=True,
    )
    rows = result["metrics"]
    assert len(rows) == 2  # one row per fold
    for row in rows:
        # Popular fallback answers everyone: full coverage, full pages, no dups.
        assert row[f"covered@{k}"] == 1.0
        assert row[f"sufficient@{k}"] == 1.0
        assert row[f"unrepeated@{k}"] == 1.0
