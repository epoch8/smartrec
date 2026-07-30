from smartrec_lib.evaluation import evaluate_e2e

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
}
POPULAR_ONLY = {"cls": "PopularModel"}


def test_e2e_reports_segments(dataset):
    table = evaluate_e2e(
        dataset,
        {"policy": POLICY_CONFIG, "popular": POPULAR_ONLY},
        k=3,
        n_splits=2,
        test_size="2D",
    )
    assert ("policy", "all") in table.index
    assert ("policy", "hot") in table.index
    assert ("policy", "cold") in table.index
    # cold segment exists in the fixture (u7 appears only on the last day)
    assert table.loc[("policy", "cold"), "n_users"] >= 1
    # popular fallback answers cold users -> full coverage on every segment
    assert table.loc[("policy", "cold"), "covered@3"] == 1.0
