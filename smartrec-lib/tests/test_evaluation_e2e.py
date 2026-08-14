from rectools.dataset import Dataset
from rectools.model_selection import TimeRangeSplitter
from rectools.models.serialization import model_from_config

from smartrec_lib.evaluation import evaluate_e2e
from smartrec_lib.evaluation.next_item import _fold_frames

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


def test_e2e_fold_dataset_keeps_item_features_for_share_cap(dataset_with_features):
    """
    Regression test for the e2e fold dataset bug: evaluate_e2e must build its
    per-fold train dataset via `dataset.filter_interactions(...)`, not
    `Dataset.construct(interactions_df=...)` on an external-id frame - the
    latter drops item features, PolicyModel._build_item_category then returns
    {} and the category share cap silently becomes a no-op inside the e2e
    protocol (it still works fine in isolation, e.g. test_policy_model.py,
    which never goes through a fold rebuild).
    """
    config = dict(POLICY_CONFIG, category_feature="tour_country_ru", category_share_cap=0.34)

    # 1. The e2e protocol must run end-to-end with the cap configured.
    table = evaluate_e2e(dataset_with_features, {"policy": config}, k=3, n_splits=2, test_size="2D")
    assert not table.empty

    # 2. Rebuild the exact fold train dataset evaluate_e2e now uses (same
    # splitter, same call) and confirm item_category is populated - i.e. the
    # cap has something to act on.
    splitter = TimeRangeSplitter(
        test_size="2D",
        n_splits=2,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=False,
    )
    train_ids, test_ids, _info = next(iter(splitter.split(dataset_with_features.interactions)))
    fold_dataset = dataset_with_features.filter_interactions(
        row_indexes_to_keep=train_ids,
        keep_external_ids=True,
        keep_features_for_removed_entities=True,
    )
    policy = model_from_config(config)
    policy.fit(fold_dataset)
    assert policy.item_category  # non-empty: features survived filter_interactions

    # 3. The OLD approach (Dataset.construct from an external-id frame) drops
    # item features entirely -> item_category ends up empty and the cap is a
    # no-op. This proves the fix (2) addresses a real, observable regression.
    train_ext, _test_ext = _fold_frames(dataset_with_features, train_ids, test_ids)
    broken_fold_dataset = Dataset.construct(interactions_df=train_ext)
    broken_policy = model_from_config(config)
    broken_policy.fit(broken_fold_dataset)
    assert broken_policy.item_category == {}
