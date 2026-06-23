"""
Test partial fit (incremental training) functionality for ALS recommender model.
"""

import pandas as pd
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings, Strategy
from smartrec_lib.recommenders.recommender_als import RecommenderALS


def _make_settings() -> ALSSettings:
    return ALSSettings(
        ALS_FACTORS=8,
        ALS_ITERATIONS=10,
        ALS_REGULARIZATION_FACTOR=0.01,
        ALS_ALPHA=1,
        RECOMMENDER_RANDOM_STATE=42,
    )


def _initial_dataset() -> Dataset:
    initial_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "item_id": [10, 20, 20, 30, 30, 40, 40, 50],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                    "2021-01-08",
                ]
            ),
        }
    )
    return Dataset.construct(initial_interactions)


def test_train_partial_als_existing_entities() -> None:
    """train_partial incrementally updates the model on EXISTING users/items.

    Per the current contract, train_partial only supports incremental updates for
    entities seen during the initial train(); new users/items are rejected (see
    test_train_partial_rejects_new_entities). So here we feed only new interaction
    pairs among existing users/items and assert the update succeeds without
    changing the known entity set.
    """

    model = RecommenderALS(recsys_config=_make_settings())
    model.train(_initial_dataset())

    initial_recommendations = model.recommend(user_ids=1, top_n=5, filter_viewed=False)
    assert len(initial_recommendations.item_ids) > 0
    assert initial_recommendations.strategy == Strategy.MODEL_HOT_USERS.value

    initial_num_users = len(model.user_ids_hot)
    initial_num_items = len(model.item_ids_hot)

    # New interactions for EXISTING users and items only (new user-item pairs).
    new_interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "item_id": [30, 50, 10, 20],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(
                ["2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12"]
            ),
        }
    )
    new_dataset = Dataset.construct(new_interactions)

    # Should complete without error - no new entities involved.
    model.train_partial(new_dataset)

    # Entity set is unchanged (incremental update, not a structural change).
    assert len(model.user_ids_hot) == initial_num_users
    assert len(model.item_ids_hot) == initial_num_items

    # Model still serves recommendations for an existing user.
    updated_recommendations = model.recommend(user_ids=1, top_n=5, filter_viewed=False)
    assert len(updated_recommendations.item_ids) > 0
    assert updated_recommendations.strategy == Strategy.MODEL_HOT_USERS.value


def test_train_partial_rejects_new_entities() -> None:
    """train_partial must reject datasets containing new users or items.

    The incremental training job relies on this ValueError to fall back to a full
    retrain (see app/experiments/incremental_training.py).
    """

    model = RecommenderALS(recsys_config=_make_settings())
    model.train(_initial_dataset())

    # Users 5, 6 and items 60, 70, 80 are new.
    new_interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 5, 5, 6],
            "item_id": [60, 60, 70, 80, 80],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(
                ["2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13"]
            ),
        }
    )
    new_dataset = Dataset.construct(new_interactions)

    try:
        model.train_partial(new_dataset)
        assert False, "train_partial should raise ValueError on new users/items"
    except ValueError as e:
        assert "cannot add new entities" in str(e)


def test_train_partial_error_before_train() -> None:
    """Test that train_partial raises error if called before train."""

    model = RecommenderALS(recsys_config=_make_settings())

    interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "weight": [1.0, 1.0],
            "datetime": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        }
    )
    dataset = Dataset.construct(interactions)

    try:
        model.train_partial(dataset)
        assert False, "Should raise AssertionError when train_partial is called before train"
    except AssertionError as e:
        assert "Model must be trained before train_partial" in str(e)


if __name__ == "__main__":
    test_train_partial_als_existing_entities()
    test_train_partial_rejects_new_entities()
    test_train_partial_error_before_train()
    print("\nAll train_partial tests passed!")
