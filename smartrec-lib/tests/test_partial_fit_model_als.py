"""
Test partial fit (incremental training) functionality for ALS recommender model.
"""

import pandas as pd
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings, Strategy
from smartrec_lib.recommenders.recommender_als import RecommenderALS


def test_train_partial_als() -> None:
    """Test that train_partial correctly updates the model with new data."""

    # Initial training data
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

    # Create initial dataset
    initial_dataset = Dataset.construct(initial_interactions)

    # Initialize and train model
    settings = ALSSettings(
        ALS_FACTORS=8,
        ALS_ITERATIONS=10,
        ALS_REGULARIZATION_FACTOR=0.01,
        ALS_ALPHA=1,
        RECOMMENDER_RANDOM_STATE=42,
    )

    model = RecommenderALS(recsys_config=settings)
    model.train(initial_dataset)

    # Get initial recommendations for user 1
    initial_recommendations = model.recommend(user_ids=1, top_n=5, filter_viewed=False)

    assert len(initial_recommendations.item_ids) > 0
    assert initial_recommendations.strategy == Strategy.MODEL_HOT_USERS.value

    # Store number of users and items before partial fit
    initial_num_users = len(model.user_ids_hot)
    initial_num_items = len(model.item_ids_hot)

    print(f"Initial number of hot users: {initial_num_users}")
    print(f"Initial number of hot items: {initial_num_items}")
    print(f"Initial recommendations: {initial_recommendations.item_ids}")

    # New interactions for partial fit
    new_interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 5, 5, 6],  # Users 5 and 6 are new
            "item_id": [60, 60, 70, 80, 80],  # Items 60, 70, 80 are new
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(
                [
                    "2021-01-09",
                    "2021-01-10",
                    "2021-01-11",
                    "2021-01-12",
                    "2021-01-13",
                ]
            ),
        }
    )

    # Combine old and new interactions for partial fit
    # (In real scenario, you might only pass new interactions depending on your use case)
    combined_interactions = pd.concat([initial_interactions, new_interactions], ignore_index=True)
    new_dataset = Dataset.construct(combined_interactions)

    # Perform partial fit
    model.train_partial(new_dataset)

    # Verify that the model was updated
    updated_num_users = len(model.user_ids_hot)
    updated_num_items = len(model.item_ids_hot)

    print(f"Updated number of hot users: {updated_num_users}")
    print(f"Updated number of hot items: {updated_num_items}")

    # Check that new users and items were added
    assert updated_num_users > initial_num_users, "New users should be added after train_partial"
    assert updated_num_items > initial_num_items, "New items should be added after train_partial"

    # Verify that new users are in the model
    assert 5 in model.user_ids_hot, "User 5 should be in hot users after train_partial"
    assert 6 in model.user_ids_hot, "User 6 should be in hot users after train_partial"

    # Verify that new items are in the model
    assert 60 in model.item_ids_hot, "Item 60 should be in hot items after train_partial"
    assert 70 in model.item_ids_hot, "Item 70 should be in hot items after train_partial"
    assert 80 in model.item_ids_hot, "Item 80 should be in hot items after train_partial"

    # Get recommendations after partial fit
    updated_recommendations = model.recommend(user_ids=1, top_n=5, filter_viewed=False)
    print(f"Updated recommendations: {updated_recommendations.item_ids}")

    # Verify recommendations are still generated
    assert len(updated_recommendations.item_ids) > 0
    assert updated_recommendations.strategy == Strategy.MODEL_HOT_USERS.value

    # Test recommendations for a new user
    new_user_recommendations = model.recommend(user_ids=5, top_n=5, filter_viewed=False)
    print(f"New user (5) recommendations: {new_user_recommendations.item_ids}")

    assert len(new_user_recommendations.item_ids) > 0
    assert new_user_recommendations.strategy == Strategy.MODEL_HOT_USERS.value

    print("Partial fit test passed successfully!")


def test_train_partial_error_before_train():
    """Test that train_partial raises error if called before train."""

    settings = ALSSettings(
        ALS_FACTORS=8,
        ALS_ITERATIONS=10,
        ALS_REGULARIZATION_FACTOR=0.01,
        ALS_ALPHA=1,
        RECOMMENDER_RANDOM_STATE=42,
    )

    model = RecommenderALS(recsys_config=settings)

    # Try to call train_partial before train
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
        print(f"Expected error caught: {e}")


if __name__ == "__main__":
    test_train_partial_als()
    test_train_partial_error_before_train()
    print("\n✅ All train_partial tests passed!")
