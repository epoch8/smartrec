from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from smartrec_lib.model import ALSSettings, Strategy
from smartrec_lib.recommenders import RecommenderALS

recsys_config_als = ALSSettings(
    ALS_ITERATIONS=10,
    RECOMMENDER_RANDOM_STATE=42,
    ALS_REGULARIZATION_FACTOR=0.2,
    ALS_FACTORS=256,  # latent embeddings size
    ALS_ALPHA=50,  # confidence multiplier for non-zero entries in interactions
    RECOMMENDER_DAYS_THRESHOLD=14,
    POPULARITY_STRATEGY="n_users",
    POPULARITY_PERIOD=timedelta(days=1),
)

test_data = Path(__file__).parent


def test_fit_als():
    df_interactions = pd.read_csv(
        test_data / "interactions.csv",
        header=0,
        names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    )

    print(f"1 - {df_interactions.head(20)=}")
    print(f"1 - {df_interactions.tail(20)=}")
    train_dataset = Dataset.construct(
        interactions_df=df_interactions,
    )
    print(f"1 - {train_dataset=}")

    model_name = "als_test_model"
    model_version = str(int(datetime.now().timestamp()))
    model = RecommenderALS(
        model_name=model_name,
        model_version=model_version,
        recsys_config=recsys_config_als,
    )

    # metrics = model.calc_metrics(k=10, dataset=train_dataset)
    model.train(train_dataset)

    predictions = model.recommend(
        user_ids=23901319232,
        top_n=3,
        filter_viewed=False,
        items_to_recommend=[110, 589, 661, 914, 1188, 1193, 1253, 1259, 2398, 3030, 3108, 3408, 2804, 3256, 3578],
        history=[595, 661, 1253, 2398, 3108, 3408, 2804, 3256, 3578],
    )
    print(f"{predictions=}")
    
    # Basic assertions
    assert df_interactions.shape[0] == 100
    assert len(predictions.item_ids) == 3
    assert predictions.strategy == Strategy.MODEL_WARM_USERS.value
    
    # Check that recommended items are from the items_to_recommend list (not from history)
    items_to_recommend = [110, 589, 661, 914, 1188, 1193, 1253, 1259, 2398, 3030, 3108, 3408, 2804, 3256, 3578]
    assert all(int(item_id) in items_to_recommend for item_id in predictions.item_ids)
    
    # Check scores are in reasonable range and sorted in descending order
    assert len(predictions.scores) == 3
    assert all(0.0 < score < 1.0 for score in predictions.scores)
    assert predictions.scores == sorted(predictions.scores, reverse=True)
    
    # Check that recommended items are similar to history items (should be items from history that are also in items_to_recommend)
    # Items 661, 2398, 3108, 3408, 2804, 3256, 3578, 1253 are in both history and items_to_recommend
    history_items_in_recommend = [661, 2398, 3108, 3408, 2804, 3256, 3578, 1253]
    # At least some of the recommendations should be from this intersection (since filter_viewed=False)
    recommended_item_ids_int = [int(item_id) for item_id in predictions.item_ids]
    assert any(item_id in history_items_in_recommend for item_id in recommended_item_ids_int)
