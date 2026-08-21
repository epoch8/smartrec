from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings, ModelSetSettings, PopularSettings, Strategy
from smartrec_lib.recommenders import RecommenderModelSet

recsys_config_als = ModelSetSettings(
    als=ALSSettings(
        ALS_ITERATIONS=10,
        RECOMMENDER_RANDOM_STATE=42,
        ALS_REGULARIZATION_FACTOR=0.2,
        ALS_FACTORS=256,  # latent embeddings size
        ALS_ALPHA=50,  # confidence multiplier for non-zero entries in interactions
        RECOMMENDER_DAYS_THRESHOLD=14,
    ),
    popular=PopularSettings(POPULARITY_STRATEGY="n_users", POPULARITY_PERIOD=timedelta(days=1)),
)

test_data = Path(__file__).parent


def test_fit_als():
    df_interactions = pd.read_csv(
        test_data / "interactions.csv",
        header=0,
        names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    )

    train_dataset = Dataset.construct(
        interactions_df=df_interactions,
    )

    model_name = "als_test_model"
    model_version = str(int(datetime.now().timestamp()))
    model = RecommenderModelSet(
        model_name=model_name,
        model_version=model_version,
        recsys_config=recsys_config_als,
    )

    # metrics = model.calc_metrics(k=10, dataset=train_dataset)
    model.train(train_dataset)

    predictions = model.recommend(
        user_ids=1,
        top_n=3,
        filter_viewed=True,
        items_to_recommend=[589, 1253, 3578],
    )
    print(f"{predictions=}")

    # Basic assertions
    assert df_interactions.shape[0] == 100
    assert len(predictions.item_ids) == 3
    assert set(predictions.item_ids) == set(["1253", "589", "3578"])
    assert predictions.strategy == Strategy.MODEL_HOT_USERS.value

    # Check scores are in reasonable range and sorted in descending order
    assert len(predictions.scores) == 3
    assert all(0.0 < score < 1.0 for score in predictions.scores)
    assert predictions.scores == sorted(predictions.scores, reverse=True)
