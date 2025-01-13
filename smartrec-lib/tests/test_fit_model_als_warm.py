from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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

    train_dataset = Dataset.construct(
        interactions_df=df_interactions,
    )

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
    assert df_interactions.shape[0] == 100
    assert set(predictions.item_ids) == set(["2398", "2804", "661"])
    np.testing.assert_allclose(predictions.scores, [0.01239, 0.01238, 0.0], atol=1e-5)
    assert predictions.strategy == Strategy.MODEL_WARM_USERS.value
