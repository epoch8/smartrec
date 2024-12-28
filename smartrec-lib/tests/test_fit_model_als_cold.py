from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings
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


def test_fit_als() -> None:
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
        user_ids=0,
        top_n=5,
        filter_viewed=True,
        items_to_recommend=[3105, 1193, 3468, 434, 1217],
    )

    assert df_interactions.shape[0] == 100
    assert predictions.item_ids == ["3105", "1193", "3468", "434", "1217"]
    assert predictions.scores == [2.0, 1.0, 1.0, 1.0, 1.0]
    assert predictions.strategy == "model_cold_users"
