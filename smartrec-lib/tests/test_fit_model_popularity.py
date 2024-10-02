from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import PopularSettings
from smartrec_lib.recommenders import RecommenderPopular

recsys_config_popular = PopularSettings(
    RECOMMENDER_DAYS_THRESHOLD=2,
    POPULARITY_STRATEGY="n_users",
    POPULARITY_PERIOD=timedelta(days=7),
)

test_data = Path(__file__).parent


def test_fit_popular():
    df_interactions = pd.read_csv(
        test_data / "interactions.csv",
        header=0,
        names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    )

    train_dataset = Dataset.construct(
        interactions_df=df_interactions,
    )

    model_name = "popularity_test_model"
    model_version = str(int(datetime.now().timestamp()))
    model = RecommenderPopular(
        model_name=model_name,
        model_version=model_version,
        recsys_config=recsys_config_popular,
    )

    # metrics = model.calc_metrics(k=10, dataset=train_dataset)
    model.train(train_dataset)

    predictions = model.recommend(
        user_ids=1,
        top_n=3,
        filter_viewed=True,
    )
    print(predictions)
    assert df_interactions.shape[0] == 100
    assert predictions.item_ids == ['3468', '434', '1217']
    assert predictions.scores == [1.0, 1.0, 1.0]
    assert predictions.strategy == "model_hot_and_cold_users"
