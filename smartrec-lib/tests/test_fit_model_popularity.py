from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import PopularSettings, Strategy
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
    viewed = set(df_interactions.loc[df_interactions[Columns.User] == 1, Columns.Item].astype(str))
    n_users = df_interactions.groupby(Columns.Item)[Columns.User].nunique()

    assert df_interactions.shape[0] == 100
    assert predictions.strategy == Strategy.MODEL_COLD_USERS.value
    assert len(predictions.item_ids) == 3
    # The fixture has 98 items tied at one user and exactly one (3105) with two,
    # and 3105 sits in user 1's history, so filter_viewed drops it. What is
    # determined is therefore: nothing viewed comes back, and everything
    # returned belongs to the one-user tie group. Pinning a specific triple
    # would pin an arbitrary slice of 98 equals - see CLAUDE.md section 8.
    assert not viewed & set(predictions.item_ids)
    assert all(n_users[int(item_id)] == 1 for item_id in predictions.item_ids)
    assert predictions.scores == [1.0, 1.0, 1.0]
