from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings, PopularSettings, Strategy
from smartrec_lib.recommenders import RecommenderALS

recsys_config_als = ALSSettings(
    ALS_ITERATIONS=10,
    RECOMMENDER_RANDOM_STATE=42,
    ALS_REGULARIZATION_FACTOR=0.2,
    ALS_FACTORS=256,  # latent embeddings size
    ALS_ALPHA=50,  # confidence multiplier for non-zero entries in interactions
    RECOMMENDER_DAYS_THRESHOLD=14,
    popular=PopularSettings(POPULARITY_STRATEGY="n_users", POPULARITY_PERIOD=timedelta(days=1)),
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

    candidates = [3105, 1193, 3468, 434, 1217]
    predictions = model.recommend(
        user_ids=0,
        top_n=5,
        filter_viewed=True,
        items_to_recommend=candidates,
    )

    assert df_interactions.shape[0] == 100
    assert predictions.strategy == Strategy.MODEL_COLD_USERS.value
    # An unknown user gets the popularity fallback restricted to `candidates`,
    # so the returned set is fully determined. Only 3105 has two users; the
    # other four tie at one and their relative order is arbitrary - see
    # CLAUDE.md section 8.
    assert set(predictions.item_ids) == {str(item_id) for item_id in candidates}
    assert predictions.item_ids[0] == "3105"
    assert predictions.scores == [2.0, 1.0, 1.0, 1.0, 1.0]
