from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel
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
    
def test_fit_als():
    df_interactions = pd.read_csv(
        test_data / "interactions.csv",
        header=0,
        names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    )
    
    train_dataset = Dataset.construct(
        interactions_df=df_interactions,
    )

    model_name = "als_build_ideas"
    model_version = str(int(datetime.now().timestamp()))
    model = RecommenderALS(
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
    )
    print(predictions)
    assert df_interactions.shape[0] == 100
    assert predictions.item_ids == ['589', '1253', '3578']
    assert predictions.scores == [0.0011700484901666641, 0.0011568143963813782, 0.0011557340621948242]
    assert predictions.strategy == 'model_hot_users'
