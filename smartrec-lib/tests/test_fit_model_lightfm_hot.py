from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import LighFMSettings, Strategy
from smartrec_lib.recommenders import RecommenderLightFM

recsys_config_als = LighFMSettings(
    RECOMMENDER_RANDOM_STATE=42,
    RECOMMENDER_DAYS_THRESHOLD=14,
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

    model_name = "ligthfm_test_model"
    model_version = str(int(datetime.now().timestamp()))
    model = RecommenderLightFM(
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
    assert df_interactions.shape[0] == 100
    assert set(predictions.item_ids) == set(["3578", "589", "1253"])
    np.testing.assert_allclose(predictions.scores, [-0.6149794, -0.651881, -0.71531], atol=1e-5)
    assert predictions.strategy == Strategy.MODEL_HOT_AND_COLD_USERS.value
