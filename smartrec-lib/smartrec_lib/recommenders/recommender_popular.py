import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pathy import Pathy
from rectools.dataset import Dataset
from rectools.dataset.identifiers import IdMap
from rectools.metrics import (
    MAP,
    AvgRecPopularity,
    Precision,
    Recall,
    Serendipity,
    novelty,
)
from rectools.model_selection import TimeRangeSplitter, cross_validate
from rectools.models import PopularModel

from smartrec_lib.model import PopularSettings, RecomItems
from smartrec_lib.recommenders import RecommenderModel
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("Popular Model")
logger.setLevel(logging.INFO)


class RecommenderPopular(RecommenderModel):
    model_architecture = "popular"

    def __init__(
        self,
        recsys_config: Optional[PopularSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        # base and feature models
        self.model: PopularModel = None
        self.dataset: Dataset = None  # might take unnecessary memory
        self.item_id_map: IdMap = None
        self.user_id_map: IdMap = None
        self.strategy = "model_hot_and_cold_users"

    def train(self, dataset: Dataset):
        assert self.recsys_config is not None

        self.dataset = dataset

        logger.info("Fitting model...")

        self.model = PopularModel(
            popularity=self.recsys_config.POPULARITY_STRATEGY,
            period=self.recsys_config.POPULARITY_PERIOD,
        )
        self.model.fit(dataset)
        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map

        logger.info("Base models trained.")

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
    ) -> RecomItems:  # Return type is a RecomItems
        logger.info(f"Predicting for user {user_ids}")
        # user can be in the short memory, long memory or nowhere
        recos: pd.DataFrame = self.model.recommend(
            users=[user_ids],
            dataset=self.dataset,
            k=top_n,
            filter_viewed=filter_viewed,
            items_to_recommend=(
                items_to_recommend
                if items_to_recommend is None
                else list(items_to_recommend)
            ),
        )

        recos = recos.sort_values(["user_id", "score"], ascending=False).reset_index(
            drop=True
        )  # Assuming 'user_id' is the column name

        return RecomItems(
            item_ids=recos.item_id.astype(str).tolist(),
            scores=recos.score.tolist(),
            strategy=self.strategy,
        )

    def save_model_triton(self, base_s3_url: Pathy, num_to_keep: int) -> None:
        """
        Save the model to either a file or a stream.

        Parameters:
            :param fs: fsspec filesystem object.
            :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
            :param num_to_keep: number of recent versions to keep.

        Returns:
            When saving to a file, returns None.
        """
        if self.model_version is None:
            raise Exception("There isn't model_version, please fill this field")

        logger.info(f"Saving model to {base_s3_url}")
        upload_model_files(
            base_s3_url,
            model_name=self.model_name,
            model_version=self.model_version,
            model_data=self.__dict__,
        )
        logger.info("Model saved successfully!")
        clean_old_model_versions(
            base_s3_url=base_s3_url, model_name=self.model_name, num_to_keep=num_to_keep
        )
        logger.info("Old models deleted!")

        return None

    def calc_metrics(self, k: int, dataset: Dataset) -> Dict[str, Any]:
        assert self.recsys_config is not None

        metrics = {
            f"serendipity@{k}": Serendipity(k=k),
            f"map@{k}": MAP(k=k),
            f"precision@{k}": Precision(k=k),
            f"recall@{k}": Recall(k=k),
            f"avgrecpopularity@{k}": AvgRecPopularity(k=1),
            f"novelty@{k}": novelty.NoveltyMetric(k=k),
        }

        models = {
            "POPULAR_MODEL": PopularModel(
                popularity=self.recsys_config.POPULARITY_STRATEGY,
                period=self.recsys_config.POPULARITY_PERIOD,
            )
        }

        n_splits = 3

        splitter = TimeRangeSplitter(
            test_size="4D",
            n_splits=n_splits,
            filter_already_seen=True,
            filter_cold_items=True,
            filter_cold_users=True,
        )

        cv_results = cross_validate(
            dataset=dataset,
            splitter=splitter,
            models=models,
            metrics=metrics,
            k=k,
            filter_viewed=True,
        )

        logger.info(f"The resutls of cross validate are - {cv_results}")

        return cv_results
