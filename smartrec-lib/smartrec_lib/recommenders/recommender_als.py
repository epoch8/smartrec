import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
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
from rectools.models import ImplicitALSWrapperModel, PopularModel
from scipy import sparse

from smartrec_lib.model import ALSSettings, RecomItems, Strategy
from smartrec_lib.recommenders import RecommenderModel
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("ALS Model")
logger.setLevel(logging.INFO)


class RecommenderALS(RecommenderModel):
    model_architecture = "als"

    def __init__(
        self,
        recsys_config: Optional[ALSSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        # base and feature models
        self.model_hot_users: ImplicitALSWrapperModel = None
        self.model_cold_users: PopularModel = None
        self.dataset: Dataset = None  # might take unnecessary memory
        self.item_id_map: IdMap = None
        self.user_id_map: IdMap = None

    @classmethod
    def compute_item_similarity(cls, model: ImplicitALSWrapperModel, k_max_values: int = 100) -> np.ndarray:
        if not isinstance(model.model.item_factors, np.ndarray):
            item_factors = model.model.item_factors.to_numpy()
        else:
            item_factors = model.model.item_factors
        print(f"2 - {item_factors=}")
        matrix = item_factors.dot(item_factors.T)
        print(f"2 - {matrix=}")
        k_set_to_zero = matrix.shape[0] - k_max_values
        print(f"2 - {k_set_to_zero=}")
        mask = np.apply_along_axis(lambda x: x < np.partition(x, k_set_to_zero)[k_set_to_zero], 1, matrix)
        print(f"2 - {mask=}")
        matrix[mask] = 0

        sparse_matrix = sparse.csr_matrix(matrix)

        print(f"2 - {sparse_matrix=}")

        return sparse_matrix

    def train(self, dataset: Dataset):
        assert self.recsys_config is not None

        self.dataset = dataset

        logger.info("Fitting model...")

        np.random.seed(self.recsys_config.RECOMMENDER_RANDOM_STATE)

        self.model_hot_users = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=self.recsys_config.ALS_FACTORS,
                regularization=self.recsys_config.ALS_REGULARIZATION_FACTOR,
                iterations=self.recsys_config.ALS_ITERATIONS,
                alpha=self.recsys_config.ALS_ALPHA,
                random_state=self.recsys_config.RECOMMENDER_RANDOM_STATE,
            ),
            fit_features_together=False,  # way to fit paired features
        )
        self.model_hot_users.fit(dataset)
        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        self.item_similarity = self.compute_item_similarity(self.model_hot_users)

        self.model_cold_users = PopularModel(
            popularity=self.recsys_config.POPULARITY_STRATEGY,
            period=self.recsys_config.POPULARITY_PERIOD,
        )
        self.model_cold_users.fit(dataset)

        self.user_ids_hot = set(self.dataset.user_id_map.convert_to_external(self.dataset.interactions.df.user_id))
        # Some item_ids may not be in interactions
        self.item_ids_hot = set(self.dataset.item_id_map.convert_to_external(self.dataset.interactions.df.item_id))

        logger.info("Base models trained.")

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
    ) -> RecomItems:  # Return type is a RecomItems
        if isinstance(items_to_recommend, list) and len(items_to_recommend) == 0:
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY,
            )

        logger.info(f"Predicting for user {user_ids}")
        # sorting items that we have in interactions
        items_to_recommend_filtered = [
            item_to_recommend for item_to_recommend in items_to_recommend if item_to_recommend in self.item_ids_hot
        ]
        print(f"{items_to_recommend_filtered=}")

        recos: pd.DataFrame

        # user can be in the short memory, long memory or nowhere
        if user_ids in self.user_id_map.external_ids and user_ids in self.user_ids_hot:
            logger.info("Hot user")
            recos = self.model_hot_users.recommend(
                users=[user_ids],
                dataset=self.dataset,
                k=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=(
                    items_to_recommend_filtered
                    if items_to_recommend_filtered is None
                    else list(items_to_recommend_filtered)
                ),
            )
            strategy = Strategy.MODEL_HOT_USERS
        elif history and len(history):
            logger.info("Warm user")
            recos = self.recommend_unknown_user_with_history(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )
            print(f"{recos=}")
            strategy = Strategy.MODEL_WARM_USERS
        else:
            logger.info("Cold user")
            recos = self.model_cold_users.recommend(
                users=[user_ids],
                dataset=self.dataset,
                k=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=(
                    items_to_recommend_filtered
                    if items_to_recommend_filtered is None
                    else list(items_to_recommend_filtered)
                ),
            )
            strategy = Strategy.MODEL_COLD_USERS

        recos = recos.sort_values(["user_id", "score"], ascending=False).reset_index(
            drop=True
        )  # Assuming 'user_id' is the column name

        return RecomItems(
            item_ids=recos.item_id.astype(str).tolist(),
            scores=recos.score.tolist(),
            strategy=strategy,
        )

    def recommend_unknown_user_with_history(
        self,
        user_ids: int,
        history: List[str],
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
    ) -> RecomItems:
        # items_to_recommend processing
        if items_to_recommend is not None:
            items_to_recommend_enc = [
                self.item_id_map.convert_to_internal([item])
                for item in items_to_recommend
                if item in self.item_id_map.external_ids
            ]
            if len(items_to_recommend_enc) > 0:
                items_enc = np.array(items_to_recommend_enc)
                items_enc_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
                items_enc_v[:, items_enc] = 1
            else:
                return pd.DataFrame(columns=["user_id", "item_id", "score"])

        print(f"{items_enc_v=}")

        # history processing
        history_enc_list = [
            self.item_id_map.convert_to_internal([item]) for item in history if item in self.item_id_map.external_ids
        ]

        print(f"{history=}")

        print(f"{self.item_id_map.external_ids=}")

        history_enc = np.array(history_enc_list)
        history_dense_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
        history_dense_v[:, history_enc] = 1
        print(f"{history_dense_v=}")
        all_scores = self.item_similarity.T.dot(history_dense_v.T).T[0]
        print(f"{all_scores=}")
        sorted_item_ids = np.argsort(all_scores)

        print(f"{sorted_item_ids=}")

        items_mask = np.ones(len(sorted_item_ids), dtype=bool)

        if filter_viewed:
            filter_already_liked_mask = ~(history_dense_v[:, sorted_item_ids][0].astype(bool))
            items_mask = items_mask & filter_already_liked_mask

        if items_to_recommend:
            filter_item_mask = items_enc_v[:, sorted_item_ids][0].astype(bool)
            items_mask = items_mask & filter_item_mask

        sorted_item_ids = sorted_item_ids[items_mask]

        nearest_item_ids_enc = sorted_item_ids[: -top_n - 1 : -1]
        print(f"{nearest_item_ids_enc=}")
        nearest_scores = all_scores[nearest_item_ids_enc]

        result = pd.DataFrame(
            {
                "user_id": [user_ids] * len(nearest_item_ids_enc),
                "item_id": self.item_id_map.convert_to_external(nearest_item_ids_enc),
                "score": nearest_scores,
            }
        )

        return result

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
        clean_old_model_versions(base_s3_url=base_s3_url, model_name=self.model_name, num_to_keep=num_to_keep)
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
            "ALS_MODEL": ImplicitALSWrapperModel(
                AlternatingLeastSquares(
                    factors=self.recsys_config.ALS_FACTORS,
                    regularization=self.recsys_config.ALS_REGULARIZATION_FACTOR,
                    iterations=self.recsys_config.ALS_ITERATIONS,
                    alpha=self.recsys_config.ALS_ALPHA,
                    random_state=self.recsys_config.RECOMMENDER_RANDOM_STATE,
                ),
                fit_features_together=False,  # way to fit paired features
            ),
            "POPULARITY_MODEL": PopularModel(
                popularity=self.recsys_config.POPULARITY_STRATEGY,
                period=self.recsys_config.POPULARITY_PERIOD,
            ),
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
