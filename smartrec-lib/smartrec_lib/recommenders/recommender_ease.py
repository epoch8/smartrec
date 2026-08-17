import logging
from typing import Any, Dict, List, Optional

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
from rectools.models import EASEModel

from smartrec_lib.model import EASESettings, RecomItems, Strategy
from smartrec_lib.recommenders.base import RecommenderModel
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("EASE Model")
logger.setLevel(logging.INFO)


class RecommenderEASE(RecommenderModel):
    """
    Item-item recommender based on EASE^R (closed-form shallow autoencoder).

    Warm ranker only. Offline k-fold benchmarks (30-day training window) showed
    EASE^R outperforming the production ALS on both the click and the real-booking
    targets, with higher catalog coverage and lower popularity bias.

    Cold users (not present in training data) are NOT served here: recommend()
    returns an empty result with a cold strategy marker, and routing them (popular,
    segment popularity, co-visitation from the first click) is the responsibility
    of the orchestrator / cascade layer, not of this model.

    The `history` argument is accepted for interface compatibility but is not used
    yet; real-time session handling is planned for the orchestrator layer.
    """

    model_architecture = "ease"

    def __init__(
        self,
        recsys_config: Optional[EASESettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        self.model: EASEModel = None
        self.dataset: Dataset = None  # might take unnecessary memory
        self.item_id_map: IdMap = None
        self.user_id_map: IdMap = None
        self.warm_users: set = set()

    def train(self, dataset: Dataset):
        assert self.recsys_config is not None

        self.dataset = dataset

        logger.info("Fitting EASE model...")
        self.model = EASEModel(regularization=self.recsys_config.EASE_REGULARIZATION)
        self.model.fit(dataset)

        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        self.warm_users = set(dataset.user_id_map.external_ids)

        logger.info(
            f"EASE model trained. warm_users={len(self.warm_users)}, " f"items={len(self.item_id_map.external_ids)}"
        )

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
    ) -> RecomItems:  # Return type is a RecomItems
        # Warm ranker only. Cold users are routed by the orchestrator/cascade layer.
        if user_ids not in self.warm_users:
            logger.info(f"User {user_ids} is cold for EASE, returning empty (route elsewhere)")
            return RecomItems(item_ids=[], scores=[], strategy=Strategy.MODEL_COLD_USERS.value)

        logger.info(f"Predicting for user {user_ids}")

        recos = self.model.recommend(
            users=[user_ids],
            dataset=self.dataset,
            k=top_n,
            filter_viewed=filter_viewed,
            items_to_recommend=(items_to_recommend if items_to_recommend is None else list(items_to_recommend)),
        )

        recos = recos.sort_values(["user_id", "score"], ascending=False).reset_index(drop=True)

        return RecomItems(
            item_ids=recos.item_id.astype(str).tolist(),
            scores=recos.score.tolist(),
            strategy=Strategy.MODEL_HOT_USERS.value,
        )

    def save_model_triton(self, base_s3_url: Pathy, num_to_keep: int) -> None:
        """
        Save the model to S3 for Triton serving.

        Parameters:
            :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
            :param num_to_keep: number of recent versions to keep.
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

    def calc_metrics(self, k: int, dataset: Dataset, n_splits: int = 3) -> Dict[str, Any]:
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
            "EASE_MODEL": EASEModel(regularization=self.recsys_config.EASE_REGULARIZATION),
        }

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
