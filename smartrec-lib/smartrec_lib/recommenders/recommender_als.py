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
        """
        Compute item similarity matrix and keep only top-k similar items for each item.
        
        Parameters:
            model: Trained ImplicitALSWrapperModel
            k_max_values: Maximum number of similar items to keep for each item
            
        Returns:
            Sparse CSR matrix with item similarities
        """
        if not isinstance(model.model.item_factors, np.ndarray):
            item_factors = model.model.item_factors.to_numpy()
        else:
            item_factors = model.model.item_factors
        
        # Compute similarity matrix
        matrix = item_factors.dot(item_factors.T)
        
        n_items = matrix.shape[0]
        
        # If we have fewer items than k_max_values, keep all similarities
        if n_items <= k_max_values:
            sparse_matrix = sparse.csr_matrix(matrix)
            return sparse_matrix
        
        # Otherwise, keep only top k_max_values for each item
        k_set_to_zero = n_items - k_max_values
        mask = np.apply_along_axis(
            lambda x: x < np.partition(x, k_set_to_zero)[k_set_to_zero], 
            1, 
            matrix
        )
        matrix[mask] = 0

        sparse_matrix = sparse.csr_matrix(matrix)

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

    def train_partial(self, dataset: Dataset, epochs: Optional[int] = None) -> None:
        """
        Perform partial (incremental) training on new data.
        Updates the existing model with new interactions without full retraining.
        
        Note: If new users or items are detected, the method will perform full retraining
        instead of partial fit, as rectools' _fit_partial doesn't support adding new entities.

        Parameters:
            dataset: New dataset with additional interactions to train on.
            epochs: Number of epochs for partial training. If None, uses ALS_ITERATIONS from config.
        """
        assert self.recsys_config is not None
        assert self.model_hot_users is not None, "Model must be trained before train_partial. Call train() first."

        logger.info("Performing partial fit...")

        # Check if there are new users or items
        new_users = set(dataset.user_id_map.external_ids) - set(self.user_id_map.external_ids)
        new_items = set(dataset.item_id_map.external_ids) - set(self.item_id_map.external_ids)
        
        if new_users or new_items:
            # If there are new users or items, we need to do full retraining
            logger.warning(
                f"Detected {len(new_users)} new users and {len(new_items)} new items. "
                "Performing full retraining instead of partial fit."
            )
            # Full retraining
            self.model_hot_users.fit(dataset)
        else:
            # Use config iterations if epochs not specified
            if epochs is None:
                epochs = self.recsys_config.ALS_ITERATIONS

            # Partial fit for hot users model - only updates existing user/item factors
            logger.info(f"No new users/items detected. Performing partial fit with {epochs} epochs.")
            self.model_hot_users._fit_partial(dataset, epochs=epochs)
        
        # Update dataset and id maps
        self.dataset = dataset
        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map

        # Recompute item similarity matrix with updated factors
        self.item_similarity = self.compute_item_similarity(self.model_hot_users)

        # Refit cold users (popularity) model with full dataset
        self.model_cold_users.fit(dataset)

        # Update hot user and item sets
        self.user_ids_hot = set(self.dataset.user_id_map.convert_to_external(self.dataset.interactions.df.user_id))
        self.item_ids_hot = set(self.dataset.item_id_map.convert_to_external(self.dataset.interactions.df.item_id))

        logger.info("Partial fit completed successfully.")

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
        
        # Filter items_to_recommend to only those that exist in our training data
        # If items_to_recommend is None, we don't filter and show all items
        items_to_recommend_filtered = None
        if items_to_recommend is not None:
            items_to_recommend_filtered = [
                item_to_recommend 
                for item_to_recommend in items_to_recommend 
                if item_to_recommend in self.item_ids_hot
            ]

        # user can be in the short memory, long memory or nowhere
        if user_ids in self.user_id_map.external_ids and user_ids in self.user_ids_hot:
            logger.info("Hot user")
            recos_df = self.model_hot_users.recommend(
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
            recos_df = recos_df.sort_values(["user_id", "score"], ascending=False).reset_index(drop=True)
            
            return RecomItems(
                item_ids=recos_df.item_id.astype(str).tolist(),
                scores=recos_df.score.tolist(),
                strategy=Strategy.MODEL_HOT_USERS,
            )
            
        elif history and len(history):
            logger.info("Warm user")
            return self.recommend_unknown_user_with_history(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )
            
        else:
            logger.info("Cold user")
            recos_df = self.model_cold_users.recommend(
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
            recos_df = recos_df.sort_values(["user_id", "score"], ascending=False).reset_index(drop=True)
            
            return RecomItems(
                item_ids=recos_df.item_id.astype(str).tolist(),
                scores=recos_df.score.tolist(),
                strategy=Strategy.MODEL_COLD_USERS,
            )

    def recommend_unknown_user_with_history(
        self,
        user_ids: int,
        history: List[str],
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
    ) -> RecomItems:
        """
        Recommend items for warm users (users with history but not in training data).
        Uses item-to-item similarity based on user's history.
        
        Parameters:
            user_ids: User ID to generate recommendations for
            history: List of item IDs that user has interacted with
            top_n: Number of recommendations to return
            filter_viewed: Whether to filter out items from history
            items_to_recommend: Optional list of item IDs to restrict recommendations to
            
        Returns:
            RecomItems object with item_ids, scores, and strategy
        """
        # Process items_to_recommend - create a mask for filtering
        items_enc_v = None
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
                # No valid items to recommend
                return RecomItems(
                    item_ids=[],
                    scores=[],
                    strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY,
                )

        # Process history - convert to internal IDs and create dense vector
        history_enc_list = [
            self.item_id_map.convert_to_internal([item]) 
            for item in history 
            if item in self.item_id_map.external_ids
        ]
        
        if len(history_enc_list) == 0:
            # No valid history items
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.MODEL_WARM_USERS,
            )

        history_enc = np.array(history_enc_list)
        history_dense_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
        history_dense_v[:, history_enc] = 1
        
        # Compute similarity scores between history items and all items
        all_scores = self.item_similarity.T.dot(history_dense_v.T).T[0]
        sorted_item_ids = np.argsort(all_scores)

        # Create mask for filtering
        items_mask = np.ones(len(sorted_item_ids), dtype=bool)

        # Filter out already viewed items if requested
        if filter_viewed:
            filter_already_liked_mask = ~(history_dense_v[:, sorted_item_ids][0].astype(bool))
            items_mask = items_mask & filter_already_liked_mask

        # Filter to only requested items if provided
        if items_to_recommend is not None and items_enc_v is not None:
            filter_item_mask = items_enc_v[:, sorted_item_ids][0].astype(bool)
            items_mask = items_mask & filter_item_mask

        sorted_item_ids = sorted_item_ids[items_mask]

        # Get top-N items (reverse order to get highest scores)
        nearest_item_ids_enc = sorted_item_ids[: -top_n - 1 : -1]
        nearest_scores = all_scores[nearest_item_ids_enc]

        # Convert to external IDs and return
        item_ids_external = self.item_id_map.convert_to_external(nearest_item_ids_enc)
        
        return RecomItems(
            item_ids=[str(item_id) for item_id in item_ids_external],
            scores=nearest_scores.tolist(),
            strategy=Strategy.MODEL_WARM_USERS,
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
