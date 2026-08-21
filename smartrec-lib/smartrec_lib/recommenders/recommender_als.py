import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
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
from rectools.models import ImplicitALSWrapperModel
from scipy import sparse

from smartrec_lib.model import ALSSettings, RecomItems, Strategy
from smartrec_lib.recommenders.base import RecommenderModel
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("ALS Model")
logger.setLevel(logging.INFO)


class RecommenderALS(RecommenderModel):
    """Matrix-factorisation ranker. Serves the users it has embeddings for.

    A MEMBER of a model set, and deliberately ignorant of everything above it:
    it does not know that cold users get popularity, that a session may be
    scored by co-visitation, or that its answers are labelled by visitor
    segment. It answers two questions - "can I score this user" (`can_serve`)
    and "here are my candidates" - and `RecommenderModelSet` decides the rest.

    Until 2026-08-22 this class WAS the whole feed: it owned a PopularModel for
    cold users, an optional RecommenderCoVis for sessions, the fusion weights,
    the routing across all four visitor segments and the Strategy vocabulary.
    That is why `als_covis_youtravel` was "the ALS artifact with extras" instead
    of what it actually is - a set of models behind one name.
    """

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

        self.model_hot_users: ImplicitALSWrapperModel = None
        self.dataset: Dataset = None  # might take unnecessary memory
        self.item_id_map: IdMap = None
        self.user_id_map: IdMap = None
        self.user_item_matrix_binary = None
        self.implicit_neginf_score = None

    def can_serve(self, user_ids: Any) -> bool:
        """True when this user has an embedding, i.e. was in the training data."""
        self._ensure_lookup_caches()
        return user_ids in self.user_id_map.external_ids and user_ids in self.user_ids_hot

    def warm_caches(self) -> None:
        """Build the lazy lookup caches ahead of the first request."""
        self._ensure_lookup_caches()
        self._ensure_user_item_matrix_binary()

    def known_items(self, items_to_recommend: Optional[List[Any]]) -> Optional[List[Any]]:
        """Restrict a candidate list to items this model can actually score."""
        if items_to_recommend is None:
            return None
        return [item for item in items_to_recommend if item in self.item_ids_hot]

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
        mask = np.apply_along_axis(lambda x: x < np.partition(x, k_set_to_zero)[k_set_to_zero], 1, matrix)
        matrix[mask] = 0

        sparse_matrix = sparse.csr_matrix(matrix)

        return sparse_matrix

    def _ensure_lookup_caches(self) -> None:
        if not hasattr(self, "user_item_matrix_binary"):
            self.user_item_matrix_binary = None
        if not hasattr(self, "implicit_neginf_score") or self.implicit_neginf_score is None:
            self.implicit_neginf_score = float(
                np.asarray(
                    np.asarray(-np.finfo(np.float32).max, dtype=np.float32).view(np.uint32) - 1,
                    dtype=np.uint32,
                ).view(np.float32)
            )

    def _ensure_user_item_matrix_binary(self) -> None:
        self._ensure_lookup_caches()
        if self.user_item_matrix_binary is None:
            build_start = perf_counter()
            self.user_item_matrix_binary = self.dataset.get_user_item_matrix(
                include_weights=False,
                include_warm_users=True,
                include_warm_items=True,
                dtype=np.float32,
            )
            build_ms = (perf_counter() - build_start) * 1000
            logger.info(f"[ALS CACHE] user_item_matrix_binary_built={build_ms:.1f}ms")

    def hot_user_candidates(
        self,
        user_ids: Any,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw ALS candidates: (internal ids, external ids, scores).

        Public because the model set fuses these with another member's ranking.
        """
        self._ensure_user_item_matrix_binary()

        user_id_internal = int(self.user_id_map.convert_to_internal([user_ids])[0])
        user_items_row = self.user_item_matrix_binary[user_id_internal]

        items_internal = None
        if items_to_recommend is not None:
            if len(items_to_recommend) == 0:
                return np.array([], dtype=int), np.array([], dtype=object), np.array([], dtype=np.float32)
            items_internal = np.asarray(self.item_id_map.convert_to_internal(items_to_recommend), dtype=np.int32)

        item_ids_internal, scores = self.model_hot_users.model.recommend(
            userid=user_id_internal,
            user_items=user_items_row,
            N=top_n,
            filter_already_liked_items=filter_viewed,
            items=items_internal,
        )

        item_ids_internal = np.atleast_1d(np.asarray(item_ids_internal))
        scores = np.atleast_1d(np.asarray(scores, dtype=np.float32))

        valid_mask = np.isfinite(scores) & (scores > self.implicit_neginf_score)
        item_ids_internal = item_ids_internal[valid_mask]
        scores = scores[valid_mask]

        if len(item_ids_internal) == 0:
            return np.array([], dtype=int), np.array([], dtype=object), np.array([], dtype=np.float32)

        item_ids_external = np.asarray(self.item_id_map.convert_to_external(item_ids_internal))
        return item_ids_internal, item_ids_external, scores

    def train(self, dataset: Dataset):
        assert self.recsys_config is not None

        self.dataset = dataset
        als = self.recsys_config

        logger.info("Fitting model...")

        np.random.seed(als.RECOMMENDER_RANDOM_STATE)

        self.model_hot_users = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=als.ALS_FACTORS,
                regularization=als.ALS_REGULARIZATION_FACTOR,
                iterations=als.ALS_ITERATIONS,
                alpha=als.ALS_ALPHA,
                random_state=als.RECOMMENDER_RANDOM_STATE,
            ),
            fit_features_together=False,  # way to fit paired features
        )
        self.model_hot_users.fit(dataset)
        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        self.item_similarity = self.compute_item_similarity(self.model_hot_users)

        self.user_ids_hot = set(self.dataset.user_id_map.convert_to_external(self.dataset.interactions.df.user_id))
        # Some item_ids may not be in interactions
        self.item_ids_hot = set(self.dataset.item_id_map.convert_to_external(self.dataset.interactions.df.item_id))

        logger.info("ALS trained.")

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
    ) -> RecomItems:
        """Rank for a user this model has an embedding for.

        `history` is accepted for interface compatibility and IGNORED - session
        scoring is `recommend_hot_user_with_session` / `recommend_from_session`,
        and choosing between them is the model set's job, not this method's.
        Callers that are not the model set should ask `can_serve` first; an
        unknown user gets an empty result rather than a fallback, because
        picking a fallback is not this model's decision.
        """
        total_start = perf_counter()
        self._ensure_lookup_caches()

        if isinstance(items_to_recommend, list) and len(items_to_recommend) == 0:
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY,
            )

        if not self.can_serve(user_ids):
            logger.info(f"[ALS] no embedding for user={user_ids}, returning empty")
            return RecomItems(item_ids=[], scores=[], strategy=Strategy.MODEL_HOT_USERS.value)

        als_start = perf_counter()
        _, item_ids_external, scores = self.hot_user_candidates(
            user_ids=user_ids,
            top_n=top_n,
            filter_viewed=filter_viewed,
            items_to_recommend=self.known_items(items_to_recommend),
        )
        als_ms = (perf_counter() - als_start) * 1000

        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            f"[ALS HOT] user={user_ids}, top_n={top_n} | "
            f"total={total_ms:.1f}ms | als={als_ms:.1f}ms | results={len(item_ids_external)}"
        )

        return RecomItems(
            item_ids=[str(item_id) for item_id in item_ids_external],
            scores=scores.astype(float).tolist(),
            strategy=Strategy.MODEL_HOT_USERS.value,
        )

    @staticmethod
    def _parse_weighted_history(history: List[str]) -> tuple[List[str], List[float]]:
        """
        Parse weighted history in format "tour_id:weight" or plain "tour_id".

        Args:
            history: List of items in format ["123:1.0", "456:2.0"], ["123", "456"], or [123, 456]

        Returns:
            Tuple of (item_ids, weights)
        """
        item_ids = []
        weights = []

        for item in history:
            # Convert to string if it's an int (backward compatibility)
            item_str = str(item)

            if ":" in item_str:
                # Weighted format: "tour_id:weight"
                parts = item_str.split(":", 1)
                item_ids.append(parts[0])
                try:
                    weights.append(float(parts[1]))
                except ValueError:
                    # Invalid weight, default to 1.0
                    weights.append(1.0)
            else:
                # Plain format: "tour_id" or int
                item_ids.append(item_str)
                weights.append(1.0)

        return item_ids, weights

    def viewed_external(self, user_ids: Any) -> set:
        """External ids of everything the user interacted with in training.

        Public because a fused ranking has to honour `filter_viewed` too, and
        only this model knows what the user was seen with during training.
        """
        self._ensure_user_item_matrix_binary()
        user_internal = int(self.user_id_map.convert_to_internal([user_ids])[0])
        viewed_internal = self.user_item_matrix_binary[user_internal].indices
        return {str(item) for item in self.item_id_map.convert_to_external(viewed_internal)}

    def recommend_hot_user_with_session(
        self,
        user_ids: int,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[int]],
        history: List[str],
    ) -> tuple[RecomItems, bool]:
        """
        Recommend for hot users enriched with real-time session history.

        Combines ALS embeddings (from training data) with recent session behavior.
        Strategy: blend ALS scores with item similarity to recent clicks.

        Parameters:
            user_ids: Hot user ID (exists in training data)
            top_n: Number of recommendations
            filter_viewed: Filter out viewed items
            items_to_recommend: Optional candidate items
            history: Real-time session history from Redis

        Returns:
            RecomItems with MODEL_REALTIME_HOT_USERS strategy
        """
        total_start = perf_counter()
        timing = {}

        # Get ALS recommendations
        als_start = perf_counter()
        als_item_ids_internal, als_item_ids_external, als_scores = self.hot_user_candidates(
            user_ids=user_ids,
            top_n=top_n * 2,  # Get more for blending
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        timing["als_recommend_ms"] = (perf_counter() - als_start) * 1000

        if len(als_item_ids_internal) == 0:
            return RecomItems(item_ids=[], scores=[], strategy=None), False

        # Parse weighted history
        parse_start = perf_counter()
        history_items, history_weights = self._parse_weighted_history(history)
        timing["parse_history_ms"] = (perf_counter() - parse_start) * 1000

        # Get item similarity based on recent history with weights
        convert_start = perf_counter()

        # Create a set for O(1) lookups instead of O(n) numpy array checks
        external_ids_set = set(self.item_id_map.external_ids)

        # ОПТИМИЗАЦИЯ: собираем валидные items и конвертируем батчем
        valid_items_for_convert = []
        valid_weights_for_convert = []

        for item, weight in zip(history_items, history_weights):
            item_for_check = item
            if item_for_check not in external_ids_set:
                try:
                    item_for_check = int(item)
                except (ValueError, TypeError):
                    pass
            if item_for_check in external_ids_set:
                valid_items_for_convert.append(item_for_check)
                valid_weights_for_convert.append(weight)

        if len(valid_items_for_convert) == 0:
            # No valid history - fall back to standard ALS
            logger.warning(f"Hot user {user_ids} has invalid history, using standard ALS")
            # No usable seed: this is a plain ALS answer, and saying otherwise
            # would mislabel the segment. The caller names it accordingly.
            return (
                RecomItems(
                    item_ids=[str(item_id) for item_id in als_item_ids_external[:top_n]],
                    scores=als_scores[:top_n].astype(float).tolist(),
                    strategy=None,
                ),
                False,
            )

        # Батчевая конвертация
        history_enc_list = self.item_id_map.convert_to_internal(valid_items_for_convert)
        valid_weights = valid_weights_for_convert
        timing["convert_ids_ms"] = (perf_counter() - convert_start) * 1000

        # Compute weighted item similarity scores for enrichment
        similarity_start = perf_counter()
        history_enc = np.array(history_enc_list)
        weights_array = np.array(valid_weights)

        history_dense_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
        history_dense_v[0, history_enc] = weights_array

        similarity_scores = self.item_similarity.T.dot(history_dense_v.T).T[0]
        timing["similarity_ms"] = (perf_counter() - similarity_start) * 1000

        logger.debug(
            f"Hot user real-time: history_items={len(history_items)}, "
            f"valid_items={len(history_enc_list)}, weights={valid_weights[:5]}..."
        )

        # Blend ALS scores with similarity scores
        blend_start = perf_counter()

        # Vectorized blending
        sim_scores_for_als = similarity_scores[als_item_ids_internal]

        # Blend: 70% ALS + 30% real-time similarity
        blended = 0.7 * als_scores + 0.3 * sim_scores_for_als

        # Get top-N indices
        top_indices = np.argsort(blended)[::-1][:top_n]
        timing["blend_ms"] = (perf_counter() - blend_start) * 1000

        total_ms = (perf_counter() - total_start) * 1000

        logger.info(
            f"[ALS HOT+RT] user={user_ids}, history={len(history)}, valid={len(history_enc_list)} | "
            f"total={total_ms:.1f}ms | "
            f"als={timing['als_recommend_ms']:.1f}ms, "
            f"parse={timing['parse_history_ms']:.1f}ms, "
            f"convert={timing['convert_ids_ms']:.1f}ms, "
            f"similarity={timing['similarity_ms']:.1f}ms, "
            f"blend={timing['blend_ms']:.1f}ms | "
            f"als_candidates={len(als_item_ids_internal)}, final={len(top_indices)}"
        )

        return (
            RecomItems(
                item_ids=[str(als_item_ids_external[i]) for i in top_indices],
                scores=[float(blended[i]) for i in top_indices],
                strategy=None,
            ),
            True,
        )

    def recommend_from_session(
        self,
        user_ids: int,
        history: List[str],
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
    ) -> tuple[RecomItems, bool]:
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
        total_start = perf_counter()
        timing = {}

        # Process items_to_recommend - create a mask for filtering
        items_start = perf_counter()
        items_enc_v = None
        if items_to_recommend is not None:
            # ОПТИМИЗАЦИЯ: батчим конвертацию ID вместо цикла
            valid_items = [item for item in items_to_recommend if item in self.item_id_map.external_ids]
            if len(valid_items) > 0:
                items_to_recommend_enc = self.item_id_map.convert_to_internal(valid_items)
                items_enc = np.array(items_to_recommend_enc)
                items_enc_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
                items_enc_v[:, items_enc] = 1
            else:
                # Nothing in the candidate list is scorable by this model.
                # Distinct from "no answer": the caller keeps this label.
                return (
                    RecomItems(
                        item_ids=[],
                        scores=[],
                        strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY.value,
                    ),
                    False,
                )
        timing["items_filter_ms"] = (perf_counter() - items_start) * 1000

        # Parse weighted history
        parse_start = perf_counter()
        history_items, history_weights = self._parse_weighted_history(history)
        timing["parse_history_ms"] = (perf_counter() - parse_start) * 1000

        # Process history - convert to internal IDs with weights
        convert_start = perf_counter()

        # Create a set for O(1) lookups instead of O(n) numpy array checks
        external_ids_set = set(self.item_id_map.external_ids)

        # ОПТИМИЗАЦИЯ: собираем валидные items и конвертируем батчем
        valid_items_for_convert = []
        valid_weights_for_convert = []

        for item, weight in zip(history_items, history_weights):
            item_for_check = item
            if item_for_check not in external_ids_set:
                try:
                    item_for_check = int(item)
                except (ValueError, TypeError):
                    pass
            if item_for_check in external_ids_set:
                valid_items_for_convert.append(item_for_check)
                valid_weights_for_convert.append(weight)

        if len(valid_items_for_convert) == 0:
            logger.warning(
                f"No valid history items found for user {user_ids}. "
                f"History items: {history_items[:5]}... "
                f"Known items: {len(self.item_id_map.external_ids)}"
            )
            return RecomItems(item_ids=[], scores=[], strategy=None), False

        # Батчевая конвертация вместо цикла
        history_enc_list = self.item_id_map.convert_to_internal(valid_items_for_convert)
        valid_weights = valid_weights_for_convert
        timing["convert_ids_ms"] = (perf_counter() - convert_start) * 1000

        # Build history vector
        vector_start = perf_counter()
        history_enc = np.array(history_enc_list)
        weights_array = np.array(valid_weights)

        history_dense_v = np.zeros((1, self.model_hot_users.model.item_factors.shape[0]))
        history_dense_v[0, history_enc] = weights_array
        timing["build_vector_ms"] = (perf_counter() - vector_start) * 1000

        logger.debug(
            f"Warm user history: items={len(history_items)}, valid={len(history_enc_list)}, "
            f"weights={valid_weights[:5]}..."
        )

        # Compute weighted similarity scores between history items and all items
        similarity_start = perf_counter()
        all_scores = self.item_similarity.T.dot(history_dense_v.T).T[0]
        timing["similarity_dot_ms"] = (perf_counter() - similarity_start) * 1000

        # ОПТИМИЗАЦИЯ: используем argpartition вместо полной сортировки
        # argpartition - O(n), argsort - O(n log n)
        sort_start = perf_counter()

        # Нам нужны top-N максимальных значений
        # argpartition даёт нам индексы так, что все значения >= k-го будут справа
        n_items = len(all_scores)
        k = min(top_n * 2, n_items - 1)  # берём с запасом для фильтрации

        if k > 0 and k < n_items:
            # Используем -k чтобы получить top-k максимальных
            top_indices = np.argpartition(all_scores, -k)[-k:]
            # Сортируем только top-k (маленький массив)
            top_indices_sorted = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        else:
            # Если items мало, используем полную сортировку
            top_indices_sorted = np.argsort(all_scores)[::-1]
        timing["sort_ms"] = (perf_counter() - sort_start) * 1000

        # Filter
        filter_start = perf_counter()
        filtered_indices = []
        for idx in top_indices_sorted:
            # Filter viewed
            if filter_viewed and history_dense_v[0, idx] > 0:
                continue
            # Filter to items_to_recommend
            if items_enc_v is not None and items_enc_v[0, idx] == 0:
                continue
            filtered_indices.append(idx)
            if len(filtered_indices) >= top_n:
                break

        nearest_item_ids_enc = np.array(filtered_indices) if filtered_indices else np.array([], dtype=int)
        timing["filter_ms"] = (perf_counter() - filter_start) * 1000

        # Get scores and convert to external IDs
        output_start = perf_counter()
        if len(nearest_item_ids_enc) > 0:
            nearest_scores = all_scores[nearest_item_ids_enc]
            item_ids_external = self.item_id_map.convert_to_external(nearest_item_ids_enc)
        else:
            nearest_scores = np.array([])
            item_ids_external = []
        timing["output_ms"] = (perf_counter() - output_start) * 1000

        total_ms = (perf_counter() - total_start) * 1000

        # Логирование профиля
        logger.info(
            f"[ALS WARM] user={user_ids}, history={len(history)}, valid={len(history_enc_list)}, top_n={top_n} | "
            f"total={total_ms:.1f}ms | "
            f"items_filter={timing['items_filter_ms']:.1f}ms, "
            f"parse={timing['parse_history_ms']:.1f}ms, "
            f"convert={timing['convert_ids_ms']:.1f}ms, "
            f"vector={timing['build_vector_ms']:.1f}ms, "
            f"similarity={timing['similarity_dot_ms']:.1f}ms, "
            f"sort={timing['sort_ms']:.1f}ms, "
            f"filter={timing['filter_ms']:.1f}ms, "
            f"output={timing['output_ms']:.1f}ms | "
            f"results={len(nearest_item_ids_enc)}"
        )

        return (
            RecomItems(
                item_ids=[str(item_id) for item_id in item_ids_external],
                scores=nearest_scores.tolist(),
                strategy=None,
            ),
            True,
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
        user_item_matrix_binary_cache = self.user_item_matrix_binary
        self.user_item_matrix_binary = None
        upload_model_files(
            base_s3_url,
            model_name=self.model_name,
            model_version=self.model_version,
            model_data=self.__dict__,
        )
        self.user_item_matrix_binary = user_item_matrix_binary_cache
        logger.info("Model saved successfully!")
        clean_old_model_versions(base_s3_url=base_s3_url, model_name=self.model_name, num_to_keep=num_to_keep)
        logger.info("Old models deleted!")

        return None

    def calc_metrics(self, k: int, dataset: Dataset, n_splits: int = 3) -> Dict[str, Any]:
        """Cross-validate THIS model. The set aggregates its members' metrics."""
        assert self.recsys_config is not None
        als = self.recsys_config

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
                    factors=als.ALS_FACTORS,
                    regularization=als.ALS_REGULARIZATION_FACTOR,
                    iterations=als.ALS_ITERATIONS,
                    alpha=als.ALS_ALPHA,
                    random_state=als.RECOMMENDER_RANDOM_STATE,
                ),
                fit_features_together=False,  # way to fit paired features
            ),
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
