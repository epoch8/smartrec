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
from rectools.models import ImplicitALSWrapperModel, PopularModel
from scipy import sparse

from smartrec_lib.model import ALSSettings, CoVisSettings, RecomItems, Strategy
from smartrec_lib.policy.fusion import rrf_fuse
from smartrec_lib.recommenders import RecommenderModel
from smartrec_lib.recommenders.recommender_covis import RecommenderCoVis
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
        self.user_item_matrix_binary = None
        self.implicit_neginf_score = None
        # Optional co-visitation session layer (SESSION_COVIS_ENABLED)
        self.covis: Optional[RecommenderCoVis] = None

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
        # Artifacts pickled before the covis session layer existed lack the attr.
        if not hasattr(self, "covis"):
            self.covis = None
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

    def _recommend_hot_user_candidates(
        self,
        user_ids: Any,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # Optional co-visitation session layer: replaces the two weak item-sim
        # session paths (see _recommend_hot_user_with_covis_blend and the covis
        # branch in recommend_unknown_user_with_history). Off by default.
        self.covis = None
        if getattr(self.recsys_config, "SESSION_COVIS_ENABLED", False):
            covis_config = CoVisSettings(
                RECOMMENDER_DAYS_THRESHOLD=self.recsys_config.RECOMMENDER_DAYS_THRESHOLD,
                RECOMMENDER_RANDOM_STATE=self.recsys_config.RECOMMENDER_RANDOM_STATE,
                COVIS_TOP_K=self.recsys_config.COVIS_TOP_K,
                COVIS_MIN_COOC=self.recsys_config.COVIS_MIN_COOC,
                COVIS_SESSION_WEIGHTS=getattr(self.recsys_config, "COVIS_SESSION_WEIGHTS", False),
            )
            self.covis = RecommenderCoVis(
                recsys_config=covis_config, model_name=self.model_name, model_version=self.model_version
            )
            self.covis.train(dataset)
            logger.info("Session covis layer trained (SESSION_COVIS_ENABLED=True).")

        logger.info("Base models trained.")

    def train_partial(self, dataset: Dataset, epochs: Optional[int] = None) -> None:
        """Incremental (partial) training for ALS.

        UNDER DEVELOPMENT - not available yet. Use train() for full retraining.
        """
        raise NotImplementedError(
            "train_partial for ALS is under development and not available yet. " "Use train() for full retraining."
        )

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
    ) -> RecomItems:  # Return type is a RecomItems
        total_start = perf_counter()
        self._ensure_lookup_caches()

        if isinstance(items_to_recommend, list) and len(items_to_recommend) == 0:
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY,
            )

        logger.info(
            f"Predicting for user {user_ids}, history={'present' if (history is not None and len(history) > 0) else 'absent'}"
        )

        # Filter items_to_recommend to only those that exist in our training data
        # If items_to_recommend is None, we don't filter and show all items
        filter_start = perf_counter()
        items_to_recommend_filtered = None
        if items_to_recommend is not None:
            items_to_recommend_filtered = [
                item_to_recommend for item_to_recommend in items_to_recommend if item_to_recommend in self.item_ids_hot
            ]
        filter_ms = (perf_counter() - filter_start) * 1000

        # Determine user type and check for real-time history
        is_hot_user = user_ids in self.user_id_map.external_ids and user_ids in self.user_ids_hot
        has_session_history = history is not None and len(history) > 0

        # HOT USER (in training data)
        if is_hot_user:
            if has_session_history:
                # Hot user + real-time events = enrich ALS with recent behavior
                logger.info(f"Hot user with real-time enrichment: user={user_ids}, history_size={len(history)}")
                return self._recommend_hot_user_with_session(
                    user_ids=user_ids,
                    top_n=top_n,
                    filter_viewed=filter_viewed,
                    items_to_recommend=items_to_recommend_filtered,
                    history=history,
                )
            else:
                # Standard hot user with ALS
                als_start = perf_counter()
                _, item_ids_external, scores = self._recommend_hot_user_candidates(
                    user_ids=user_ids,
                    top_n=top_n,
                    filter_viewed=filter_viewed,
                    items_to_recommend=items_to_recommend_filtered,
                )
                als_ms = (perf_counter() - als_start) * 1000

                total_ms = (perf_counter() - total_start) * 1000
                logger.info(
                    f"[ALS HOT] user={user_ids}, top_n={top_n} | "
                    f"total={total_ms:.1f}ms | filter={filter_ms:.1f}ms, als={als_ms:.1f}ms | "
                    f"results={len(item_ids_external)}"
                )

                return RecomItems(
                    item_ids=[str(item_id) for item_id in item_ids_external],
                    scores=scores.astype(float).tolist(),
                    strategy=Strategy.MODEL_HOT_USERS.value,
                )

        # WARM USER (new user with real-time events)
        elif has_session_history:
            logger.info(f"Warm user with real-time history: user={user_ids}, history_size={len(history)}")
            return self.recommend_unknown_user_with_history(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )

        # COLD USER (new user without events)
        else:
            cold_start = perf_counter()
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
            cold_ms = (perf_counter() - cold_start) * 1000

            sort_start = perf_counter()
            recos_df = recos_df.sort_values(["user_id", "score"], ascending=False).reset_index(drop=True)
            sort_ms = (perf_counter() - sort_start) * 1000

            total_ms = (perf_counter() - total_start) * 1000
            logger.info(
                f"[ALS COLD] user={user_ids}, top_n={top_n} | "
                f"total={total_ms:.1f}ms | filter={filter_ms:.1f}ms, popular={cold_ms:.1f}ms, sort={sort_ms:.1f}ms | "
                f"results={len(recos_df)}"
            )

            return RecomItems(
                item_ids=recos_df.item_id.astype(str).tolist(),
                scores=recos_df.score.tolist(),
                strategy=Strategy.MODEL_COLD_USERS.value,
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

    def _session_covis_active(self) -> bool:
        """True when the covis session layer was trained and is usable."""
        self._ensure_lookup_caches()
        covis = getattr(self, "covis", None)
        return covis is not None and bool(getattr(covis, "neighbors", None))

    def _viewed_external(self, user_ids: Any) -> set:
        """External ids of everything the user interacted with in training."""
        self._ensure_user_item_matrix_binary()
        user_internal = int(self.user_id_map.convert_to_internal([user_ids])[0])
        viewed_internal = self.user_item_matrix_binary[user_internal].indices
        return {str(item) for item in self.item_id_map.convert_to_external(viewed_internal)}

    def _recommend_hot_user_with_covis_blend(
        self,
        user_ids: Any,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[Any]],
        history: List[str],
    ) -> RecomItems:
        """Hot user + session: weighted RRF blend of pure ALS and co-visitation.

        Blend weights come from the offline policy grid (EXPERIMENTS.md
        2026-08-03): als=1.0 + covis=1.0 beat pure ALS by ~6% map@10 on warm
        users. Both sources are overfetched 2x so the rank fusion has real
        overlap to work with before the final cut to top_n.
        """
        config = self.recsys_config
        fetch_n = top_n * 2
        covis_result = self.covis.recommend(
            user_ids,
            top_n=fetch_n,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
            history=history,
        )
        _, als_items_external, als_scores = self._recommend_hot_user_candidates(
            user_ids=user_ids,
            top_n=fetch_n,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        als_list = [str(item) for item in als_items_external]

        if not covis_result.item_ids:
            logger.info(f"[ALS_COVIS BLEND] covis empty, serving pure ALS: user={user_ids}")
            return RecomItems(
                item_ids=als_list[:top_n],
                scores=als_scores.astype(float).tolist()[:top_n],
                strategy=Strategy.MODEL_HOT_USERS.value,
            )

        fused = rrf_fuse(
            {"als": als_list, "covis": list(covis_result.item_ids)},
            {
                "als": getattr(config, "BLEND_ALS_WEIGHT", 1.0),
                "covis": getattr(config, "BLEND_COVIS_WEIGHT", 1.0),
            },
            rrf_k=getattr(config, "BLEND_RRF_K", 60),
        )
        # CoVis only filters its own session seed; drop training-viewed items so
        # filter_viewed semantics stay identical to the pure-ALS path.
        exclude = self._viewed_external(user_ids) if filter_viewed else set()
        items = [item for item, _ in fused if item not in exclude][:top_n]
        scores = [1.0 / rank for rank in range(1, len(items) + 1)]
        logger.info(
            f"[ALS_COVIS BLEND] user={user_ids} | als={len(als_list)} covis={len(covis_result.item_ids)} "
            f"-> fused={len(items)}"
        )
        return RecomItems(item_ids=items, scores=scores, strategy=Strategy.MODEL_ALS_COVIS_BLEND.value)

    def _recommend_hot_user_with_session(
        self,
        user_ids: int,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[int]],
        history: List[str],
    ) -> RecomItems:
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
        # Covis session layer (SESSION_COVIS_ENABLED): replace the 70/30
        # ALS + item-sim mix with the RRF blend of ALS and co-visitation -
        # the item-sim half measured below plain popularity offline
        # (EXPERIMENTS.md 2026-08-02/03).
        if self._session_covis_active():
            return self._recommend_hot_user_with_covis_blend(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )

        total_start = perf_counter()
        timing = {}

        # Get ALS recommendations
        als_start = perf_counter()
        als_item_ids_internal, als_item_ids_external, als_scores = self._recommend_hot_user_candidates(
            user_ids=user_ids,
            top_n=top_n * 2,  # Get more for blending
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        timing["als_recommend_ms"] = (perf_counter() - als_start) * 1000

        if len(als_item_ids_internal) == 0:
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.MODEL_HOT_USERS.value,
            )

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
            return RecomItems(
                item_ids=[str(item_id) for item_id in als_item_ids_external[:top_n]],
                scores=als_scores[:top_n].astype(float).tolist(),
                strategy=Strategy.MODEL_HOT_USERS.value,  # Fallback to standard
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

        return RecomItems(
            item_ids=[str(als_item_ids_external[i]) for i in top_indices],
            scores=[float(blended[i]) for i in top_indices],
            strategy=Strategy.MODEL_REALTIME_HOT_USERS.value,
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
        # Covis session layer (SESSION_COVIS_ENABLED): co-visitation instead of
        # the ALS item-sim path, which measured below plain popularity offline
        # while covis was 4x better (EXPERIMENTS.md 2026-08-02).
        if self._session_covis_active():
            covis_result = self.covis.recommend(
                user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )
            if covis_result.item_ids:
                logger.info(f"[ALS SESSION->COVIS] user={user_ids} -> covis ({len(covis_result.item_ids)} items)")
                return RecomItems(
                    item_ids=list(covis_result.item_ids),
                    scores=list(covis_result.scores),
                    strategy=Strategy.MODEL_COVIS_SESSION.value,
                )
            logger.info(f"[ALS SESSION->COVIS] covis empty, falling back to item-sim: user={user_ids}")

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
                # No valid items to recommend
                return RecomItems(
                    item_ids=[],
                    scores=[],
                    strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY,
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
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.MODEL_REALTIME_WARM_USERS.value,
            )

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

        return RecomItems(
            item_ids=[str(item_id) for item_id in item_ids_external],
            scores=nearest_scores.tolist(),
            strategy=Strategy.MODEL_REALTIME_WARM_USERS.value,
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
