"""One artifact, several models. The only place routing and fusion live.

An artifact served by Triton is not one model. It is a main ranker for the users
it has embeddings for, a fallback for everyone else, optionally a session layer,
and the weights that fuse them. Until 2026-08-22 all of that lived inside
`RecommenderALS`, which is why the two shipped artifacts read as "ALS with
extras" - the ALS class owned a PopularModel, an optional RecommenderCoVis, the
blend weights, the four-way segment routing and the whole Strategy vocabulary.

The split is:

    RecommenderModelSet  - which member answers, and under what Strategy
    members              - how to score, and nothing else

A member knows only its own algorithm. It never learns that a sibling exists,
that cold users go to popularity, or how the answer will be labelled. Adding a
member (EASE as a warm ranker, a content model for cold tours) is a change to
this file and to `ModelSetSettings` - not to any existing model.
"""

import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

from pathy import Pathy
from rectools.dataset import Dataset

from smartrec_lib.kernels.fusion import rrf_fuse
from smartrec_lib.model import (
    ALSSettings,
    CoVisSettings,
    EASESettings,
    ModelSetSettings,
    PopularSettings,
    RecomItems,
    Strategy,
)
from smartrec_lib.recommenders.base import RecommenderModel
from smartrec_lib.recommenders.recommender_als import RecommenderALS
from smartrec_lib.recommenders.recommender_covis import RecommenderCoVis
from smartrec_lib.recommenders.recommender_ease import RecommenderEASE
from smartrec_lib.recommenders.recommender_popular import RecommenderPopular
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("Model Set")
logger.setLevel(logging.INFO)

# Which model implements which settings. This is the ONLY place in the library
# that maps the two, which is what lets a role hold any model: the config says
# `main=LightFMSettings(...)` and the set builds a RecommenderLightFM without a
# single conditional anywhere else. Adding a model = one entry here plus the type
# in `RankerSettings`.
MODEL_FOR_SETTINGS = {
    ALSSettings: RecommenderALS,
    CoVisSettings: RecommenderCoVis,
    PopularSettings: RecommenderPopular,
    EASESettings: RecommenderEASE,
}


class RecommenderModelSet(RecommenderModel):
    """Routes a request to one of its members and labels the answer.

    Members, by role:

    - `main` - ranks users it has a representation for. Today RecommenderALS.
    - `fallback` - ranks everyone else. Today RecommenderPopular. Always present.
    - `session` - ranks from the live session. Optional; `None` means the artifact
      has no session layer and `main` scores sessions with its own machinery,
      which is what `als_youtravel` does.

    The routing table is `recommend()` and it is deliberately the whole of the
    decision-making in this package. `Strategy` is set here and nowhere else:
    a strategy names the visitor segment and the signal used, so only the thing
    that knows which segment it just decided on can name it.
    """

    model_architecture = "model_set"

    def __init__(
        self,
        recsys_config: Optional[ModelSetSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        self.main: Optional[RecommenderModel] = None
        self.fallback: Optional[RecommenderModel] = None
        self.session: Optional[RecommenderModel] = None

    # --- construction ---------------------------------------------------------

    def _build(self, member_config) -> Optional[RecommenderModel]:
        """Instantiate the model that implements this settings type.

        No role is hardcoded to a class: the role is the config FIELD, the model
        is the config's TYPE. Swapping the algorithm in a role is a change to
        `app/src/settings.py`, not to this file.
        """
        if member_config is None:
            return None
        model_class = MODEL_FOR_SETTINGS.get(type(member_config))
        if model_class is None:
            raise ValueError(
                f"No model registered for {type(member_config).__name__}. "
                f"Add it to MODEL_FOR_SETTINGS and to RankerSettings."
            )
        return model_class(
            recsys_config=member_config,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def train(self, dataset: Dataset) -> None:
        assert self.recsys_config is not None
        config = self.recsys_config

        self.main = self._build(config.main)
        self.main.train(dataset)

        self.fallback = self._build(config.fallback)
        self.fallback.train(dataset)

        self.session = self._build(config.session)
        if self.session is not None:
            self.session.train(dataset)

        logger.info(
            "Model set trained [%s]: main=%s fallback=%s session=%s",
            self.model_name,
            type(self.main).__name__,
            type(self.fallback).__name__,
            "none" if self.session is None else type(self.session).__name__,
        )

    @classmethod
    def from_legacy_als_state(cls, state: Dict[str, Any]) -> "RecommenderModelSet":
        """Adapt an artifact pickled when RecommenderALS was the whole feed.

        Every model.pkl currently in either bucket is a RecommenderALS __dict__
        carrying `model_hot_users`, `model_cold_users`, an optional `covis` and a
        `recsys_config`. Rather than keeping a second copy of the routing for
        those, we rebuild the members from that state and route them through the
        same table as a freshly trained set. One adapter instead of two code
        paths, which is the point: the legacy surface is this method, and it is
        deleted once both artifacts have been retrained.

        Note `model_cold_users` is a bare rectools PopularModel, not a
        RecommenderPopular, so the member is assembled field by field.
        """
        config = ModelSetSettings.from_legacy_als_settings(state["recsys_config"])

        main = RecommenderALS(
            recsys_config=config.main,
            model_name=state.get("model_name"),
            model_version=state.get("model_version"),
        )
        # Everything the ALS half of the legacy state owns. Names are the frozen
        # instance-attribute names from before the split, so this is a copy, not
        # a rename - see CLAUDE.md §5.2 for why a rename here would be silent.
        for attr in (
            "model_hot_users",
            "dataset",
            "item_id_map",
            "user_id_map",
            "user_item_matrix_binary",
            "implicit_neginf_score",
            "item_similarity",
            "user_ids_hot",
            "item_ids_hot",
        ):
            if attr in state:
                setattr(main, attr, state[attr])

        fallback = RecommenderPopular(
            recsys_config=config.fallback,
            model_name=state.get("model_name"),
            model_version=state.get("model_version"),
        )
        fallback.model = state.get("model_cold_users")
        fallback.dataset = state.get("dataset")
        fallback.item_id_map = state.get("item_id_map")
        fallback.user_id_map = state.get("user_id_map")

        # The covis member was already a full RecommenderCoVis inside the pickle.
        session = state.get("covis")

        instance = cls(
            recsys_config=config,
            model_name=state.get("model_name"),
            model_version=state.get("model_version"),
        )
        instance.main = main
        instance.fallback = fallback
        instance.session = session
        logger.info(
            "Adapted a legacy ALS artifact into a model set: session=%s",
            "none" if session is None else type(session).__name__,
        )
        return instance

    # --- routing --------------------------------------------------------------

    def _session_available(self) -> bool:
        """True when the session member exists and actually has a model behind it."""
        session = self.session
        return session is not None and bool(getattr(session, "neighbors", None))

    def recommend(
        self,
        user_ids: Any,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[Any]] = None,
        history: Optional[List[str]] = None,
    ) -> RecomItems:
        """THE routing table. Segment in, one member's answer out, labelled here."""
        total_start = perf_counter()

        if isinstance(items_to_recommend, list) and len(items_to_recommend) == 0:
            return RecomItems(
                item_ids=[],
                scores=[],
                strategy=Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY.value,
            )

        has_session = history is not None and len(history) > 0
        known_user = self.main is not None and self.main.can_serve(user_ids)
        # The main ranker cannot score items it never saw; the session and
        # fallback members do their own filtering.
        main_items = self.main.known_items(items_to_recommend) if self.main is not None else items_to_recommend

        if known_user and has_session:
            result, session_used = self._known_user_with_session(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=main_items,
                history=history,
            )
            # A session that contributed nothing - every seed unknown to the
            # models, or the session member came back empty - must NOT be
            # labelled realtime. That answer is plain main-ranker output, and
            # mislabelling it would poison every A/B readout split on strategy.
            segment = Strategy.MODEL_REALTIME_HOT_USERS if session_used else Strategy.MODEL_HOT_USERS
        elif known_user:
            result = self.main.recommend(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=main_items,
            )
            segment = Strategy.MODEL_HOT_USERS
        elif has_session:
            result = self._unknown_user_with_session(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )
            segment = Strategy.MODEL_REALTIME_WARM_USERS
            # "Nothing in the candidate list is scorable" is its own outcome and
            # not a segment; the main ranker reports it and it passes through.
            if result.strategy == Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY.value:
                segment = Strategy.NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY
        else:
            result = self.fallback.recommend(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
            )
            segment = Strategy.MODEL_COLD_USERS

        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            "[SET %s] user=%s known=%s session=%s -> %s | total=%.1fms | results=%s",
            self.model_name,
            user_ids,
            known_user,
            has_session,
            segment.value,
            total_ms,
            len(result.item_ids),
        )
        # Members return whatever label suits them standalone; the segment is the
        # set's call, so it is stamped here unconditionally.
        return RecomItems(item_ids=result.item_ids, scores=result.scores, strategy=segment.value)

    def _known_user_with_session(
        self,
        user_ids: Any,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[Any]],
        history: List[str],
    ) -> tuple[RecomItems, bool]:
        """Known user with a live session: fuse the two rankings, or let main enrich itself.

        With a session member present this is a weighted RRF blend of the main
        ranker and the session ranker - offline that beat pure ALS by ~6% map@10
        (EXPERIMENTS.md 2026-08-03). Both sides are overfetched 2x so the fusion
        has real overlap before the cut to top_n. With no session member, the
        main ranker enriches itself from the session and we simply pass that on.

        Returns the answer and whether the session actually contributed, which
        is what the caller needs to name the segment honestly.
        """
        if not self._session_available():
            return self.main.recommend_hot_user_with_session(
                user_ids=user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )

        blend = self.recsys_config.blend
        fetch_n = top_n * 2

        session_result = self.session.recommend(
            user_ids,
            top_n=fetch_n,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
            history=history,
        )
        _, main_external, main_scores = self.main.hot_user_candidates(
            user_ids=user_ids,
            top_n=fetch_n,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        main_list = [str(item) for item in main_external]

        if not session_result.item_ids:
            logger.info("[SET BLEND] session empty, serving main only: user=%s", user_ids)
            return (
                RecomItems(
                    item_ids=main_list[:top_n],
                    scores=main_scores.astype(float).tolist()[:top_n],
                    strategy=None,
                ),
                False,
            )

        fused = rrf_fuse(
            {"main": main_list, "session": list(session_result.item_ids)},
            {"main": blend.MAIN_WEIGHT, "session": blend.SESSION_WEIGHT},
            rrf_k=blend.RRF_K,
        )
        # The session member only filters its own seed, so training-viewed items
        # have to be dropped here for filter_viewed to mean the same thing as on
        # the main-only path. Only the main ranker knows that set.
        exclude = self.main.viewed_external(user_ids) if filter_viewed else set()
        items = [item for item, _ in fused if item not in exclude][:top_n]
        scores = [1.0 / rank for rank in range(1, len(items) + 1)]
        logger.info(
            "[SET BLEND] user=%s | main=%s session=%s -> fused=%s",
            user_ids,
            len(main_list),
            len(session_result.item_ids),
            len(items),
        )
        return RecomItems(item_ids=items, scores=scores, strategy=None), True

    def _unknown_user_with_session(
        self,
        user_ids: Any,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[Any]],
        history: List[str],
    ) -> RecomItems:
        """Unknown visitor with a live session: the session member, else main's own.

        The session member is preferred because the main ranker's item-similarity
        path measured BELOW plain popularity offline while co-visitation was 4x
        better (EXPERIMENTS.md 2026-08-02). An empty session answer falls through
        to the main ranker rather than to popularity - a session we cannot score
        is still more informative than no session at all.
        """
        if self._session_available():
            session_result = self.session.recommend(
                user_ids,
                top_n=top_n,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                history=history,
            )
            if session_result.item_ids:
                return session_result
            logger.info("[SET] session member returned nothing, falling back to main: user=%s", user_ids)

        result, _ = self.main.recommend_from_session(
            user_ids=user_ids,
            top_n=top_n,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
            history=history,
        )
        return result

    # --- plumbing -------------------------------------------------------------

    def warm_caches(self) -> None:
        """Warm every member that has lazy caches, before the first request."""
        for member in (self.main, self.fallback, self.session):
            warm = getattr(member, "warm_caches", None)
            if warm is not None:
                warm()

    def calc_metrics(self, k: int, dataset: Dataset, n_splits: int = 3) -> Dict[str, Any]:
        """Metrics per member, keyed by role. Each member measures only itself."""
        results: Dict[str, Any] = {}
        for role, member in (("main", self.main), ("fallback", self.fallback), ("session", self.session)):
            if member is None:
                continue
            try:
                results[role] = member.calc_metrics(k=k, dataset=dataset, n_splits=n_splits)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metrics for member %s failed: %s", role, exc)
        return results

    def save_model_triton(self, base_s3_url: Pathy, num_to_keep: int) -> None:
        if self.model_version is None:
            raise Exception("There isn't model_version, please fill this field")

        logger.info(f"Saving model set to {base_s3_url}")
        # user_item_matrix_binary is a derived cache rebuilt on load; shipping it
        # inflated the artifact for nothing. The legacy ALS artifact dropped it
        # the same way, so keep doing it or the pickle grows on the next retrain.
        cached = getattr(self.main, "user_item_matrix_binary", None)
        if cached is not None:
            self.main.user_item_matrix_binary = None
        try:
            upload_model_files(
                base_s3_url,
                model_name=self.model_name,
                model_version=self.model_version,
                model_data=self.__dict__,
            )
        finally:
            if cached is not None:
                self.main.user_item_matrix_binary = cached
        clean_old_model_versions(base_s3_url=base_s3_url, model_name=self.model_name, num_to_keep=num_to_keep)
        logger.info("Model set saved, old versions deleted.")
