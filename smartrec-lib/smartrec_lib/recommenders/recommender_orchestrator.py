import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pathy import Pathy
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import (
    CoVisSettings,
    EASESettings,
    OrchestratorSettings,
    PopularSettings,
    RecomItems,
)
from smartrec_lib.recommenders import (
    RecommenderCoVis,
    RecommenderEASE,
    RecommenderModel,
    RecommenderPopular,
)
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("Orchestrator Model")
logger.setLevel(logging.INFO)


class RecommenderOrchestrator(RecommenderModel):
    """
    Serving policy that owns routing across sub-models in a single artifact.

    Cascade (with backfill to reach top_n):
      1. session history present -> co-visitation
      2. warm user (in training data) -> EASE^R
      3. cold + context has country/region/type -> segment popularity
      4. otherwise -> global popularity

    Keeping the routing here (rather than in the API service) gives one testable
    place for the logic and offline == online parity. Cold-user handling that used
    to be scattered (popular fallback inside each model, filters in serp_subfeeds)
    is expressed once, as an explicit cascade driven by `context`.

    `context` is an optional dict with keys among SEGMENT_DIMS (e.g. {"country":
    "turkey"}), forwarded from the request filters. When absent the orchestrator
    behaves as session -> warm -> global (backward compatible).
    """

    model_architecture = "orchestrator"

    def __init__(
        self,
        recsys_config: Optional[OrchestratorSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        cfg = recsys_config or OrchestratorSettings()
        self.ease = RecommenderEASE(recsys_config=cfg.ease)
        self.covis = RecommenderCoVis(recsys_config=cfg.covis)
        self.popular = RecommenderPopular(
            recsys_config=PopularSettings(
                POPULARITY_STRATEGY=cfg.POPULARITY_STRATEGY,
                POPULARITY_PERIOD=cfg.POPULARITY_PERIOD,
            )
        )
        # {dim: {value: [tour_id, ...]}} cold segment popularity, ordered by popularity
        self.segment_pop: Dict[str, Dict[str, List[str]]] = {}

    def train(self, dataset: Dataset, item_meta: Optional[pd.DataFrame] = None):
        """
        Train all sub-models. `item_meta` (optional) is a DataFrame with a
        `tour_id` column plus SEGMENT_DIMS columns (country/region/type); it is
        used to build cold segment popularity. Without it, the cascade skips the
        segment step and falls back to global popularity.
        """
        assert self.recsys_config is not None

        logger.info("Training orchestrator sub-models (EASE, CoVis, Popular)...")
        self.ease.train(dataset)
        self.covis.train(dataset)
        self.popular.train(dataset)

        self.segment_pop = {}
        if item_meta is not None and len(item_meta):
            self._build_segment_pop(dataset, item_meta)

        logger.info(
            "Orchestrator trained. segment dims=%s",
            {d: len(v) for d, v in self.segment_pop.items()},
        )

    def _build_segment_pop(self, dataset: Dataset, item_meta: pd.DataFrame) -> None:
        df = dataset.interactions.df
        items_ext = [str(i) for i in dataset.item_id_map.convert_to_external(df[Columns.Item].values)]
        users_ext = [str(u) for u in dataset.user_id_map.convert_to_external(df[Columns.User].values)]
        pop = (
            pd.DataFrame({"item": items_ext, "user": users_ext})
            .groupby("item")["user"]
            .nunique()
        )
        pop_map = pop.to_dict()

        meta = item_meta.copy()
        meta["tour_id"] = meta["tour_id"].astype(str)
        top_n = self.recsys_config.SEGMENT_TOP_N

        for dim in self.recsys_config.SEGMENT_DIMS:
            if dim not in meta.columns:
                continue
            dim_lists: Dict[str, List[str]] = {}
            for val, grp in meta.groupby(dim):
                if val is None or str(val) == "" or str(val).lower() == "nan":
                    continue
                tours = [t for t in grp["tour_id"].tolist() if t in pop_map]
                tours.sort(key=lambda t: pop_map[t], reverse=True)
                if tours:
                    dim_lists[str(val)] = tours[:top_n]
            if dim_lists:
                self.segment_pop[dim] = dim_lists

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecomItems:
        allow = set(str(x) for x in items_to_recommend) if items_to_recommend is not None else None
        acc_items: List[str] = []
        acc_scores: List[float] = []
        seen: set = set()
        first_strategy: Optional[str] = None

        # Session-history items are "already viewed": exclude them across every
        # cascade tier (co-vis filters its own seed, but segment/global backfill
        # otherwise could re-surface them).
        if filter_viewed and history:
            for h in history:
                seen.add(str(h).split(":", 1)[0])

        def take(item_ids: List[str], scores: List[float], strat: str) -> None:
            nonlocal first_strategy
            added = False
            for it, sc in zip(item_ids, scores):
                if it in seen:
                    continue
                if allow is not None and it not in allow:
                    continue
                seen.add(it)
                acc_items.append(it)
                acc_scores.append(sc)
                added = True
                if len(acc_items) >= top_n:
                    break
            if added and first_strategy is None:
                first_strategy = strat

        # 1. session -> co-visitation
        if history:
            r = self.covis.recommend(user_ids, top_n, filter_viewed, items_to_recommend, history)
            take(r.item_ids, r.scores, "covis")

        # 2. warm -> EASE (returns empty for cold users)
        if len(acc_items) < top_n:
            r = self.ease.recommend(user_ids, top_n, filter_viewed, items_to_recommend, history)
            take(r.item_ids, r.scores, "ease")

        # 3. cold + context -> segment popularity (most specific dim first)
        if len(acc_items) < top_n and context:
            for dim in self.recsys_config.SEGMENT_DIMS:
                val = context.get(dim)
                if val is None:
                    continue
                lst = self.segment_pop.get(dim, {}).get(str(val))
                if lst:
                    n = len(lst)
                    take(lst, [(n - i) / n for i in range(n)], f"segment_{dim}")
                    break

        # 4. fallback -> global popularity
        if len(acc_items) < top_n:
            r = self.popular.recommend(user_ids, top_n, filter_viewed, items_to_recommend, history)
            take(r.item_ids, r.scores, "global")

        return RecomItems(
            item_ids=acc_items[:top_n],
            scores=acc_scores[:top_n],
            strategy=first_strategy or "empty",
        )

    def save_model_triton(self, base_s3_url: Pathy, num_to_keep: int) -> None:
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

    def calc_metrics(
        self, k: int, dataset: Dataset, n_splits: int = 3
    ) -> Dict[str, Any]:
        # The orchestrator is a routing policy; components are evaluated with their
        # own harnesses (EASE via rectools CV, CoVis via next-item offline).
        logger.info("calc_metrics is not applicable to the orchestrator (evaluate components separately)")
        return {}
