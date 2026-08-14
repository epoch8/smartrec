import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from pathy import Pathy
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import CoVisSettings, RecomItems, Strategy
from smartrec_lib.recommenders import RecommenderModel
from smartrec_lib.save_and_load_triton_models import (
    clean_old_model_versions,
    upload_model_files,
)

logger = logging.getLogger("CoVis Model")
logger.setLevel(logging.INFO)


class RecommenderCoVis(RecommenderModel):
    """
    Session-based item-item co-visitation recommender.

    Builds a tour->tour co-occurrence matrix from user baskets and, at serve time,
    scores candidate tours by their co-occurrence with the items in the user's
    real-time history (session). In offline next-item benchmarks this beat static
    popularity by a wide margin and personalized cold users right after their first
    click. Requires session history; users with no history return an empty result
    and are routed elsewhere by the orchestrator / cascade layer.

    Stored artifact is compact: only top-K neighbors per item (external tour ids).
    """

    model_architecture = "covis"

    def __init__(
        self,
        recsys_config: Optional[CoVisSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version or "-"
        self.model_name = model_name or "-"
        self.recsys_config = recsys_config

        # {tour_id: [(neighbor_tour_id, score), ...]} sorted by score desc
        self.neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self.top_k: int = 100
        # "sw" scoring variant: multiply seed recency by the API event weight.
        self.session_weights_enabled: bool = False

    @staticmethod
    def _parse_history(history: Optional[List]) -> List[Tuple[str, float]]:
        """Parse history entries "tour_id:weight" / "tour_id" / int into
        (tour_id, weight) pairs; missing or malformed weight -> 1.0.

        Accepts any sequence, including the numpy array of (possibly bytes)
        strings that Triton serving passes in: `if not history` on a
        multi-element ndarray raises ValueError, hence the explicit len check.
        """
        if history is None or len(history) == 0:
            return []
        items: List[Tuple[str, float]] = []
        for entry in history:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8", errors="ignore")
            s = str(entry)
            if ":" in s:
                tour_id, _, raw_w = s.partition(":")
                try:
                    weight = float(raw_w)
                except ValueError:
                    weight = 1.0
                items.append((tour_id, weight))
            else:
                items.append((s, 1.0))
        return items

    def train(self, dataset: Dataset):
        assert self.recsys_config is not None

        self.top_k = self.recsys_config.COVIS_TOP_K
        min_cooc = self.recsys_config.COVIS_MIN_COOC
        self.session_weights_enabled = bool(self.recsys_config.COVIS_SESSION_WEIGHTS)

        logger.info("Building co-visitation matrix...")
        df = dataset.interactions.df
        # rectools stores internal ids; map back to external tour ids.
        users = dataset.user_id_map.convert_to_external(df[Columns.User].values)
        items = dataset.item_id_map.convert_to_external(df[Columns.Item].values)

        baskets: Dict[Any, set] = defaultdict(set)
        for u, it in zip(users, items):
            baskets[u].add(str(it))

        cooc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for basket in baskets.values():
            if len(basket) < 2:
                continue
            for a, b in combinations(basket, 2):
                cooc[a][b] += 1
                cooc[b][a] += 1

        self.neighbors = {}
        for item, nb in cooc.items():
            kept = [(n, float(c)) for n, c in nb.items() if c >= min_cooc]
            kept.sort(key=lambda x: x[1], reverse=True)
            if kept:
                self.neighbors[item] = kept[: self.top_k]

        logger.info(
            f"Co-visitation matrix built. items_with_neighbors={len(self.neighbors)}, " f"baskets={len(baskets)}"
        )

    def recommend(
        self,
        user_ids: int,
        top_n: int = 20,
        filter_viewed: bool = True,
        items_to_recommend: Optional[List[int]] = None,
        history: Optional[List[int]] = None,
    ) -> RecomItems:  # Return type is a RecomItems
        seed = self._parse_history(history)
        if not seed:
            # No session signal: routed elsewhere (popularity / segment / cascade).
            return RecomItems(item_ids=[], scores=[], strategy=Strategy.MODEL_REALTIME_WARM_USERS.value)

        allow = set(str(x) for x in items_to_recommend) if items_to_recommend is not None else None
        seen = set(item for item, _ in seed)

        # getattr: artifacts pickled before the flag existed lack the attribute.
        use_session_weights = getattr(self, "session_weights_enabled", False)

        # Recency-weighted aggregation of neighbor scores. get_user_history returns
        # most-recent-first, so earlier positions get a higher weight. With
        # COVIS_SESSION_WEIGHTS on, the API event weight of the seed (view 1.0,
        # booking intent 2.0, paid 3.0) multiplies recency ("sw" variant,
        # EXPERIMENTS.md 2026-08-03/04).
        scores: Dict[str, float] = defaultdict(float)
        n = len(seed)
        for pos, (item, seed_weight) in enumerate(seed):
            recency = (n - pos) / n
            mult = recency * (seed_weight if use_session_weights else 1.0)
            for neighbor, w in self.neighbors.get(item, []):
                if filter_viewed and neighbor in seen:
                    continue
                if allow is not None and neighbor not in allow:
                    continue
                scores[neighbor] += w * mult

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        logger.info(f"CoVis for user {user_ids}: seed={len(seed)}, results={len(ranked)}")

        return RecomItems(
            item_ids=[it for it, _ in ranked],
            scores=[s for _, s in ranked],
            strategy=Strategy.MODEL_REALTIME_WARM_USERS.value,
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

    def calc_metrics(self, k: int, dataset: Dataset, n_splits: int = 3) -> Dict[str, Any]:
        # Co-visitation is a session/next-item model; it is evaluated with a
        # dedicated next-item harness offline, not with rectools TimeRangeSplitter CV.
        logger.info("calc_metrics is not applicable to CoVis (evaluated offline via next-item harness)")
        return {}
