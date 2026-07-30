import typing as tp
from collections import defaultdict
from itertools import combinations

import typing_extensions as tpe
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, ModelConfig


class CoVisModelConfig(ModelConfig):
    """Config for the session co-visitation model."""

    top_k: int = 100             # neighbors kept per item
    min_cooc: int = 2            # minimum co-occurrence count to keep an edge
    session_size: int = 20       # max most-recent history items used as the session seed
    fit_basket_size: int = 100   # max most-recent train interactions per user used to build baskets in _fit


class CoVisModel(ModelBase[CoVisModelConfig]):
    """
    Item-item co-visitation over per-user baskets.

    Offline (u2i path) the session seed for a user is their train history from
    the dataset, most recent first. Online the same scoring is exposed through
    `recommend_for_session`, which takes the session explicitly. Both paths go
    through `_score_session`, which is what guarantees offline == online parity.

    Hot users only: warm and cold users get no recommendations here and are
    routed elsewhere by the policy layer.
    """

    recommends_for_warm = False
    recommends_for_cold = False
    config_class = CoVisModelConfig

    def __init__(
        self,
        top_k: int = 100,
        min_cooc: int = 2,
        session_size: int = 20,
        fit_basket_size: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.top_k = top_k
        self.min_cooc = min_cooc
        self.session_size = session_size
        self.fit_basket_size = fit_basket_size
        self.neighbors: tp.Dict[int, tp.List[tp.Tuple[int, float]]] = {}

    def _get_config(self) -> CoVisModelConfig:
        return CoVisModelConfig(
            cls=self.__class__,
            top_k=self.top_k,
            min_cooc=self.min_cooc,
            session_size=self.session_size,
            fit_basket_size=self.fit_basket_size,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: CoVisModelConfig) -> tpe.Self:
        return cls(
            top_k=config.top_k,
            min_cooc=config.min_cooc,
            session_size=config.session_size,
            fit_basket_size=config.fit_basket_size,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:
        df = dataset.interactions.df  # internal ids
        # Cap each user's basket to their most recent `fit_basket_size`
        # interactions before building co-occurrence pairs: a basket of size N
        # produces C(N, 2) pairs, so bot/power users with thousands of
        # interactions would otherwise blow up _fit's runtime and memory.
        if len(df) > 0:
            df = df.sort_values(Columns.Datetime, ascending=False).groupby(Columns.User, sort=False).head(
                self.fit_basket_size
            )
        baskets: tp.Dict[int, tp.Set[int]] = defaultdict(set)
        for user, item in zip(df[Columns.User].values, df[Columns.Item].values):
            baskets[int(user)].add(int(item))

        cooc: tp.Dict[int, tp.Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for basket in baskets.values():
            if len(basket) < 2:
                continue
            for a, b in combinations(sorted(basket), 2):
                cooc[a][b] += 1
                cooc[b][a] += 1

        self.neighbors = {}
        for item, nbrs in cooc.items():
            kept = [(n, float(c)) for n, c in nbrs.items() if c >= self.min_cooc]
            kept.sort(key=lambda pair: (-pair[1], pair[0]))  # deterministic: count desc, id asc
            if kept:
                self.neighbors[item] = kept[: self.top_k]

    def _score_session(
        self,
        session: tp.Sequence[int],
        exclude: tp.AbstractSet[int],
        allowed: tp.Optional[tp.AbstractSet[int]],
        k: int,
    ) -> tp.List[tp.Tuple[int, float]]:
        """Recency-weighted neighbor scores for a most-recent-first session (internal ids)."""
        seed = list(session)[: self.session_size]
        if not seed:
            return []
        scores: tp.Dict[int, float] = defaultdict(float)
        n = len(seed)
        for pos, item in enumerate(seed):
            recency = (n - pos) / n
            for neighbor, weight in self.neighbors.get(item, []):
                if neighbor in exclude:
                    continue
                if allowed is not None and neighbor not in allowed:
                    continue
                scores[neighbor] += weight * recency
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]

    def _recommend_u2i(
        self,
        user_ids,  # InternalIdsArray
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend,  # Optional[InternalIdsArray]
    ) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[float]]:
        df = dataset.interactions.df
        wanted = {int(u) for u in user_ids}
        hist = df[df[Columns.User].isin(wanted)].sort_values(Columns.Datetime, ascending=False)
        sessions = hist.groupby(Columns.User, sort=False)[Columns.Item].agg(list).to_dict()
        allowed = {int(i) for i in sorted_item_ids_to_recommend} if sorted_item_ids_to_recommend is not None else None

        all_users: tp.List[int] = []
        all_items: tp.List[int] = []
        all_scores: tp.List[float] = []
        for user in user_ids:
            session = [int(i) for i in sessions.get(int(user), [])]
            # filter_viewed must exclude EVERYTHING the user saw, not only the
            # (possibly truncated) session seed.
            exclude = set(session) if filter_viewed else set()
            for item, score in self._score_session(session, exclude, allowed, k):
                all_users.append(int(user))
                all_items.append(item)
                all_scores.append(score)
        return all_users, all_items, all_scores

    def recommend_for_session(
        self,
        session: tp.Sequence[tp.Any],
        dataset: Dataset,
        k: int,
        filter_viewed: bool = True,
    ) -> tp.List[tp.Tuple[tp.Any, float]]:
        """
        Online-path scoring: explicit most-recent-first session of EXTERNAL item
        ids (e.g. from Redis / the request). Unknown items are skipped. Shares
        `_score_session` with the offline u2i path (parity guarantee).
        """
        internal = [int(i) for i in dataset.item_id_map.convert_to_internal(session, strict=False)]
        exclude = set(internal) if filter_viewed else set()
        ranked = self._score_session(internal, exclude, None, k)
        externals = dataset.item_id_map.convert_to_external([item for item, _ in ranked])
        return [(ext, score) for ext, (_, score) in zip(externals, ranked)]
