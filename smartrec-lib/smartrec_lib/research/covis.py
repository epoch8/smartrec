"""Layer L3: the research co-visitation model. Shares one algorithm with the
serving shell (`recommenders/recommender_covis.py`) via `kernels.cooccurrence`;
the two differ only in the explicit parameters they pass, which is exactly what
`tests/test_covis_equivalence.py` pins.
"""

import typing as tp
from collections import defaultdict

import typing_extensions as tpe
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, ModelConfig

from smartrec_lib.kernels.cooccurrence import build_neighbor_map, score_session


class CoVisModelConfig(ModelConfig):
    """Config for the session co-visitation model."""

    top_k: int = 100  # neighbors kept per item
    min_cooc: int = 2  # minimum co-occurrence count to keep an edge
    session_size: int = 20  # max most-recent history items used as the session seed
    fit_basket_size: int = 100  # max most-recent train interactions per user used to build baskets in _fit


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
        # Baskets are handed to the kernel most-recent-first, so its
        # `fit_basket_size` truncation keeps each user's most recent
        # interactions. The cap matters because a basket of size N produces
        # C(N, 2) pairs: bot/power users with thousands of interactions would
        # otherwise blow up fit runtime and memory.
        if len(df) > 0:
            df = df.sort_values(Columns.Datetime, ascending=False)
        baskets: tp.Dict[int, tp.List[int]] = defaultdict(list)
        for user, item in zip(df[Columns.User].values, df[Columns.Item].values):
            baskets[int(user)].append(int(item))

        self.neighbors = build_neighbor_map(
            baskets.values(),
            min_cooc=self.min_cooc,
            top_k=self.top_k,
            fit_basket_size=self.fit_basket_size,
            # Research results must be reproducible run to run; the serving
            # shell deliberately keeps hash order instead (CLAUDE.md section 8).
            deterministic_ties=True,
        )

    def _score_session(
        self,
        session: tp.Sequence[int],
        exclude: tp.AbstractSet[int],
        allowed: tp.Optional[tp.AbstractSet[int]],
        k: int,
    ) -> tp.List[tp.Tuple[int, float]]:
        """Recency-weighted neighbor scores for a most-recent-first session (internal ids).

        No per-seed event weights here: the API event weight only exists on the
        serving side, where history arrives as "tour_id:weight".
        """
        return score_session(
            self.neighbors,
            session,
            k=k,
            exclude=exclude,
            allowed=allowed,
            session_size=self.session_size,
            deterministic_ties=True,
        )

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
