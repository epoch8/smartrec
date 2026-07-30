import typing as tp
from collections import defaultdict
from itertools import combinations

import typing_extensions as tpe
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, ModelConfig


class CoVisModelConfig(ModelConfig):
    """Config for the session co-visitation model."""

    top_k: int = 100        # neighbors kept per item
    min_cooc: int = 2       # minimum co-occurrence count to keep an edge
    session_size: int = 20  # max most-recent history items used as the session seed


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

    def __init__(self, top_k: int = 100, min_cooc: int = 2, session_size: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.top_k = top_k
        self.min_cooc = min_cooc
        self.session_size = session_size
        self.neighbors: tp.Dict[int, tp.List[tp.Tuple[int, float]]] = {}

    def _get_config(self) -> CoVisModelConfig:
        return CoVisModelConfig(
            cls=self.__class__,
            top_k=self.top_k,
            min_cooc=self.min_cooc,
            session_size=self.session_size,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: CoVisModelConfig) -> tpe.Self:
        return cls(
            top_k=config.top_k,
            min_cooc=config.min_cooc,
            session_size=config.session_size,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:
        df = dataset.interactions.df  # internal ids
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
