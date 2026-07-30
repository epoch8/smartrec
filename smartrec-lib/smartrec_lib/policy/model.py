import typing as tp

import typing_extensions as tpe
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.base import ModelBase
from rectools.models.serialization import model_from_config

from smartrec_lib.policy.config import PolicyModelConfig, SourceSpec
from smartrec_lib.policy.constraints import apply_share_cap
from smartrec_lib.policy.fusion import rrf_fuse, session_weight


class PolicyModel(ModelBase[PolicyModelConfig]):
    """
    The serving policy as a regular rectools model: candidate sources ->
    weighted RRF fusion -> category share cap. Cold and warm users fall back to
    `fallback_source`. Being a ModelBase means the whole cascade is evaluated
    end-to-end with the standard `cross_validate` - no bespoke e2e harness.

    Session-strength proxy: `_recommend_u2i` derives the session-weight tier
    from the user's TOTAL train interaction count (a proxy for "how engaged is
    this user"), not from a live session. Online serving instead uses real-time
    session events, so tier semantics will differ between offline evaluation
    and production - keep this in mind when comparing offline numbers to A/B results.
    """

    recommends_for_warm = True
    recommends_for_cold = True
    config_class = PolicyModelConfig

    def __init__(
        self,
        sources: tp.Optional[tp.Mapping[str, tp.Union[SourceSpec, dict]]] = None,
        fallback_source: str = "popular",
        rrf_k: int = 60,
        overfetch: int = 3,
        category_feature: tp.Optional[str] = None,
        category_share_cap: float = 1.0,
        session_weight_tiers: tp.Sequence[tp.Tuple[int, float]] = ((1, 1.0),),
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.source_specs: tp.Dict[str, SourceSpec] = {
            name: spec if isinstance(spec, SourceSpec) else SourceSpec(**spec) for name, spec in (sources or {}).items()
        }
        self.fallback_source = fallback_source
        self.rrf_k = rrf_k
        self.overfetch = overfetch
        self.category_feature = category_feature
        self.category_share_cap = category_share_cap
        self.session_weight_tiers = [tuple(tier) for tier in session_weight_tiers]
        self.models: tp.Dict[str, ModelBase] = {}
        self.item_category: tp.Dict[tp.Any, str] = {}

    def _get_config(self) -> PolicyModelConfig:
        return PolicyModelConfig(
            cls=self.__class__,
            sources=self.source_specs,
            fallback_source=self.fallback_source,
            rrf_k=self.rrf_k,
            overfetch=self.overfetch,
            category_feature=self.category_feature,
            category_share_cap=self.category_share_cap,
            session_weight_tiers=list(self.session_weight_tiers),
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: PolicyModelConfig) -> tpe.Self:
        return cls(
            sources=config.sources,
            fallback_source=config.fallback_source,
            rrf_k=config.rrf_k,
            overfetch=config.overfetch,
            category_feature=config.category_feature,
            category_share_cap=config.category_share_cap,
            session_weight_tiers=config.session_weight_tiers,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:
        if self.fallback_source not in self.source_specs:
            raise ValueError(f"fallback_source '{self.fallback_source}' is not among sources")
        self.models = {}
        for name, spec in self.source_specs.items():
            model = model_from_config(spec.model)
            model.fit(dataset)
            self.models[name] = model
        self.item_category = self._build_item_category(dataset)

    def _build_item_category(self, dataset: Dataset) -> tp.Dict[tp.Any, str]:
        """Map EXTERNAL item id -> category value from a one-hot categorical
        item feature. External keys keep the map valid for any recommend call."""
        if self.category_feature is None or dataset.item_features is None:
            return {}
        features = dataset.item_features
        result: tp.Dict[tp.Any, str] = {}
        values_csc = features.values.tocsc()
        for column, name in enumerate(features.names):
            if not (isinstance(name, tuple) and name[0] == self.category_feature):
                continue
            internal_rows = values_csc[:, column].tocoo().row
            externals = dataset.item_id_map.convert_to_external(internal_rows)
            for external in externals:
                result[external] = str(name[1])
        return result

    def _source_weight(self, name: str, n_session_events: int) -> float:
        spec = self.source_specs[name]
        weight = spec.weight
        if spec.is_session:
            weight *= session_weight(n_session_events, self.session_weight_tiers)
        return weight

    def _recommend_u2i(
        self,
        user_ids,  # InternalIdsArray (hot users)
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend,  # Optional[InternalIdsArray]
    ) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[float]]:
        external_users = list(dataset.user_id_map.convert_to_external(user_ids))
        items_whitelist = None
        if sorted_item_ids_to_recommend is not None:
            items_whitelist = list(dataset.item_id_map.convert_to_external(sorted_item_ids_to_recommend))
        n_fetch = k * self.overfetch

        per_source: tp.Dict[str, tp.Dict[tp.Any, tp.List[tp.Any]]] = {}
        for name, model in self.models.items():
            reco = model.recommend(
                users=external_users,
                dataset=dataset,
                k=n_fetch,
                filter_viewed=filter_viewed,
                items_to_recommend=items_whitelist,
                on_unsupported_targets="ignore",
            )
            per_source[name] = reco.groupby(Columns.User, sort=False)[Columns.Item].agg(list).to_dict()

        # session length per internal user id (train interactions count)
        session_len = dataset.interactions.df[Columns.User].value_counts().to_dict()

        out_users: tp.List[int] = []
        out_items: tp.List[int] = []
        out_scores: tp.List[float] = []
        for internal_user, external_user in zip(user_ids, external_users):
            n_events = int(session_len.get(int(internal_user), 0))
            rankings: tp.Dict[str, tp.List[tp.Any]] = {}
            weights: tp.Dict[str, float] = {}
            for name in self.models:
                items = per_source[name].get(external_user, [])
                weight = self._source_weight(name, n_events)
                if items and weight > 0:
                    rankings[name] = items
                    weights[name] = weight
            fused_items = [item for item, _ in rrf_fuse(rankings, weights, self.rrf_k)]
            if self.category_share_cap < 1.0 and self.item_category:
                fused_items = apply_share_cap(fused_items, self.item_category, k, self.category_share_cap)
            top = fused_items[:k]
            internal_items = dataset.item_id_map.convert_to_internal(top)
            for rank, internal_item in enumerate(internal_items, start=1):
                out_users.append(int(internal_user))
                out_items.append(int(internal_item))
                out_scores.append(1.0 / rank)  # post-cap order is the truth; scores follow rank
        return out_users, out_items, out_scores

    def _fallback_recommend(self, external_users: tp.List[tp.Any], dataset: Dataset, k: int, items_whitelist):
        fallback = self.models[self.fallback_source]
        return fallback.recommend(
            users=external_users,
            dataset=dataset,
            k=k,
            filter_viewed=False,  # cold users have nothing viewed in train
            items_to_recommend=items_whitelist,
            on_unsupported_targets="ignore",
        )

    def _recommend_cold(
        self,
        target_ids,  # ExternalIdsArray (users not in the id map)
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend,
    ) -> tp.Tuple[tp.List[tp.Any], tp.List[int], tp.List[float]]:
        items_whitelist = None
        if sorted_item_ids_to_recommend is not None:
            items_whitelist = list(dataset.item_id_map.convert_to_external(sorted_item_ids_to_recommend))
        reco = self._fallback_recommend(list(target_ids), dataset, k, items_whitelist)
        internal_items = dataset.item_id_map.convert_to_internal(reco[Columns.Item])
        return reco[Columns.User].tolist(), [int(i) for i in internal_items], reco[Columns.Score].tolist()

    def _recommend_u2i_warm(
        self,
        user_ids,  # InternalIdsArray (known users without interactions)
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend,
    ) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[float]]:
        external_users = list(dataset.user_id_map.convert_to_external(user_ids))
        items_whitelist = None
        if sorted_item_ids_to_recommend is not None:
            items_whitelist = list(dataset.item_id_map.convert_to_external(sorted_item_ids_to_recommend))
        reco = self._fallback_recommend(external_users, dataset, k, items_whitelist)
        internal_users = dataset.user_id_map.convert_to_internal(reco[Columns.User])
        internal_items = dataset.item_id_map.convert_to_internal(reco[Columns.Item])
        return (
            [int(u) for u in internal_users],
            [int(i) for i in internal_items],
            reco[Columns.Score].tolist(),
        )
