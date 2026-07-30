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
            name: spec if isinstance(spec, SourceSpec) else SourceSpec(**spec)
            for name, spec in (sources or {}).items()
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
