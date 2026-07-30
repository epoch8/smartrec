import typing as tp

from pydantic import BaseModel
from rectools.models.base import ModelConfig


class SourceSpec(BaseModel):
    """One candidate source: a rectools model config plus its fusion weight."""

    model: tp.Dict[str, tp.Any]  # config for rectools model_from_config, incl. "cls"
    weight: float = 1.0
    is_session: bool = False  # session source: weight is additionally scaled by session_weight()


class PolicyModelConfig(ModelConfig):
    """Config for the candidate -> fuse -> constrain serving policy."""

    sources: tp.Dict[str, SourceSpec] = {}
    fallback_source: str = "popular"
    rrf_k: int = 60
    overfetch: int = 3  # each source is asked for k * overfetch candidates
    category_feature: tp.Optional[str] = None  # item feature used by the share cap
    category_share_cap: float = 1.0  # 1.0 disables the cap
    session_weight_tiers: tp.List[tp.Tuple[int, float]] = [(1, 1.0)]
