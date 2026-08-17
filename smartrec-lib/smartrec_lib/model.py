from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RecomItems(BaseModel):
    item_ids: List[str]
    scores: List[float]
    strategy: Optional[str] = None


class CommonRecommenderSettings(BaseSettings):
    RECOMMENDER_DAYS_THRESHOLD: int = 7
    RECOMMENDER_RANDOM_STATE: int = 42


class PopularSettings(CommonRecommenderSettings):
    POPULARITY_STRATEGY: Literal["n_users", "n_interactions", "mean_weight", "sum_weight"] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)


class CoVisSettings(CommonRecommenderSettings):
    # Item-item co-visitation from user baskets. Session-based ranker: scores
    # candidates by co-occurrence with the items in the user's real-time history.
    COVIS_TOP_K: int = 100  # neighbors kept per item
    COVIS_MIN_COOC: int = 2  # minimum co-occurrence count to keep an edge
    COVIS_SESSION_WEIGHTS: bool = False  # "sw": seed recency x API event weight


class BlendSettings(BaseModel):
    """Weighted reciprocal-rank fusion of two candidate lists.

    Used only when a model composes two rankers (today: ALS x CoVis for hot
    users with a live session). Weights come from the offline policy grid
    (EXPERIMENTS.md 2026-08-03): als=1.0 + covis=1.0 beat pure ALS by ~6% map@10.
    """

    ALS_WEIGHT: float = 1.0
    COVIS_WEIGHT: float = 1.0
    RRF_K: int = 60


# Flat fields that ALSSettings used to carry for its sub-models, in the order
# (legacy name -> where the value now lives). Kept only to migrate artifacts
# pickled before the nested shape existed; nothing writes them any more.
_LEGACY_ALS_POPULAR_FIELDS = ("POPULARITY_STRATEGY", "POPULARITY_PERIOD")
_LEGACY_ALS_COVIS_FIELDS = ("COVIS_TOP_K", "COVIS_MIN_COOC", "COVIS_SESSION_WEIGHTS")
_LEGACY_ALS_BLEND_FIELDS = {
    "BLEND_ALS_WEIGHT": "ALS_WEIGHT",
    "BLEND_COVIS_WEIGHT": "COVIS_WEIGHT",
    "BLEND_RRF_K": "RRF_K",
}


class ALSSettings(CommonRecommenderSettings):
    """Config for RecommenderALS, which is a model set rather than one model.

    Composition is explicit: `popular` configures the PopularModel that serves
    cold users, and `covis` configures the optional co-visitation session layer.
    `covis is None` means there is no session layer at all - that is what the
    old SESSION_COVIS_ENABLED=False meant. `blend` is only read when `covis` is
    set (it fuses the ALS and CoVis candidate lists for hot users in session).
    """

    ALS_ITERATIONS: int
    ALS_REGULARIZATION_FACTOR: float
    ALS_FACTORS: int  # latent embeddings size
    ALS_ALPHA: int  # confidence multiplier for non-zero entries in interactions

    # Cold-user fallback embedded in the ALS artifact.
    popular: PopularSettings = Field(default_factory=PopularSettings)
    # Session layer via co-visitation (absent by default: prod behavior unchanged).
    # When present, the two ALS session paths are replaced per the offline research
    # (EXPERIMENTS.md 2026-08-02/03): hot user + session -> RRF blend of ALS and
    # CoVis (+6% map@10 vs pure ALS); unknown user + session -> CoVis (the ALS
    # item-sim it replaces measured below plain popularity, covis is 4x).
    covis: Optional[CoVisSettings] = None
    blend: BlendSettings = Field(default_factory=BlendSettings)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Load artifacts pickled with the pre-composition (flat) field shape.

        `recsys_config` is pickled inside every model.pkl in S3, so a 900MB
        artifact trained before this refactor still unpickles into this class.
        Pickle restores __dict__ verbatim and skips validation, so the nested
        sub-configs would simply be missing; rebuild them from the flat fields.
        The stale flat keys are left in __dict__ so any straggler reader of
        e.g. config.POPULARITY_STRATEGY keeps working.
        """
        super().__setstate__(state)
        self._migrate_legacy_flat_fields()

    def _migrate_legacy_flat_fields(self) -> None:
        fields = self.__dict__

        if "popular" not in fields:
            fields["popular"] = PopularSettings(
                RECOMMENDER_DAYS_THRESHOLD=fields.get("RECOMMENDER_DAYS_THRESHOLD", 7),
                RECOMMENDER_RANDOM_STATE=fields.get("RECOMMENDER_RANDOM_STATE", 42),
                **{name: fields[name] for name in _LEGACY_ALS_POPULAR_FIELDS if name in fields},
            )

        if "covis" not in fields:
            # A flat config only had a session layer when the boolean was on.
            if fields.get("SESSION_COVIS_ENABLED", False):
                fields["covis"] = CoVisSettings(
                    RECOMMENDER_DAYS_THRESHOLD=fields.get("RECOMMENDER_DAYS_THRESHOLD", 7),
                    RECOMMENDER_RANDOM_STATE=fields.get("RECOMMENDER_RANDOM_STATE", 42),
                    **{name: fields[name] for name in _LEGACY_ALS_COVIS_FIELDS if name in fields},
                )
            else:
                fields["covis"] = None

        if "blend" not in fields:
            fields["blend"] = BlendSettings(
                **{new: fields[old] for old, new in _LEGACY_ALS_BLEND_FIELDS.items() if old in fields}
            )


class EASESettings(CommonRecommenderSettings):
    # Regularization for the closed-form item-item EASE model.
    # 250 was the best value in offline k-fold sweeps (30-day training window).
    # Warm ranker only: cold-user routing is handled by the Popular fallback.
    EASE_REGULARIZATION: float = 250.0


class Strategy(Enum):
    """How a recommendation was produced, in terms of what was known about the
    visitor - NOT which algorithm ran inside.

    A strategy names the user segment and the signal used. Which artifact
    answered, and therefore which algorithm, is already carried by `model_name`
    alongside it, so the two together are unambiguous. Introducing a new strategy
    string for a new internal algorithm is a mistake: it breaks every dashboard
    and every consumer watching for the segment, and tells them nothing they
    could not read off `model_name`. Add a value here only for a genuinely new
    segment or signal.
    """

    # Standard model-based strategies (trained on historical data)
    MODEL_HOT_USERS = "model_hot_users"  # User in training data, ALS embeddings
    MODEL_COLD_USERS = "model_cold_users"  # New user, popular items
    MODEL_WARM_USERS = "model_warm_users"
    # No model emits this any more (its last emitter, RecommenderRandom, is gone).
    # Kept because the string is a published contract - see api/docs/DEBUG_INFO_CODEC.md.
    MODEL_HOT_AND_COLD_USERS = "model_hot_and_cold_users"

    # Real-time strategies (enriched with session events from Redis). These are
    # emitted whichever session algorithm the artifact uses: als_youtravel scores
    # the session with ALS item-similarity, als_covis_youtravel with an RRF blend
    # of ALS and co-visitation. Same segment, same signal, same label.
    MODEL_REALTIME_HOT_USERS = "model_realtime_hot_users"  # Hot user + real-time events
    MODEL_REALTIME_WARM_USERS = "model_realtime_warm_users"  # New user with real-time events
    MODEL_REALTIME_COLD_USERS = "model_realtime_cold_users"  # Cold user with filtered popular by events

    # Retired, reserved. These briefly labelled the two als_covis session paths,
    # which was the mistake described above: consumers waiting for
    # model_realtime_* stopped seeing anything, while the strings carried only
    # what model_name already said. Nothing emits them now; the values stay
    # because their numeric ids in api/docs/DEBUG_INFO_CODEC.md are append-only.
    MODEL_ALS_COVIS_BLEND = "als_covis_blend"
    MODEL_COVIS_SESSION = "covis_session"

    # Fallback
    NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY = "no_strategy_items_to_recommend_filtered_is_empty"
