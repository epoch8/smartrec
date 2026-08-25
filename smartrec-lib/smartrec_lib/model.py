from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

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

    Weights are named after the ROLES being fused, not after the models filling
    them. They were `ALS_WEIGHT`/`COVIS_WEIGHT` until 2026-08-22, which meant the
    generic fusion config named two specific algorithms and the fusion call had
    to read `{"main": blend.ALS_WEIGHT, ...: blend.COVIS_WEIGHT}` - visibly
    translating between the two vocabularies. Swap ALS for LightFM and the old
    names would have been actively wrong.

    `REALTIME_WEIGHT` is deliberately not called `SESSION_WEIGHT`: that would
    have been the third distinct meaning of "session weight" in this package,
    next to `CoVisSettings.COVIS_SESSION_WEIGHTS` (a per-seed event multiplier)
    and the `session_weight` argument of the co-occurrence kernel. See CLAUDE.md
    section 4 - one concept, one name.

    Values come from the offline policy grid (EXPERIMENTS.md 2026-08-03):
    1.0 + 1.0 beat the main ranker alone by ~6% map@10.
    """

    MAIN_WEIGHT: float = 1.0
    REALTIME_WEIGHT: float = 1.0
    RRF_K: int = 60


# Flat fields that ALSSettings used to carry for its sub-models, in the order
# (legacy name -> where the value now lives). Kept only to migrate artifacts
# pickled before the nested shape existed; nothing writes them any more.
_LEGACY_ALS_POPULAR_FIELDS = ("POPULARITY_STRATEGY", "POPULARITY_PERIOD")
_LEGACY_ALS_COVIS_FIELDS = ("COVIS_TOP_K", "COVIS_MIN_COOC", "COVIS_SESSION_WEIGHTS")
_LEGACY_ALS_BLEND_FIELDS = {
    "BLEND_ALS_WEIGHT": "MAIN_WEIGHT",
    "BLEND_COVIS_WEIGHT": "REALTIME_WEIGHT",
    "BLEND_RRF_K": "RRF_K",
}


class ALSSettings(CommonRecommenderSettings):
    """Hyperparameters of the ALS matrix-factorisation ranker. Nothing else.

    This class used to also carry `popular`, `covis` and `blend`, i.e. the
    composition of the whole artifact, which put "who serves cold users" and
    "how are two rankers fused" inside the config of one ranker. Composition
    now lives in `ModelSetSettings`; this stays a leaf config.

    The `__setstate__` below is the one reason the old names still appear here:
    artifacts in S3 were pickled with that shape and must keep loading.
    """

    ALS_ITERATIONS: int
    ALS_REGULARIZATION_FACTOR: float
    ALS_FACTORS: int  # latent embeddings size
    ALS_ALPHA: int  # confidence multiplier for non-zero entries in interactions

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Normalise artifacts pickled with the flat, pre-composition shape.

        `recsys_config` is pickled inside every model.pkl in S3, so a 900MB
        artifact trained before composition existed still unpickles into this
        class. Pickle restores __dict__ verbatim and skips validation, so its
        sub-configs would simply be missing; rebuild them from the flat fields.

        They land in __dict__ under names this class no longer declares, which
        is deliberate and is what `ModelSetSettings.from_legacy_als_settings`
        reads. Pydantic resolves __dict__ entries for undeclared names, so
        `config.covis` on such an instance still works. The stale flat keys are
        left alone so any straggler reader of e.g. `config.POPULARITY_STRATEGY`
        keeps working.
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


# A model that can rank the users it has a representation for. The main and
# realtime roles below accept any of these, which is the whole point: swapping the
# algorithm in a role is a change of TYPE, not of schema. Adding LightFM means
# adding LightFMSettings here and to the class registry in recommender_model_set,
# and touching nothing else.
RankerSettings = Union[ALSSettings, EASESettings, CoVisSettings, PopularSettings]


class ModelSetSettings(BaseModel):
    """One servable artifact composed of several models, keyed BY ROLE.

    An artifact is not one model: it is a ranker for the users it can represent,
    a fallback for everyone else, optionally something that scores the live
    realtime signal, and the weights that fuse them.

    **Each field is a ROLE and its value's type is the model filling it.** That
    is deliberate and it is the second half of a lesson learned twice:

    - `ALSSettings.covis` named a sibling model inside one ranker's config, so
      "who serves the realtime signal" looked like an ALS hyperparameter (fixed
      2026-08-22);
    - then the fields here were `als`/`popular`/`covis`, i.e. named after the
      algorithms currently filling the roles. Putting LightFM in as the main
      ranker would have meant either a second field for the same role or a field
      called `als` holding a LightFMSettings. Now it is `main=LightFMSettings(...)`
      and the schema does not move.

    Roles:

    - `main` - ranks users it has a representation for. Required.
    - `fallback` - ranks everyone else. Required; there is always someone the
      main ranker cannot score.
    - `realtime` - ranks from the live session events. `None` means the artifact
      has no realtime layer - exactly what the retired `SESSION_COVIS_ENABLED=False`
      meant - and `main` then scores the live events with its own machinery.
    - `blend` - fusion weights, read only when `realtime` is set.

    The role is called `realtime` because that is the word the published contract
    already uses: the strategies this layer produces are `model_realtime_hot_users`
    and `model_realtime_warm_users` (see `Strategy` and
    api/docs/DEBUG_INFO_CODEC.md). It was briefly `session`, which left the config
    and the wire disagreeing about the same thing.

    The two shipped artifacts are two shapes of this class: `als_youtravel` is
    main+fallback, `als_covis_youtravel` adds realtime+blend. Per the offline
    research (EXPERIMENTS.md 2026-08-02/03) the blend beat the main ranker alone
    by ~6% map@10, and co-visitation beat the item-similarity realtime path it
    replaces by 4x.

    Plain BaseModel, not BaseSettings: composition is not env-driven, and this is
    constructed on the serving path for legacy artifacts, where scanning the
    environment on every request would be a waste.

    Note on shape: one model per role. Fusing two candidate sources within a
    single role (ALS *and* LightFM as main) would want `Dict[str, Member]` with
    the weight on the member instead - that is the next shape, and the trigger
    for it is a second model in one role, not a fourth model overall.
    """

    main: RankerSettings
    fallback: RankerSettings = Field(default_factory=PopularSettings)
    realtime: Optional[RankerSettings] = None
    blend: BlendSettings = Field(default_factory=BlendSettings)

    @classmethod
    def from_legacy_als_settings(cls, config: ALSSettings) -> "ModelSetSettings":
        """Read an artifact pickled while composition still lived on ALSSettings.

        Covers both older shapes: the flat one (`__setstate__` above has already
        rebuilt the sub-configs by the time we get here) and the nested one that
        `als_covis_youtravel` carries today. Reads `__dict__` directly because
        after the split these names are no longer declared fields.

        Note the vocabulary change: those artifacts named the roles after the
        models (`popular`, `covis`), so this is also where the old names are
        translated into roles.
        """
        fields = config.__dict__
        return cls(
            main=ALSSettings(
                RECOMMENDER_DAYS_THRESHOLD=fields.get("RECOMMENDER_DAYS_THRESHOLD", 7),
                RECOMMENDER_RANDOM_STATE=fields.get("RECOMMENDER_RANDOM_STATE", 42),
                ALS_ITERATIONS=fields["ALS_ITERATIONS"],
                ALS_REGULARIZATION_FACTOR=fields["ALS_REGULARIZATION_FACTOR"],
                ALS_FACTORS=fields["ALS_FACTORS"],
                ALS_ALPHA=fields["ALS_ALPHA"],
            ),
            fallback=fields.get("popular") or PopularSettings(),
            realtime=fields.get("covis"),
            blend=fields.get("blend") or BlendSettings(),
        )


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
