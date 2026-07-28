from datetime import timedelta
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RecomItems(BaseModel):
    item_ids: List[str]
    scores: List[float]
    strategy: Optional[str] = None


class CommonRecommenderSettings(BaseSettings):
    RECOMMENDER_DAYS_THRESHOLD: int = 7
    RECOMMENDER_RANDOM_STATE: int = 42


class ALSSettings(CommonRecommenderSettings):
    ALS_ITERATIONS: int
    ALS_REGULARIZATION_FACTOR: float
    ALS_FACTORS: int  # latent embeddings size
    ALS_ALPHA: int  # confidence multiplier for non-zero entries in interactions
    POPULARITY_STRATEGY: Literal["n_users", "n_interactions", "mean_weight", "sum_weight"] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)


class LighFMSettings(CommonRecommenderSettings):
    RECOMMENDER_RANDOM_STATE: int = 42
    LIGHTFM_NO_COMPONENTS: int = 50
    LIGHTFM_LOSS: Literal["logistic", "warp", "bpr", "warp-kos"] = "bpr"
    LIGHTFM_EPOCHS: int = 1


class EASESettings(CommonRecommenderSettings):
    # Regularization for the closed-form item-item EASE model.
    # 250 was the best value in offline k-fold sweeps (30-day training window).
    # Warm ranker only: cold-user routing is handled by the orchestrator layer.
    EASE_REGULARIZATION: float = 250.0


class CoVisSettings(CommonRecommenderSettings):
    # Item-item co-visitation from user baskets. Session-based ranker: scores
    # candidates by co-occurrence with the items in the user's real-time history.
    COVIS_TOP_K: int = 100  # neighbors kept per item
    COVIS_MIN_COOC: int = 2  # minimum co-occurrence count to keep an edge


class OrchestratorSettings(CommonRecommenderSettings):
    # Composes the sub-model configs plus segment/global popularity knobs.
    # One config drives building every component of the orchestrator.
    ease: EASESettings = EASESettings()
    covis: CoVisSettings = CoVisSettings()
    POPULARITY_STRATEGY: Literal["n_users", "n_interactions", "mean_weight", "sum_weight"] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)
    # Item-metadata dimensions for cold segment popularity, most specific first.
    SEGMENT_DIMS: List[str] = ["country", "region", "type"]
    SEGMENT_TOP_N: int = 200


class RandomSettings(CommonRecommenderSettings):
    pass


class PopularSettings(CommonRecommenderSettings):
    POPULARITY_STRATEGY: Literal["n_users", "n_interactions", "mean_weight", "sum_weight"] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)


class Strategy(Enum):
    # Standard model-based strategies (trained on historical data)
    MODEL_HOT_USERS = "model_hot_users"  # User in training data, ALS embeddings
    MODEL_COLD_USERS = "model_cold_users"  # New user, popular items
    MODEL_WARM_USERS = "model_warm_users"
    MODEL_HOT_AND_COLD_USERS = "model_hot_and_cold_users"

    # Real-time strategies (enriched with session events from Redis)
    MODEL_REALTIME_HOT_USERS = "model_realtime_hot_users"  # Hot user + real-time events
    MODEL_REALTIME_WARM_USERS = "model_realtime_warm_users"  # New user with real-time events
    MODEL_REALTIME_COLD_USERS = "model_realtime_cold_users"  # Cold user with filtered popular by events

    # Fallback
    NO_STRATEGY_ITEMS_TO_RECOMMEND_FILTERED_IS_EMPTY = "no_strategy_items_to_recommend_filtered_is_empty"
