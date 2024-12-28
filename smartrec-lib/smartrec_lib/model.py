from datetime import timedelta
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
    POPULARITY_STRATEGY: Literal[
        "n_users", "n_interactions", "mean_weight", "sum_weight"
    ] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)


class LighFMSettings(CommonRecommenderSettings):
    RECOMMENDER_RANDOM_STATE: int = 42
    LIGHTFM_NO_COMPONENTS: int = 50
    LIGHTFM_LOSS: Literal["logistic", "warp", "bpr", "warp-kos"] = "bpr"
    LIGHTFM_EPOCHS: int = 1


class RandomSettings(CommonRecommenderSettings):
    pass


class PopularSettings(CommonRecommenderSettings):
    POPULARITY_STRATEGY: Literal[
        "n_users", "n_interactions", "mean_weight", "sum_weight"
    ] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)
