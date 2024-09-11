from token import OP
from typing import Literal, Tuple, Optional, List
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from datetime import datetime, timedelta


class RecomItems(BaseModel):
    item_ids: List[str]
    scores: List[float]
    strategy: Optional[str] = None
    
    
class CommonRecommenderSettings(BaseSettings):
    RECOMMENDER_DAYS_THRESHOLD: int = 7
    

class ALSSettings(CommonRecommenderSettings):
    ALS_ITERATIONS: int
    ALS_RANDOM_STATE: int  = 42
    ALS_REGULARIZATION_FACTOR: float
    ALS_FACTORS: int  # latent embeddings size
    ALS_ALPHA: int # confidence multiplier for non-zero entries in interactions
    POPULARITY_STRATEGY: Literal["n_users", "n_interactions", "mean_weight", "sum_weight"] = "n_users"
    POPULARITY_PERIOD: Optional[timedelta] = timedelta(days=7)
    

class LighFMSettings(CommonRecommenderSettings):
    LIGHTFM_RANDOM_STATE: int = 42
    LIGHTFM_NO_COMPONENTS: int = 50
    LIGHTFM_LOSS: Literal["logistic", "warp", "bpr", "warp-kos"] = "bpr"
    LIGHTFM_EPOCHS: int = 1