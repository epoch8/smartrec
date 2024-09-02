from typing import Tuple
from pydantic_settings import BaseSettings
from pydantic import BaseModel


class RecomItem(BaseModel):
    item_id: str
    score: float

class RecomItemUsers(BaseModel):
    user_id: Tuple[str, int]
    reco_items: RecomItem

class ClusterSettings(BaseSettings):
    TRUSTED_CLUSTER_OLD_WEEKS_THRESHOLD: int
    TRUSTED_CLUSTER_N_CLUSTERS: int
    TRUSTED_CLUSTER_MAX_ITER: int
    TRUSTED_CLUSTER_INIT_STEPS: int
    TRUSTED_CLUSTER_CHUNK_SIZE: int
    TRUSTED_CLUSTER_SAMPLES_PER_CLUSTER: int
    TRUSTED_CLUSTER_LIKED_SAMPLES_PER_CLUSTER: int
    TRUSTED_CLUSTER_LIKED_SIMILARITY_THRESHOLD: float


class ALSSettings(ClusterSettings):
    ALS_ITERATIONS: int
    ALS_RANDOM_STATE: int
    ALS_REGULARIZATION_FACTOR: float
    ALS_FACTORS: int  # latent embeddings size
    ALS_ALPHA: int # confidence multiplier for non-zero entries in interactions
    ALS_TRAINING_TYPE: str # can be step or init
    ALS_LOAD_FROM_S3: bool
    ALS_MAX_FOLLOWER_POSTS: int
    ALS_RECOMMENDER_DAYS_THRESHOLD: int
    ALS_UPDATE_TABLE: str
    BATCH_SIZE: int
    