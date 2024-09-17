__version__ = "0.0.1"

from lib.recommenders.base import RecommenderModel
from lib.recommenders.recommender_als import RecommenderALS
from lib.recommenders.recommender_lightfm import RecommenderLightFM
from lib.recommenders.recommender_popular import RecommenderPopular
from lib.recommenders.recommender_random import RecommenderRandom
