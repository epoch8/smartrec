__version__ = "0.0.1"

from smartrec_lib.recommenders.base import RecommenderModel
from smartrec_lib.recommenders.recommender_als import RecommenderALS

# from smartrec_lib.recommenders.recommender_lightfm import RecommenderLightFM
from smartrec_lib.recommenders.recommender_popular import RecommenderPopular
from smartrec_lib.recommenders.recommender_random import RecommenderRandom
