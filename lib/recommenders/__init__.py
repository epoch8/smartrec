__version__ = "0.0.1"

from smartrec.lib.recommenders.base import RecommenderModel
from smartrec.lib.recommenders.recommender_als import RecommenderALS
from smartrec.lib.recommenders.recommender_lightfm import RecommenderLightFM
from smartrec.lib.recommenders.recommender_popular import RecommenderPopular
from smartrec.lib.recommenders.recommender_random import RecommenderRandom
