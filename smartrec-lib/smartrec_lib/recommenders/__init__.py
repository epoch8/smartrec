"""Layer L2: the servable models. See ../../CLAUDE.md sections 2 and 5.

These classes and their module paths are frozen: `RecommenderCoVis` is
reconstructed by module path when an `als_covis_youtravel` artifact is
unpickled, and every settings object they hold comes from `smartrec_lib.model`.
Submodules import `RecommenderModel` from `recommenders.base` directly, so the
order of the re-exports below carries no meaning.
"""

__version__ = "0.0.1"

from smartrec_lib.recommenders.base import RecommenderModel
from smartrec_lib.recommenders.recommender_als import RecommenderALS
from smartrec_lib.recommenders.recommender_covis import RecommenderCoVis
from smartrec_lib.recommenders.recommender_ease import RecommenderEASE
from smartrec_lib.recommenders.recommender_popular import RecommenderPopular

__all__ = [
    "RecommenderModel",
    "RecommenderALS",
    "RecommenderCoVis",
    "RecommenderEASE",
    "RecommenderPopular",
]
