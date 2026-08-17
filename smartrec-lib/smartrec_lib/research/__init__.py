"""Layer L3: rectools-native models and the serving policy, for offline work.

These follow the rectools `ModelBase` contract, so configs, save/load and
`cross_validate` come from the framework. Nothing here produces a Triton
artifact, and nothing here may be imported by `recommenders/` or `serving/` -
see ../../CLAUDE.md section 2. Offline protocols live in `evaluation/`.
"""

from smartrec_lib.research.covis import CoVisModel, CoVisModelConfig
from smartrec_lib.research.policy import PolicyModel, PolicyModelConfig, SourceSpec

__all__ = ["CoVisModel", "CoVisModelConfig", "PolicyModel", "PolicyModelConfig", "SourceSpec"]
