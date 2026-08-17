"""Layer L1: pure algorithm kernels. See ../../CLAUDE.md section 2.

Everything here takes primitives and returns primitives: no classes with state,
no I/O, no rectools, no pydantic, no logging. That is what lets the serving
layer keep external tour-id strings while the research layer keeps rectools
internal ints, with one implementation between them.

Nothing here is ever pickled - artifacts store data and scalars, never
callables - so these modules can be moved or renamed freely.

Modules import from the submodule directly (`from smartrec_lib.kernels.fusion
import rrf_fuse`); these re-exports are for callers outside the package.
"""

from smartrec_lib.kernels.constraints import apply_share_cap
from smartrec_lib.kernels.cooccurrence import build_neighbor_map, score_session
from smartrec_lib.kernels.fusion import rrf_fuse, session_weight

__all__ = ["apply_share_cap", "build_neighbor_map", "rrf_fuse", "score_session", "session_weight"]
