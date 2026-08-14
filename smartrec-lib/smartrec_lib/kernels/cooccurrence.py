"""Co-visitation algorithm kernel - shared by both layers, belongs to neither.

Layer L1 (see ../../CLAUDE.md). Generic over the item id type: it groups, counts
and sorts ids but never converts them, which is what lets the serving layer keep
external tour-id strings and the research layer keep rectools internal ints.

This module is the single implementation of the algorithm that used to be
written twice. Its two callers are `recommenders/recommender_covis.py` (serving)
and `research/covis.py` (research). The four ways those copies had drifted are
explicit parameters rather than hard-coded choices, so each caller states its
own behaviour:

    fit_basket_size    cap on the interactions per user that enter a basket
    session_size       cap on the session seed length
    seed_weights       per-seed multiplier on top of recency
    deterministic_ties total ordering for equal counts / equal scores

`deterministic_ties=False` reproduces the historical serving behaviour, where
equal values keep the incidental order they were produced in. It is a
reproducibility defect (the fitted graph then depends on PYTHONHASHSEED, see
docs/DESIGN_UNIFICATION.md section 1.4), kept only so that the serving shell
stays bit-identical to the artifacts already in S3. `True` requires the id type
to be orderable.
"""

import typing as tp
from collections import defaultdict
from itertools import combinations

ItemT = tp.TypeVar("ItemT")

# {item: [(neighbour, co-occurrence count as float), ...]}, best first.
NeighborMap = tp.Dict[ItemT, tp.List[tp.Tuple[ItemT, float]]]


def build_neighbor_map(
    baskets: tp.Iterable[tp.Sequence[ItemT]],
    *,
    min_cooc: int,
    top_k: int,
    deterministic_ties: bool,
    fit_basket_size: tp.Optional[int] = None,
) -> NeighborMap:
    """Count item-item co-occurrence over baskets and keep the top `top_k` per item.

    `baskets` yields one sequence of item ids per user, MOST RECENT FIRST -
    ordering only matters when `fit_basket_size` truncates. Duplicates are
    allowed; each basket is deduplicated before pairing, so an item pair counts
    once per user regardless of how often either side was interacted with.

    `fit_basket_size` caps a basket to that many most recent entries before
    deduplication. A basket of N distinct items yields C(N, 2) pairs, so without
    a cap the fit cost is unbounded in the longest user history. `None` means no
    cap.

    Edges with fewer than `min_cooc` co-occurrences are dropped.
    """
    cooc: tp.Dict[ItemT, tp.Dict[ItemT, int]] = defaultdict(lambda: defaultdict(int))
    for basket in baskets:
        recent = basket if fit_basket_size is None else basket[:fit_basket_size]
        distinct = set(recent)
        if len(distinct) < 2:
            continue
        # With deterministic_ties off, the pair enumeration order decides which of
        # several equally-co-occurring neighbours survive `top_k` below, because the
        # sort is stable. Enumerating a sorted basket removes that dependency.
        pairs = combinations(sorted(distinct), 2) if deterministic_ties else combinations(distinct, 2)
        for a, b in pairs:
            cooc[a][b] += 1
            cooc[b][a] += 1

    neighbors: NeighborMap = {}
    for item, counts in cooc.items():
        kept = [(neighbor, float(count)) for neighbor, count in counts.items() if count >= min_cooc]
        if deterministic_ties:
            kept.sort(key=lambda pair: (-pair[1], pair[0]))
        else:
            kept.sort(key=lambda pair: pair[1], reverse=True)
        if kept:
            neighbors[item] = kept[:top_k]
    return neighbors


def score_session(
    neighbors: tp.Mapping[ItemT, tp.Sequence[tp.Tuple[ItemT, float]]],
    session: tp.Sequence[ItemT],
    *,
    k: int,
    deterministic_ties: bool,
    exclude: tp.AbstractSet[ItemT] = frozenset(),
    allowed: tp.Optional[tp.AbstractSet[ItemT]] = None,
    session_size: tp.Optional[int] = None,
    seed_weights: tp.Optional[tp.Sequence[float]] = None,
) -> tp.List[tp.Tuple[ItemT, float]]:
    """Rank candidates by recency-weighted co-occurrence with a session.

    `session` is most recent first; seed at position `pos` of `n` contributes
    with weight `(n - pos) / n`, so `n` - and therefore every score - depends on
    the seed length. `session_size` truncates the seed to that many most recent
    items (`None` = no cap); because the cap changes `n`, it changes scores as
    well as which seeds contribute.

    `seed_weights` is an optional per-seed multiplier aligned with `session`
    (before truncation), e.g. the API event weight of each session event. `None`
    means plain recency.

    `exclude` is subtracted from the candidates - callers pass the user's whole
    viewed set, not just the (possibly truncated) seed. `allowed`, when given,
    restricts candidates to that set. Seeds absent from `neighbors` contribute
    nothing.
    """
    seed = list(session) if session_size is None else list(session)[:session_size]
    if not seed:
        return []

    weights: tp.Optional[tp.List[float]] = None
    if seed_weights is not None:
        if len(seed_weights) < len(seed):
            raise ValueError(f"seed_weights is shorter than the session: {len(seed_weights)} < {len(seed)}")
        weights = list(seed_weights[: len(seed)])

    scores: tp.Dict[ItemT, float] = defaultdict(float)
    n = len(seed)
    for pos, item in enumerate(seed):
        recency = (n - pos) / n
        multiplier = recency if weights is None else recency * weights[pos]
        for neighbor, weight in neighbors.get(item, ()):
            if neighbor in exclude:
                continue
            if allowed is not None and neighbor not in allowed:
                continue
            scores[neighbor] += weight * multiplier

    if deterministic_ties:
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    else:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
