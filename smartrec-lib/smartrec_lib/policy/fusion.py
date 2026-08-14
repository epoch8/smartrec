import typing as tp


def rrf_fuse(
    rankings: tp.Mapping[str, tp.Sequence[tp.Any]],
    weights: tp.Mapping[str, float],
    rrf_k: int = 60,
) -> tp.List[tp.Tuple[tp.Any, float]]:
    """
    Weighted Reciprocal Rank Fusion.

    score(item) = sum over sources of weight_s / (rrf_k + rank_s(item)), ranks
    are 1-based. Raw model scores are NOT comparable across sources, ranks are -
    that is the whole point of using RRF instead of score blending. Ties are
    broken by str(item) for determinism.
    """
    scores: tp.Dict[tp.Any, float] = {}
    for source, items in rankings.items():
        weight = weights.get(source, 1.0)
        if weight <= 0:
            continue
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def session_weight(n_session_events: int, tiers: tp.Sequence[tp.Tuple[int, float]]) -> float:
    """
    Piecewise multiplier for the session source: one click is weak evidence of
    intent, repeated activity is strong. `tiers` is [(min_events, multiplier)]
    sorted by min_events ascending, e.g. [(1, 0.3), (3, 1.0)]. Returns 0.0 when
    no tier matches (no session -> the session source is switched off).
    """
    result = 0.0
    for min_events, multiplier in tiers:
        if n_session_events >= min_events:
            result = multiplier
    return result
