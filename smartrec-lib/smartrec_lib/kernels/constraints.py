import math
import typing as tp


def apply_share_cap(
    ranked_items: tp.Sequence[tp.Any],
    item_category: tp.Mapping[tp.Any, str],
    k: int,
    max_share: float,
) -> tp.List[tp.Any]:
    """
    Greedy diversification: walk the ranked list, skip an item once its category
    already holds ceil(max_share * k) slots. If the walk ends short of k, the
    skipped items backfill in their original order - a full page beats a
    strictly capped short one. Items without a category are never capped.

    This is the guard against the mono-feed failure mode: one curious click on
    the Maldives must not turn the whole page into Maldives tours.
    """
    cap = max(1, math.ceil(max_share * k))
    taken: tp.List[tp.Any] = []
    counts: tp.Dict[str, int] = {}
    skipped: tp.List[tp.Any] = []
    for item in ranked_items:
        if len(taken) >= k:
            break
        category = item_category.get(item)
        if category is not None and counts.get(category, 0) >= cap:
            skipped.append(item)
            continue
        taken.append(item)
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    for item in skipped:
        if len(taken) >= k:
            break
        taken.append(item)
    return taken
