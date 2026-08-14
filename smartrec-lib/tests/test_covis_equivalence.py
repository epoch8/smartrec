"""Characterisation tests for the two co-visitation callers.

`research/covis.py::CoVisModel` (rectools-native, used by `evaluation/`) and
`recommenders/recommender_covis.py::RecommenderCoVis` (serving, pickled into
`covis_youtravel` and into the session layer of `als_covis_youtravel`) now share
one algorithm - `kernels/cooccurrence.py` - and differ only in the parameters
they pass and in their id spaces (rectools internal ints vs external tour-id
strings).

This file is an executable spec of exactly where they agree and where they do
not. It asserts nothing about which behaviour is preferable - it pins the facts,
so that changing one caller's parameters without the other fails loudly. The
kernel itself is tested directly in `test_kernels_cooccurrence.py`.

Verdict recorded here (see docs/DESIGN_UNIFICATION.md section 1.4): the cores are
equivalent; the divergences are the two research-only caps, the serving-only
session event weights, and tie-break determinism.
"""

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import CoVisSettings
from smartrec_lib.research.covis import CoVisModel
from smartrec_lib.recommenders import RecommenderCoVis

BASE = pd.Timestamp("2026-07-01")

# Session used by the parity tests, most-recent-first, all items known to the
# shared fixture. Five items, so a session_size of 3 actually truncates.
SESSION = ["pop1", "m3", "m2", "m1", "t1"]


def _dataset(rows) -> Dataset:
    """rows: (user, item, day_offset) triples."""
    df = pd.DataFrame(rows, columns=[Columns.User, Columns.Item, "day"])
    df[Columns.Weight] = 1.0
    df[Columns.Datetime] = df["day"].map(lambda d: BASE + pd.Timedelta(days=int(d)))
    return Dataset.construct(interactions_df=df[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]])


def _external(dataset: Dataset, internal_ids) -> list:
    return [str(item) for item in dataset.item_id_map.convert_to_external(list(internal_ids))]


def _research_graph(model: CoVisModel, dataset: Dataset) -> dict:
    """CoVisModel.neighbors is keyed by INTERNAL ids; project it into the
    external-string space that RecommenderCoVis uses, so the two are comparable."""
    graph = {}
    for item, neighbours in model.neighbors.items():
        key = _external(dataset, [item])[0]
        names = _external(dataset, [n for n, _ in neighbours])
        graph[key] = {name: weight for name, (_, weight) in zip(names, neighbours)}
    return graph


def _serving_graph(model: RecommenderCoVis) -> dict:
    return {key: dict(neighbours) for key, neighbours in model.neighbors.items()}


def _fit_research(dataset: Dataset, **kwargs) -> CoVisModel:
    model = CoVisModel(**kwargs)
    model.fit(dataset)
    return model


def _fit_serving(dataset: Dataset, top_k=100, min_cooc=2, session_weights=False) -> RecommenderCoVis:
    model = RecommenderCoVis(
        recsys_config=CoVisSettings(
            COVIS_TOP_K=top_k,
            COVIS_MIN_COOC=min_cooc,
            COVIS_SESSION_WEIGHTS=session_weights,
        ),
        model_name="covis_equivalence",
        model_version="1",
    )
    model.train(dataset)
    return model


# --- equivalence: the load-bearing claim -------------------------------------


def test_neighbour_graphs_are_identical_when_caps_are_neutral(dataset):
    """The cores build the same graph. The shared fixture's largest basket is 4
    items, well under fit_basket_size, so the research-only cap is inactive and
    nothing else should differ - same keys, same edges, same weights."""
    research = _research_graph(_fit_research(dataset, top_k=100, min_cooc=2), dataset)
    serving = _serving_graph(_fit_serving(dataset, top_k=100, min_cooc=2))
    assert research == serving
    assert research  # non-vacuous: the fixture really does produce edges


def test_min_cooc_filters_the_same_edges(dataset):
    """min_cooc handling is identical (`count >= min_cooc` on both sides)."""
    for min_cooc in (1, 2, 3):
        research = _research_graph(_fit_research(dataset, top_k=100, min_cooc=min_cooc), dataset)
        serving = _serving_graph(_fit_serving(dataset, top_k=100, min_cooc=min_cooc))
        assert research == serving, min_cooc


def test_session_scoring_is_identical_when_session_cap_is_neutral(dataset):
    """Identical session in, identical ranking AND identical float scores out -
    provided the research session_size does not truncate. This is the property
    any shared-kernel refactor has to keep."""
    research = _fit_research(dataset, top_k=100, min_cooc=1, session_size=100)
    serving = _fit_serving(dataset, top_k=100, min_cooc=1)

    online = research.recommend_for_session(SESSION, dataset, k=10, filter_viewed=True)
    served = serving.recommend("u-any", top_n=10, filter_viewed=True, history=list(SESSION))

    assert [str(item) for item, _ in online] == list(served.item_ids)
    assert [score for _, score in online] == list(served.scores)
    assert served.item_ids  # non-vacuous


# --- divergence 1: basket cap at fit time is research-only -------------------


def test_fit_basket_size_is_research_only():
    """A user with more interactions than fit_basket_size: the research model
    drops their oldest items from the basket entirely, serving keeps everything.
    Consequence: serving's C(N,2) pair count is unbounded in user history."""
    rows = [("power", f"i{n}", n) for n in range(1, 7)] + [("other", "i1", 1), ("other", "i2", 2)]
    small = _dataset(rows)

    research = _research_graph(_fit_research(small, top_k=100, min_cooc=1, fit_basket_size=3), small)
    serving = _serving_graph(_fit_serving(small, top_k=100, min_cooc=1))

    assert research != serving
    # The three most recent items (i4, i5, i6) are the whole research basket, so
    # the oldest one that is not also in another user's basket gets no edges.
    assert "i3" not in research
    assert "i3" in serving


def test_fit_basket_size_inactive_when_baskets_are_small():
    """Sanity check on the above: with the cap above every basket size, the two
    agree again. So divergence 1 is a cap, not an algorithmic difference."""
    rows = [("power", f"i{n}", n) for n in range(1, 7)] + [("other", "i1", 1), ("other", "i2", 2)]
    small = _dataset(rows)

    research = _research_graph(_fit_research(small, top_k=100, min_cooc=1, fit_basket_size=100), small)
    serving = _serving_graph(_fit_serving(small, top_k=100, min_cooc=1))

    assert research == serving


# --- divergence 2: session seed cap is research-only ------------------------


def test_session_size_is_research_only(dataset):
    """Truncating the seed changes the recency denominator n = len(seed), so the
    research model returns different SCORES for the same session. Serving applies
    no cap at all (production relies on the Redis-side 50-item cap instead)."""
    research = _fit_research(dataset, top_k=100, min_cooc=1, session_size=3)
    serving = _fit_serving(dataset, top_k=100, min_cooc=1)

    online = research.recommend_for_session(SESSION, dataset, k=10, filter_viewed=True)
    served = serving.recommend("u-any", top_n=10, filter_viewed=True, history=list(SESSION))

    assert [score for _, score in online] != list(served.scores)


# --- divergence 3: session event weights are serving-only -------------------


def test_session_event_weights_have_no_research_counterpart(dataset):
    """The "sw" variant (COVIS_SESSION_WEIGHTS) multiplies seed recency by the API
    event weight parsed out of "tour_id:weight" history entries. CoVisModel has no
    weight concept: neither its constructor nor recommend_for_session takes one."""
    assert "seed_weights" not in CoVisModel.__init__.__code__.co_varnames
    assert "weight" not in CoVisModel.__init__.__code__.co_varnames
    assert "weights" not in CoVisModel.recommend_for_session.__code__.co_varnames

    weighted_history = ["m1:1.0", "t1:3.0"]
    off = _fit_serving(dataset, min_cooc=2, session_weights=False)
    on = _fit_serving(dataset, min_cooc=2, session_weights=True)

    off_result = off.recommend("u-any", top_n=5, filter_viewed=True, history=list(weighted_history))
    on_result = on.recommend("u-any", top_n=5, filter_viewed=True, history=list(weighted_history))
    assert off_result.item_ids != on_result.item_ids  # the flag really does bite

    # ... and the research model matches the flag-OFF branch, since plain recency
    # is what it implements.
    research = _fit_research(dataset, top_k=100, min_cooc=2, session_size=100)
    plain = research.recommend_for_session(["m1", "t1"], dataset, k=5, filter_viewed=True)
    assert [str(item) for item, _ in plain] == list(off_result.item_ids)


# --- divergence 4: tie-break determinism ------------------------------------


def test_research_tie_break_is_deterministic_by_internal_id():
    """CoVisModel resolves equal co-occurrence counts by internal item id
    ascending, so its top_k truncation is reproducible run to run.

    RecommenderCoVis sorts by count only (`key=lambda x: x[1], reverse=True`), so
    ties fall back to dict insertion order, which follows set-iteration order over
    Python STRINGS and therefore varies with PYTHONHASHSEED across processes. That
    non-determinism cannot be asserted from inside one interpreter; see
    docs/DESIGN_UNIFICATION.md section 6 for the reproduction and the measured
    output. Here we pin only the deterministic side.
    """
    # "hub" co-occurs exactly once with each of cand00..cand09 (one distinct user
    # per pair), so all ten edges tie at count 1 and top_k=3 must choose.
    rows = []
    for n in range(10):
        rows += [(f"u{n}", "hub", 1), (f"u{n}", f"cand{n:02d}", 2)]
    tied = _dataset(rows)

    research = _fit_research(tied, top_k=3, min_cooc=1)
    hub_internal = int(tied.item_id_map.convert_to_internal(["hub"])[0])
    kept = [n for n, _ in research.neighbors[hub_internal]]

    assert kept == sorted(kept)  # internal-id ascending, i.e. reproducible
    assert len(kept) == 3
    # Every kept neighbour is one of the tied candidates, all at weight 1.0.
    assert all(weight == 1.0 for _, weight in research.neighbors[hub_internal])
    assert all(name.startswith("cand") for name in _external(tied, kept))
