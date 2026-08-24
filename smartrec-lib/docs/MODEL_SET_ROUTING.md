# Model sets and routing

**Status: DEFERRED (2026-08-24).** The owner chose to ship the simple
role-field shape (`main`/`fallback`/`session`, PR #12) instead - the routing
table below was judged more machinery than the current two artifacts need.
This document stays as the worked-out design for the day a trigger appears:
a second add-on in one artifact, a second project, or a scenario the role
fields cannot express. Nothing below is implemented.

Cost note recorded at deferral time: adopting this AFTER the first model-set
retrain will require one more legacy config hook (the role-shaped
`ModelSetSettings` will by then exist in pickles). That is the accepted price
of shipping now.

## Goals

Stated requirements, in the owner's words:

- R1. Different main models; a fallback when needed; add-ons (realtime, covis,
  and future ones) - plural, not a single hardcoded slot.
- R2. The framework must transfer to other projects.
- R3. Adding models must be easy; old artifacts must keep serving.
- R4. It must be explicit which model runs in which scenario.
- R5. A convenient layer over rectools; combining models for different purposes
  must be easy.

Constraints carried over from production experience:

- C1. No silent degradation: the pickle contract is frozen, strategies name
  segments, never algorithms.
- C2. dev == prod; everything is verified on the stand before prod.
- C3. The API contract does not change; no torch in the serving runtime; feed
  mixing stays in SmartFeed.

## Decision log

| Date | Decision |
|---|---|
| 2026-08-23 | **Leaves stay `Recommender*` classes.** The rectools-native leaf replacement (generic `model_from_config` members instead of `RecommenderALS` etc.) is explicitly rejected for now: prod runs on these classes and they are not to be rewritten. A generic `RecommenderRectools` member may be ADDED later for new models; existing classes are never removed. |
| 2026-08-23 | **Routing is data, in Python.** No YAML, no config files: routing lives in `app/src/settings.py` as typed pydantic objects. A typo is an import error, not a wrong feed. |
| 2026-08-23 | **Labels belong to routes, not models.** A label names the visitor segment and the signal used. One model may appear in several routes under different labels; which algorithm answered is already carried by `model_name`. |
| 2026-08-23 | **Fall-through on unused signal.** A route whose defining signal was not actually used by any contributing member does not bind; the request falls to the next applicable route. This makes label honesty a property of the interpreter instead of a hidden if. |
| 2026-08-23 | **Empty results do not fall through.** An empty ranking is returned with the route's label. Filling an empty feed is the API layer's job (SmartFeed backfills from elastic); the artifact silently switching segments would hide signal problems. This preserves current prod semantics exactly. |
| 2026-08-23 | **Fusion is rank-based (RRF) only.** Raw scores never cross a member boundary - co-occurrence counts, dot products and popularity counters are not comparable (the documented failure of the old `RecommenderOrchestrator`). |

## The shape

```python
# app/src/settings.py - the real thing, not pseudocode
recsys_config_als_covis_prod = ModelSetSettings(
    models={
        "als":     ALSSettings(ALS_ITERATIONS=40, ALS_FACTORS=100, ...),
        "covis":   CoVisSettings(COVIS_TOP_K=100, COVIS_MIN_COOC=2, ...),
        "popular": PopularSettings(POPULARITY_STRATEGY="n_users", ...),
    },
    routing=[
        Route(
            when=(Signal.KNOWN, Signal.SESSION),
            use=["als", "covis"],
            fuse=Fuse(weights={"als": 1.0, "covis": 1.0}, rrf_k=60),
            label="model_realtime_hot_users",
        ),
        Route(when=(Signal.KNOWN,), use=["als"], label="model_hot_users"),
        Route(
            when=(Signal.SESSION,),
            use=["covis"],
            on_empty=["als"],
            label="model_realtime_warm_users",
        ),
        Route(when=(), use=["popular"], label="model_cold_users"),
    ],
)
```

`als_youtravel` is the same table with `covis` removed from `models` and every
`use` list - ALS then serves the session scenarios itself (item-similarity from
its own factors), under the same labels. Same segments, same labels, different
algorithm; the A/B reads by `model_name`.

Everything in the table is L0 (`smartrec_lib/model.py`): `ModelSetSettings`,
`Route`, `Fuse`, `Signal`. All pydantic, all pickled into the artifact.

## Member protocol

A member is a `Recommender*` class. It knows its own algorithm and nothing
else: no siblings, no set config, no labels.

```python
class RecommenderModel:                    # the existing base, formalised
    # capabilities - checked at set construction, not at request time
    scores_users: bool        # can rank a user by their training profile
    scores_sessions: bool     # can rank from a live history
    needs_nothing: bool       # answers anyone (popular)

    def train(self, dataset) -> None
    def can_serve(self, user_id) -> bool
    def score_user(self, user_id, n, candidates, filter_viewed) -> Ranking
    def score_session(self, history, n, candidates) -> tuple[Ranking, bool]
    #                                    the bool: "did I actually use it" ----^
```

The entry point is implied by the scenario: routes whose `when` includes
`KNOWN` call `score_user`; routes gated on `SESSION` call `score_session`. A
route with both fuses one of each per member according to what the member
offers. Members never emit labels; the `session_used` fact is what lets the
interpreter keep labels honest.

Current members and their capabilities:

| Member | scores_users | scores_sessions | needs_nothing |
|---|---|---|---|
| RecommenderALS | yes | yes (item-sim from own factors) | no |
| RecommenderCoVis | no | yes | no |
| RecommenderPopular | no | no | yes |
| RecommenderEASE | yes | no | no |

## Routing semantics

For a request, walk routes top-down:

1. **Applicable**: every signal in `when` holds. `KNOWN` = at least one member
   in `use` returns `can_serve(user)`. `SESSION` = a non-empty history arrived.
   Empty `when` always holds (the terminal route).
2. **Execute**: call the implied entry point on each member in `use`; if more
   than one produced a ranking, fuse by RRF with the route's weights. If the
   result is empty and `on_empty` is set, run the same procedure over the
   `on_empty` members - same segment, same label, different scorer.
3. **Bind**: the route binds if every signal in `when` was actually used by at
   least one contributing member. A `(KNOWN, SESSION)` route where the history
   turned out unusable does not bind - the request falls through, typically to
   the plain `(KNOWN,)` route, and gets the honest label.
4. An empty ranking from a bound route is returned as-is with the route's
   label (see decision log - emptiness is the API layer's problem).

`Signal` is a closed enum: `KNOWN`, `SESSION`, and reserved `CONTEXT` (request
filters as a signal - the old orchestrator's segment-popularity case; not
implemented until needed). Adding a signal is a deliberate contract change,
never a config-side invention. This is what keeps the table from becoming a DSL.

### Validation at construction (import time, not request time)

- The last route has empty `when`; no other route does.
- Every name in `use`, `on_empty` and `fuse.weights` exists in `models`.
- `fuse.weights` keys are exactly the `use` list when present; `fuse` present
  iff `len(use) > 1`.
- Capability check: a route gated on `SESSION` must include at least one
  member with `scores_sessions`; a route gated on `KNOWN` at least one with
  `scores_users`; the terminal route at least one with `needs_nothing`.
- Unreachable routes (a route after an equal-or-weaker `when`) are an error.
- Every label must be a published `Strategy` value - checked in THIS project's
  tests, not in the library: labels are plain strings in the lib so other
  projects can bring their own vocabulary (R2).

All of this fails the import of `settings.py`, i.e. the trainer run and the
test suite - never a live request.

## Behaviour parity with current prod

The table above must reproduce today's behaviour bit-for-bit before anything
else changes. The four scenarios, per artifact:

| Scenario | als_covis_youtravel | als_youtravel | Label |
|---|---|---|---|
| known + session | RRF(als, covis) | ALS enriched by own item-sim | model_realtime_hot_users |
| known | ALS profile | ALS profile | model_hot_users |
| unknown + session | covis, else ALS item-sim | ALS item-sim | model_realtime_warm_users |
| unknown + session, history unusable | falls through -> empty from session scorers -> route binds with empty | same | model_realtime_warm_users (empty result) |
| known + session, history unusable | falls through to known | same | model_hot_users |
| cold | popular | popular | model_cold_users |

The equivalence test extends the existing legacy-artifact test: a legacy pickle
adapted into a routed set answers all scenarios with identical items, scores
and labels.

## Back-compat and artifacts

- The artifact format does not change in this step: still one dill of the
  composite `__dict__`. The three existing legacy hooks stay
  (`ALSSettings.__setstate__`, `ModelSetSettings.from_legacy_als_settings`,
  `RecommenderModelSet.from_legacy_als_state`); the legacy adapters now build
  the default routing table for old artifacts, so there is exactly one
  interpreter in the library.
- `from_legacy_als_settings` maps the role fields of the parked branch and the
  pre-split shapes into `models` + default `routing`.
- **Gate before any deploy**: load a real artifact pulled from S3 (not a
  synthetic fixture) through the adapter and verify all scenarios. A wrongly
  assembled member raises nothing - it just serves worse.
- Manifest-based artifacts (manifest.json with schema version, config, git
  sha, metrics + per-member state files + dataset stored once) remain the
  planned next step, orthogonal to routing. Not in this change.

## Extension recipes

- **New model**: a `Recommender*` class implementing the protocol (three
  methods + capability flags) + one entry in `MODEL_FOR_SETTINGS` + its
  settings type in `RankerSettings`. No existing model, no composite, no
  routing change.
- **New scenario / add-on**: a `Route` row. No code.
- **New rectools model without a new class** (future, additive): one generic
  `RecommenderRectools(cls=...)` member via `model_from_config` - existing
  classes untouched.
- **New project** (R2): same library, own `models`/`routing`/labels in its own
  settings. The `Signal` vocabulary is universal (the cold-start matrix);
  labels are project vocabulary.
- **Per-member datasets** (future; measured need: CoVis on clicks+views, ALS
  on clicks only, +3.4% map): `train(datasets={"default": ..., "with_views":
  ...})` with a `dataset:` key on the member config. Fusion is on external ids
  so it survives differing item spaces; `filter_viewed` and `can_serve` always
  evaluate against `default`.

## Observability

Per request, one log line from the interpreter: matched route index, signals
held/used, members consulted with result counts, fused weights, final label.
The strategy label on the wire is unchanged (published contract,
DEBUG_INFO_CODEC.md); richer mix composition stays in logs until there is a
consumer for it.

## Rollout

1. Implement `Route`/`Fuse`/`Signal` + interpreter + validation in the parked
   branch, replacing the role fields (nothing is pickled with them yet).
   Adapt legacy loaders to emit default routing. Parity tests.
2. The S3 real-artifact gate.
3. Runtime repack, then retrain (§5.11 ordering), verify all four scenarios on
   the stand - same checklist as the 2026-08-21 verification.
4. Only after that: manifest artifacts, `RecommenderRectools`, per-member
   datasets - each as its own reviewed step.

## Non-goals

- No YAML or external config files - configs are typed Python.
- No replacement of `Recommender*` leaf classes.
- No learned re-ranker (LTR), no torch in the serving runtime.
- No feed mixing in the library - that is SmartFeed's job.
- No new `Strategy` values and no API contract changes.
