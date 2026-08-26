# CLAUDE.md — smartrec

Architecture contract for this repository. Read this before adding a file, renaming a symbol,
or moving code. It exists because the library serves two masters with opposite change costs,
and mixing them is how the structure erodes.

- `smartrec-lib/` — the models, the artifact format, the Triton entrypoint. **This document is about it.**
- `smartrec-client/` — the thin gRPC client the feed API uses. Independent, no imports from the lib.

## 1. The system in one paragraph

A nightly Airflow job runs `app/experiments/model_training_hybrid.py` in the *parent* repo. It pulls
interactions from ClickHouse, builds a `rectools.Dataset`, constructs a `Recommender*` class per
`--models` token, calls `.train()`, then `.save_model_triton()`, which dills the instance `__dict__`
into `s3://youtravel-recsys[-dev]/models/<name>/<version>/model.pkl` **and** overwrites
`config.pbtxt` + `serving/model.py` in that bucket from the working tree. Triton polls the bucket,
unpickles the artifact inside a packed conda env (`runtime_env.tgz`), and answers `recommend()`
calls from the feed API. Nothing else consumes this library.

Two consequences drive every rule below:

1. **The pickle is a bare `dict`, not an object.** Load is `cls(); instance.__dict__.update(state)`.
   A renamed attribute does not raise — it silently reverts to the `__init__` default and the model
   serves degraded results with no error anywhere.
2. **`recommend()` ships in `runtime_env.tgz`, not in the pickle.** Changing library *code* without
   repacking the runtime changes nothing in production; changing it in the wrong order breaks every
   model in the bucket at once.

## 2. Layers

Imports flow strictly downward. There are exactly four layers, and every module belongs to one.

| Layer | Contents | May import | Change cost |
|---|---|---|---|
| **L0 contract** | `model.py`: `RecomItems`, `Strategy`, all `*Settings` | nothing (stdlib + pydantic) | **Frozen.** Pickled into every artifact and consumed by the API. |
| **L1 kernels** | pure algorithms: co-occurrence, RRF, share cap | L0 only | **Free.** Never pickled, no I/O, no rectools. Move and rename at will. |
| **L2 serving** | `recommenders/*`, artifact I/O, `serving/model.py` | L0, L1 | **Frozen paths, free bodies.** Module+class FQNs are load-bearing; logic inside them is not. |
| **L3 research** | rectools-native models, policy, evaluation harnesses | L0, L1 | **Free.** Must never appear on the serving import path. |

**Inside L2 there is one more direction, and it is just as strict: the composite may know its
members; a member may never know the composite.** `RecommenderModelSet` owns the routing across
visitor segments, the fusion of two rankings, and the entire `Strategy` vocabulary. A member
(`RecommenderALS`, `RecommenderPopular`, `RecommenderCoVis`, `RecommenderEASE`) knows only its own
algorithm: it answers `can_serve(user)` and returns candidates. It must not import a sibling, hold
one as a field, read the set's config, or name a segment.

This is not stylistic. Until 2026-08-22 `RecommenderALS` *was* the whole feed - it held a
PopularModel for cold users, an optional `RecommenderCoVis` for sessions, the blend weights and the
four-way routing - so "add a model" meant editing ALS, and `als_covis_youtravel` read as "the ALS
artifact with extras" rather than as a set of models behind one name. Adding a member now touches
`recommender_model_set.py` and `ModelSetSettings`, and no existing model at all.

**L2 must not import L3.** This is the rule that is currently violated (§6), and the violation is
not theoretical: an `ImportError` anywhere in `policy/` takes down inference for *all* models,
including ones that have nothing to do with policy.

Verify the rule at any time:

```bash
python -c "
import sys; import smartrec_lib.recommenders
bad = [m for m in sys.modules if m.startswith(('smartrec_lib.research','smartrec_lib.evaluation'))]
print('research modules on serving path:', bad or 'none')"
```

## 3. Where does a new file go?

Answer it *before* writing the file. The table exists because a shared co-visitation kernel was
once dropped at the package root with no declared home — not because extracting it was wrong, but
because there was nowhere for it to go.

| What you are adding | Where it goes | Constraints |
|---|---|---|
| A pure function/algorithm shared by two callers | **L1** `kernels/<domain>.py` | No rectools, no pandas, no I/O, no settings objects. Takes primitives, returns primitives. Fully unit-testable without a `Dataset`. |
| A new ranker to put **inside** an existing artifact (a warm ranker, a content model for cold tours) | **L2** `recommenders/recommender_<name>.py`, plus a member field on `ModelSetSettings` and a branch in `RecommenderModelSet` | This is the common case and it touches **no existing model**. Implement `train`, `recommend`, `can_serve`, `calc_metrics`. Do not give it knowledge of its siblings or of `Strategy`. |
| A whole new servable artifact (its own name in S3) | **L2** as above **and** `--models` in the trainer, the Makefile runtime-env targets, `app/src/settings.py`, the DAG | Almost always the wrong choice: a second artifact doubles the retrain, the bucket and the runtime env. Prefer a member of an existing set. |
| A knob for an existing model | **L0** a field on the matching `*Settings`, with a default | Adding a field is safe (old pickles skip validation); removing or renaming one needs `_migrate_legacy_flat_fields`. Put it on the LEAF config of the model it configures, never on `ModelSetSettings` unless it is about composition. |
| A new user segment / routing outcome | **L0** a new `Strategy` member, append-only, **and** a branch in `RecommenderModelSet.recommend` | Coordinate with `api/docs/DEBUG_INFO_CODEC.md` — the numeric ids there are a published contract. The routing table is the only place a segment is decided; no model may name one. |
| An offline experiment, metric, or protocol | **L3** `research/` or `evaluation/` | May import rectools freely. Must not be imported by L2. |
| A one-off analysis | Not here. The parent repo's `app/experiments/` | The library is not a scratchpad. |

If a change does not fit any row, the layer model is wrong and this document should be amended
first — do not resolve it by dropping a module at the package root.

## 4. Naming

- **No new module named `model.py` or package named `models/`.** There used to be four
  (`model.py`, `models/covis.py`, `policy/model.py`, `serving/model.py`), and
  `from smartrec_lib.model import CoVisSettings` vs `from smartrec_lib.models import CoVisModel`
  differed by one character while meaning opposite things. Two remain, and both are *forced*:
  - `smartrec_lib/model.py` — its FQN is baked into every pickle in S3 (`smartrec_lib.model.ALSSettings`).
    It is the **contract** module despite the name. Do not add model classes to it.
  - `smartrec_lib/serving/model.py` — Triton's python backend loads a file by that exact name.
- Modules are named for what they contain (`cooccurrence.py`, `fusion.py`), not for their layer
  (`base.py`, `utils.py`, `common.py`, `core.py`, `helpers.py` are all banned).
- One concept, one name. `session_weight` (a per-source multiplier in the co-occurrence kernel) and
  `COVIS_SESSION_WEIGHTS` (a per-seed event multiplier) are unrelated — do not add a third. This is
  why the fusion weight is `BlendSettings.REALTIME_WEIGHT` and not `SESSION_WEIGHT`.
- **The signal is "session"; the role, the weight and the strategy are "realtime".** Live events
  arriving from Redis are a session (`history=`, `has_session`, `session_used`); the member that
  consumes them is `ModelSetSettings.realtime` / `RecommenderModelSet.realtime`, its weight is
  `REALTIME_WEIGHT`, and what it produces is `model_realtime_*`. The role was briefly called
  `session`, which left the config and the published strategy vocabulary disagreeing about the same
  thing.

## 5. Frozen invariants

Each of these breaks production silently or globally. None is enforced by a test that runs on
every change; they are enforced by this list.

**Pickle**

1. The outer class is *not* in the pickle — `serving/model.py` picks it — but **every class
   reachable as a value inside `__dict__` is**, reconstructed by fully-qualified path:
   `smartrec_lib.model.{ModelSetSettings, ALSSettings, PopularSettings, CoVisSettings, BlendSettings,
   EASESettings, CommonRecommenderSettings, Strategy}` and, as members of a set,
   `smartrec_lib.recommenders.{recommender_als.RecommenderALS, recommender_covis.RecommenderCoVis,
   recommender_popular.RecommenderPopular}`. Moving or renaming any of these makes existing artifacts
   unloadable on the next Triton poll — without a deploy.
2. Instance attribute names are frozen per class. A rename is silent: the fresh `__init__` default
   survives `__dict__.update()`. `RecommenderModelSet.realtime = None` means `als_covis_youtravel`
   quietly serves without its realtime layer, and nothing errors.
3. Every `hasattr`/`getattr` shim in the recommenders marks an artifact shape that still exists in
   S3. Removing one produces a 100% request-time error rate on that model, with a clean load.
4. **Two generations of artifact exist in the buckets, and both must serve.** The old one is a
   `RecommenderALS` `__dict__` that carried every member itself (`model_cold_users`, `covis`) plus a
   flat or nested `ALSSettings`. The current one is a `RecommenderModelSet` `__dict__` with `main` /
   `fallback` / `realtime`. Three hooks, covering *shape* only:
   - `ALSSettings.__setstate__` — rebuilds sub-configs from flat fields on unpickle;
   - `ModelSetSettings.from_legacy_als_settings` — either legacy config shape into a model set config;
   - `RecommenderModelSet.from_legacy_als_state` — a legacy ARTIFACT into a model set, member by
     member, so there is exactly **one** routing path in the library rather than a second copy kept
     alive for old pickles.

   The equivalence is pinned by `test_legacy_artifact_loads_and_recommends_identically`, and it has
   to be: a member assigned from the wrong key raises nothing and just serves worse. Delete all
   three once both artifacts have been retrained twice. Changing the *base class* of a settings
   object is covered by none of them and needs a `__reduce__` proven against a real prod pickle.
5. Version directories must be `str.isdigit()` — otherwise they are invisible to both loading and
   retention cleanup.

**Wire**

6. Serving calls `recommend(user_ids=, items_to_recommend=, top_n=, filter_viewed=, history=)` by
   **keyword**, and reads `.item_ids` / `.scores` / `.strategy` off the result. Argument names are
   part of the contract.
7. `serving/config.pbtxt` and `serving/model.py` in this tree *are* the deployed files — the trainer
   overwrites them in S3 on every run. A bad import in `serving/model.py` breaks every model in the
   bucket, including ones that were not retrained.
8. `Strategy` declares what a model MAY emit; `api/docs/DEBUG_INFO_CODEC.md` owns the published
   vocabulary and its numeric ids, which are append-only and never renumbered. The two are NOT the
   same set: the codec has rows with no member here (`popular`/8, `random`/9, and since 2026-08-25
   `model_hot_and_cold_users`/4, `als_covis_blend`/11, `covis_session`/12, dropped from the enum
   because nothing had emitted them for weeks and a member no model can produce reads as a live
   option). **Never remove a row from the codec** - the consumer decodes historical records by id,
   and our API only ever passes the string through, so the codec is the sole authority for what an
   old id meant. Removing a MEMBER is safe and was verified so: no pickle in either bucket contains
   a `Strategy` reference (checked in the pickle bytecode of the prod artifact), nothing does
   `Strategy(value)` and nothing iterates the enum.
   **A strategy names the user segment and the signal used, never the algorithm.** Which artifact
   answered — and therefore which algorithm — is already carried by `model_name` next to it. Adding
   a strategy string for a new internal algorithm silently breaks every consumer watching for the
   segment while telling them nothing new: `als_covis_blend`/`covis_session` did exactly that, and
   the two ALS artifacts now both report `model_realtime_hot_users` / `model_realtime_warm_users`.
   A/B readouts split on `model_name`.
   **Only `RecommenderModelSet.recommend` sets a strategy.** Members report facts - "I used the
   session", "nothing in the candidate list was scorable" - and the set names the segment from them.
   A member that labels its own answer will eventually lie: a hot user whose session seeds are all
   unknown to the models gets a plain ALS ranking, and calling that `model_realtime_hot_users`
   silently corrupts every A/B split on strategy. That regression was caught by
   `test_hot_user_ignores_history` during the 2026-08-22 refactor, which is why the session methods
   return `(RecomItems, session_used)` instead of a strategy.
   Always emit `.value`, never the enum member.
9. Artifact names (`als_youtravel`, `popular_youtravel`, `covis_youtravel`, `als_covis_youtravel`)
   are consumed as a *public query parameter* (`model_name` on the feed endpoint), as API defaults,
   and as Makefile runtime-env paths. Never rename, never reuse a name for a different model.
10. `serving/model.py` resolves the class **from the pickle's shape first, and only then from the
    artifact name.** The name cannot distinguish the two generations: `als_covis_youtravel` is a
    `RecommenderALS` dict in the buckets today and a `RecommenderModelSet` dict after the next
    retrain, under the same name (§5.9 forbids renaming it). Members present → model set; a
    `model_hot_users` key → legacy artifact, adapted; otherwise the name ladder.
    The ladder stays an **ordered** elif chain, most specific first: `als_covis` must match before
    `covis`. Getting that wrong once already served empty feeds. Pinned by `test_serving_loader.py`,
    which stubs Triton's backend utils so the real `_load_model` can be tested off-cluster.

**Deploy ordering**

11. A **code** change requires runtime repack *and* retrain, in that order: upload
    `runtime_env.tgz` first, then bump the trainer image tag and retrain. A **config/data** change
    requires only a retrain. Reversing the order raises `ImportError` on model reload for the whole
    bucket.
12. Every artifact name needs its own `runtime_envs/runtime_env.tgz` copy in the Makefile upload
    targets. A new model with no runtime env cannot load.
13. `--models` tokens are baked into the Airflow DAG in `youtravel-etl-yc`. An unknown token, or an
    unknown `--stand`, now **exits non-zero** — until 2026-08-22 it logged and continued, so a typo
    in the DAG left the artifact silently un-refreshed while Triton served a stale pickle
    indefinitely, and Airflow reported success. Keep it that way: a bad input must fail the task.

14. The trainer keys a person by their **account id** whenever it has ever seen them signed in, and
    by their `guest_id` otherwise. The feed API must ask Triton for that same id — it receives only
    one id per request, so a previously-signed-in visitor browsing anonymously arrives as a
    `guest_id` whose embedding lives under the account. `EventStorageService.resolve_canonical_user_id`
    is the API-side half of this rule; the two link sources differ (180 days of ClickHouse vs what
    our own ingestion wrote to Redis), so they narrow the gap rather than close it. Change one side
    and you must change the other, or hot users quietly fall to the warm path.

## 6. Known deviations from this document

Recorded honestly so they are fixed rather than copied. Deep analysis lives in
`smartrec-lib/docs/DESIGN_UNIFICATION.md`.

Fixed by the refactor that introduced this document: the L2 → L3 import leak (`fusion.py` and
`constraints.py` moved down to L1, so `policy` no longer loads inside Triton); the third copy of
co-occurrence (`covis_kernel.py` → `kernels/cooccurrence.py`, with `research/covis.py` ported onto
it); the `model.py` / `models/` collision (`models/` and `policy/` → `research/`, leaving only the
two forced `model.py` files); and the `recommenders/` package import cycle.

Fixed on 2026-08-22, in two steps that belong together. First the config: composition moved off
`ALSSettings` into `ModelSetSettings`, so one ranker's config no longer declares who serves cold
users. Then the code, which was the half that mattered: `RecommenderModelSet` took over routing,
fusion and the `Strategy` vocabulary, and `RecommenderALS` became a member that knows nothing about
its siblings or about segments. The cost is the three legacy hooks in §5.4, removable after two
retrain cycles.

Still open:

1. **Duplicated logic.** `_parse_history` (covis) vs `_parse_weighted_history` (als) differ in bytes
   handling; a 19-line block is byte-identical twice inside `recommender_als.py`; `calc_metrics`
   repeats the same CV preset in three recommenders with a metric set that disagrees with
   `evaluation/warm_cv.py`; `evaluation/next_item.py` reimplements rectools' `PopularModel` by hand
   with a hard-coded 14-day window. That CV preset is the obvious next extraction now that
   `RecommenderModelSet.calc_metrics` just fans out to its members.
2. **Six different id-conversion policies** across the package (`str()` vs native, strict vs
   lenient). This is why the equivalence test has to reproject an entire graph before comparing.
3. **`RECOMMENDER_DAYS_THRESHOLD`** is declared on four settings classes and read zero times inside
   the library — only the trainer reads it.
4. **A `RecomItems.strategy` of `None` is legal on the wire between a member and the set**, because
   members no longer label their own answers. Nothing outside the library ever sees it - the set
   always stamps a value - but the field is still typed `Optional[str]`, so a future member returned
   straight to Triton would emit a null strategy. Tighten it when the legacy hooks go.

## 7. Checklist

Before opening a PR that touches `smartrec-lib`:

- [ ] Every new module has a layer from §2, and its imports only go downward.
- [ ] No new `model.py` / `models/` / `utils.py` / `core.py`.
- [ ] No renamed class, module, or instance attribute from §5 — or an explicit migration proven
      against a real artifact pulled from S3.
- [ ] New settings fields have defaults; removed ones are handled in `_migrate_legacy_flat_fields`.
- [ ] New `Strategy` members are appended and mirrored in `DEBUG_INFO_CODEC.md`.
- [ ] Ran the L2→L3 check from §2.
- [ ] Formatted and linted from `app/smartrec/` (there is no Makefile here — CI hard-fails on the
      flake8 gate):
      `uv run black smartrec-lib/smartrec_lib smartrec-lib/tests`,
      `uv run flake8 smartrec-lib --count --select=E9,F63,F7,F82 --show-source --ignore=F821`,
      `uv run pytest smartrec-lib/tests`.
- [ ] Stated in the PR whether this is a **code** change (needs runtime repack + retrain) or a
      **config/data** change (retrain only), per §5.11.

## 8. Tests

Assert behaviour, not tie order. Several tests pin exact result lists over items with identical
scores; on a fixture where 98 of 99 items tie at one user, that pins an arbitrary permutation and
turns a dependency bump into a red build. If ranks tie, assert set membership and that scores are
non-increasing.

Determinism is a *choice per call site*, not a global: the serving co-occurrence graph deliberately
keeps hash-order tie-breaking so existing S3 graphs stay reproducible, while research code sorts
deterministically. Whichever a new call site picks, it must pass the flag explicitly and say why.
