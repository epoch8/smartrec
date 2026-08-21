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
| A new servable model | **L2** `recommenders/recommender_<name>.py` | Subclass `RecommenderModel`; implement `train`, `recommend`, `save_model_triton`, `calc_metrics`. Add to the `serving/model.py` name ladder, to `--models` in the trainer, to the Makefile runtime-env targets, and to `app/src/settings.py`. |
| A knob for an existing model | **L0** a field on the matching `*Settings`, with a default | Adding a field is safe (old pickles skip validation); removing or renaming one needs `_migrate_legacy_flat_fields`. |
| A new user segment / routing outcome | **L0** a new `Strategy` member, append-only | Coordinate with `api/docs/DEBUG_INFO_CODEC.md` — the numeric ids there are a published contract. |
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
- One concept, one name. `session_weight` (a per-source multiplier in fusion) and
  `COVIS_SESSION_WEIGHTS` (a per-seed event multiplier) are unrelated — do not add a third.

## 5. Frozen invariants

Each of these breaks production silently or globally. None is enforced by a test that runs on
every change; they are enforced by this list.

**Pickle**

1. The outer `Recommender*` class is *not* in the pickle (Triton picks it by artifact name), but
   **every class reachable as a value inside `__dict__` is**, reconstructed by fully-qualified path:
   `smartrec_lib.model.{ModelSetSettings, ALSSettings, PopularSettings, CoVisSettings, BlendSettings,
   EASESettings, CommonRecommenderSettings, Strategy}` and
   `smartrec_lib.recommenders.recommender_covis.RecommenderCoVis`
   (it lives inside the `als_covis_youtravel` artifact as `RecommenderALS.covis`).
   Moving or renaming any of these makes existing artifacts unloadable on the next Triton poll —
   without a deploy.
2. Instance attribute names are frozen per class. A rename is silent: the fresh `__init__` default
   survives `__dict__.update()`. `RecommenderALS.covis = None` means `als_covis_youtravel` quietly
   degrades to the old item-sim path with no error.
3. Every `hasattr`/`getattr` shim in `recommender_als.py` and `recommender_covis.py` marks an
   artifact shape that still exists in S3. Removing one produces a 100% request-time error rate on
   that model, with a clean load.
4. `recsys_config` is pickled inside the artifact, and **three shapes of it exist in S3**: flat
   (`ALSSettings` with `POPULARITY_*`/`COVIS_*`/`BLEND_*` fields), nested (`ALSSettings` with
   `popular`/`covis`/`blend` sub-configs) and current (`ModelSetSettings`). Two hooks handle that,
   and they cover *field shape* only:
   - `ALSSettings.__setstate__` rebuilds sub-configs from flat fields on unpickle;
   - `RecommenderALS.model_set` normalises either legacy shape into a `ModelSetSettings`.

   Read composition through `self.model_set`, never off `self.recsys_config` — the latter is an
   `ALSSettings` on every artifact currently in the buckets, and the only composition field the
   serving path touches is `blend`, so getting this wrong does not raise. It silently fuses with
   default weights. Changing the base class of a settings object is covered by neither hook and
   needs a `__reduce__` proven against a real prod pickle.
5. Version directories must be `str.isdigit()` — otherwise they are invisible to both loading and
   retention cleanup.

**Wire**

6. Serving calls `recommend(user_ids=, items_to_recommend=, top_n=, filter_viewed=, history=)` by
   **keyword**, and reads `.item_ids` / `.scores` / `.strategy` off the result. Argument names are
   part of the contract.
7. `serving/config.pbtxt` and `serving/model.py` in this tree *are* the deployed files — the trainer
   overwrites them in S3 on every run. A bad import in `serving/model.py` breaks every model in the
   bucket, including ones that were not retrained.
8. `Strategy` values are a published vocabulary mirrored by numeric id in
   `api/docs/DEBUG_INFO_CODEC.md`. Append-only; ids are never renumbered.
   `MODEL_HOT_AND_COLD_USERS`, `MODEL_ALS_COVIS_BLEND` and `MODEL_COVIS_SESSION` have no emitter
   and stay anyway.
   **A strategy names the user segment and the signal used, never the algorithm.** Which artifact
   answered — and therefore which algorithm — is already carried by `model_name` next to it. Adding
   a strategy string for a new internal algorithm silently breaks every consumer watching for the
   segment while telling them nothing new: `als_covis_blend`/`covis_session` did exactly that, and
   the two ALS artifacts now both report `model_realtime_hot_users` / `model_realtime_warm_users`.
   A/B readouts split on `model_name`.
   Watch the mixed convention: some paths return `Strategy.X`, others `Strategy.X.value`. Pydantic
   coerces both today; do not rely on it in new code — always return `.value`.
9. Artifact names (`als_youtravel`, `popular_youtravel`, `covis_youtravel`, `als_covis_youtravel`)
   are consumed as a *public query parameter* (`model_name` on the feed endpoint), as API defaults,
   and as Makefile runtime-env paths. Never rename, never reuse a name for a different model.
10. `serving/model.py` resolves the class from the artifact name by an **ordered substring ladder**.
    Most specific first: `als_covis` must match before `covis`. Getting this wrong once already
    served empty feeds.

**Deploy ordering**

11. A **code** change requires runtime repack *and* retrain, in that order: upload
    `runtime_env.tgz` first, then bump the trainer image tag and retrain. A **config/data** change
    requires only a retrain. Reversing the order raises `ImportError` on model reload for the whole
    bucket.
12. Every artifact name needs its own `runtime_envs/runtime_env.tgz` copy in the Makefile upload
    targets. A new model with no runtime env cannot load.
13. `--models` tokens are baked into the Airflow DAG in `youtravel-etl-yc`. An unknown token does
    not fail the job — it logs and continues, so the artifact silently stops being refreshed while
    Triton keeps serving a stale pickle indefinitely.

## 6. Known deviations from this document

Recorded honestly so they are fixed rather than copied. Deep analysis lives in
`smartrec-lib/docs/DESIGN_UNIFICATION.md`.

Fixed by the refactor that introduced this document: the L2 → L3 import leak (`fusion.py` and
`constraints.py` moved down to L1, so `policy` no longer loads inside Triton); the third copy of
co-occurrence (`covis_kernel.py` → `kernels/cooccurrence.py`, with `research/covis.py` ported onto
it); the `model.py` / `models/` collision (`models/` and `policy/` → `research/`, leaving only the
two forced `model.py` files); and the `recommenders/` package import cycle.

Fixed on 2026-08-22: composition moved off `ALSSettings` into `ModelSetSettings`, so the config of
one ranker no longer declares who serves cold users and how two rankers are fused. `ALSSettings` is
a leaf again. The cost is the shim in §5.4, removable after two retrain cycles.

Still open:

1. **Duplicated logic.** `_parse_history` (covis) vs `_parse_weighted_history` (als) differ in bytes
   handling; a 19-line block is byte-identical twice inside `recommender_als.py`; `calc_metrics`
   repeats the same CV preset in three recommenders with a metric set that disagrees with
   `evaluation/warm_cv.py`; `evaluation/next_item.py` reimplements rectools' `PopularModel` by hand
   with a hard-coded 14-day window.
2. **Six different id-conversion policies** across the package (`str()` vs native, strict vs
   lenient). This is why the equivalence test has to reproject an entire graph before comparing.
3. **`RECOMMENDER_DAYS_THRESHOLD`** is declared on four settings classes and read zero times inside
   the library — only the trainer reads it.
4. **`serving/model.py` reaches into private ALS methods** (`_ensure_lookup_caches`,
   `_ensure_user_item_matrix_binary`) behind a `hasattr` guard.

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
