# Unifying the two model hierarchies in smartrec-lib

Status: **decided and implemented in part.** Option B (one algorithm, two thin
shells) was taken: the co-visitation kernel now lives in
`kernels/cooccurrence.py` and both callers go through it. Alongside it the
library gained the layer model in [../../CLAUDE.md](../../CLAUDE.md), which is
where the rules live now; `models/` and `policy/` became `research/`, and
`policy/fusion.py` and `policy/constraints.py` moved to `kernels/` so that the
research layer is no longer inside the Triton import closure. Option C
(promoting `PolicyModel` to a serving citizen) remains **not** taken, for the
structural reason in section 1.6.

Paths and line numbers below are as they were when this analysis was written -
before those moves - and are kept as the record of the reasoning. Written on
`chore/drop-dead-code`, on top of the three cleanup commits (orchestrator
deleted, `ALSSettings` given nested composition, LightFM/Random/incremental
training removed).

The complaint this document answers: composition is expressed twice, in two
different idioms, and it is not obvious whether that can be collapsed into one.
The short answer is yes for the *algorithm*, no for the *policy* - and the
reason is a structural gap in `PolicyModel`, not an effort estimate. Details
below.

Everything marked "verified" was executed against the code on this branch.
Everything marked "assumed" was not; those are flagged individually.

---

## 1. The map

### 1.1 Serving hierarchy (`recommenders/`)

Duck-typed base class, pickled `__dict__`, Triton-shaped `recommend`.

| Class | File | Artifact in S3 | Strategies emitted | Composes |
|---|---|---|---|---|
| `RecommenderModel` | `recommenders/base.py:17` | - | - | - |
| `RecommenderALS` | `recommenders/recommender_als.py:35` | `als_youtravel`, `als_covis_youtravel` | `model_hot_users`, `model_cold_users`, `model_realtime_hot_users`, `model_realtime_warm_users`, `als_covis_blend`, `covis_session`, `no_strategy_...` | `ImplicitALSWrapperModel` + `PopularModel` + optional `RecommenderCoVis` |
| `RecommenderCoVis` | `recommenders/recommender_covis.py:21` | `covis_youtravel` | `model_realtime_warm_users` | - |
| `RecommenderPopular` | `recommenders/recommender_popular.py:30` | `popular_youtravel` | `model_cold_users` | `PopularModel` |
| `RecommenderEASE` | `recommenders/recommender_ease.py:29` | `ease_youtravel` | `model_hot_users`, `model_cold_users` | `EASEModel` |

Consumers, verified:

- `serving/model.py:126-135` resolves the class from the Triton model name by an
  ordered `elif` ladder (`als_covis` before `als` before `popular`/`ease`/`covis`),
  then `load_model(load_dir=script_path)`.
- `serving/model.py:200-234` parses the six `config.pbtxt` inputs and calls
  `recommend(user_ids, top_n, filter_viewed, items_to_recommend, history)`.
- `app/experiments/model_training_hybrid.py:905-921` (parent repo) maps
  `als|popular|ease|covis|als_covis` to classes and calls
  `train(dataset)` then `save_model_triton(...)`.
- `save_and_load_triton_models.py:76` writes `model.pkl` and syncs
  `model.py` + `config.pbtxt` into every version directory.

### 1.2 Research hierarchy (`models/`, `policy/`)

rectools `ModelBase` subclasses, `ModelConfig` pydantic configs, dict-of-dicts
construction via `model_from_config`.

| Class | File | Consumers | Artifact |
|---|---|---|---|
| `CoVisModel` | `models/covis.py:20` | `evaluation/next_item.py:98` (`covis_scorer`), tests, `evaluate_e2e` source configs | none |
| `PolicyModel` | `policy/model.py:14` | `evaluation/e2e.py` via config only, tests | none |
| `SourceSpec`, `PolicyModelConfig` | `policy/config.py:7,15` | `PolicyModel` | - |

Verified by grep across the whole monorepo: **no training script, notebook, ETL
job or API path references `PolicyModel` or `CoVisModel`.** Their only non-test
consumers are the three offline protocols in `evaluation/`. Neither implements
`save_model_triton`; `app/docs/SMARTREC_V2_RESEARCH.md:293-298` already states
this explicitly.

### 1.3 What is genuinely duplicated

Three axes, and they are in very different shape:

| Axis | Serving side | Research side | Verdict |
|---|---|---|---|
| Co-visitation algorithm | `recommender_covis.py:82-162` | `models/covis.py:72-171` | **Duplicated.** ~45 lines of the same algorithm, drifted in 4 ways (section 1.4) |
| Rank fusion | `rrf_fuse` from `policy/fusion.py:4` | same function | **Already unified.** Both import the same `rrf_fuse` |
| Composition / routing | `recommender_als.py:375-429` (2-source, hand-rolled) + `:601-616` | `policy/model.py:116-167` (n-ary, generic) | **Duplicated in intent, not in code.** Different capabilities; neither is a superset (section 1.6) |
| Constraints (share cap) | none | `policy/constraints.py:5` | Research-only; no serving equivalent |
| Configuration | pydantic `BaseSettings`, `model.py:20-127` | rectools `ModelConfig` dicts, `policy/config.py` | **Duplicated.** Same hyperparameters expressed twice |
| Session event weights | `recommender_covis.py:135-152` | none | Serving-only |
| Session-strength tiers | none | `policy/fusion.py:27` + `policy/model.py:109-114` | Research-only |

Note the last two rows: each side has a session feature the other lacks. This is
not one hierarchy trailing the other, it is bidirectional drift.

### 1.4 The CoVis behavioural diff - resolved

**Verdict: same algorithm, four real divergences, all of them parameter-or-defect
level rather than algorithmic.** Verified empirically, not just by reading, with a
differential probe (reproduction in section 6); the equivalence half of that probe
is now committed as `tests/test_covis_equivalence.py`.

The equivalence result first, because it is the load-bearing one:

> With the research-only caps neutralised (`session_size >= len(session)`, all
> baskets smaller than `fit_basket_size`), `CoVisModel` and `RecommenderCoVis`
> produce **identical neighbour graphs and identical session scores** on the
> shared fixture - same keys, same edge weights, same ranked items, same float
> scores. There is no algorithmic drift in the core.

The divergences:

| # | Divergence | `CoVisModel` | `RecommenderCoVis` | Bites in production when | Evidence |
|---|---|---|---|---|---|
| 1 | Basket cap at fit | caps each user to the `fit_basket_size=100` most recent interactions (`covis.py:78-83`) | no cap - full user history enters the basket (`recommender_covis.py:95-97`) | a user has >100 interactions in the 30d window; C(N,2) pair blow-up is also unbounded in serving | probe B: research drops the old item, serving keeps it, graphs differ |
| 2 | Session seed cap | truncates the seed to `session_size=20` (`covis.py:111`) | no cap - whole `history` is used (`recommender_covis.py:143-144`) | session >20 items. Redis caps at 50, so 21-50 diverges | probe C: same order, different scores (3.0/2.0 vs 3.4/2.2) - the recency denominator `n` differs. Ranking can also flip, since truncation removes seeds outright |
| 3 | Session event weights ("sw") | absent - no weight concept at all | seed recency x API event weight when `COVIS_SESSION_WEIGHTS` is on (`recommender_covis.py:146`) | the flag is on. Currently dev-only (`recsys_config_als_covis_dev`) | probe F |
| 4 | Tie-break determinism | `(-count, internal_id)` - deterministic (`covis.py:99,124`) | `count` desc only; ties resolved by dict insertion order, which follows **set iteration order over Python strings** (`recommender_covis.py:110,154`) | always, whenever counts tie at the `top_k` boundary | probe: same data, 4 `PYTHONHASHSEED` values -> **4 different top-3 neighbour sets** for the serving model; research identical in all 4 |

Divergence 4 is not a stylistic difference, it is a **reproducibility defect in the
production implementation**: retraining on byte-identical data can produce a
different `covis_youtravel` artifact from run to run. Measured:

```
serving  x00 top-3, PYTHONHASHSEED=0   -> ['x02', 'x07', 'x06']
serving  x00 top-3, PYTHONHASHSEED=1   -> ['x05', 'x02', 'x01']
serving  x00 top-3, PYTHONHASHSEED=42  -> ['x04', 'x07', 'x02']
serving  x00 top-3, PYTHONHASHSEED=999 -> ['x04', 'x01', 'x06']
research x00 top-3, all four seeds     -> ['x01', 'x02', 'x03']
```

Two further differences that are plumbing, not behaviour: the id space
(`CoVisModel.neighbors` is keyed by rectools **internal ints**, `RecommenderCoVis.neighbors`
by **external tour-id strings** - which is why the serving artifact survives a
retrain with a different id map and the research one would not), and the
empty-session return (`[]` vs a `RecomItems` carrying
`model_realtime_warm_users` even when empty, `recommender_covis.py:129`).

### 1.5 Which implementation produced which number in EXPERIMENTS.md

This is the question that makes the duplication expensive, and the answer is
"both, at different times". Verified by reading the journal; the runner scripts
themselves live in a scratchpad and are **not in the repo**, so the attribution
below is read off the journal's own prose, not off the runners.

| Journal section | Implementation used | What it justified |
|---|---|---|
| 2026-07-31 B (next-item, covis 4x popular) | `CoVisModel` via `covis_scorer` | the decision to build a session layer at all |
| 2026-08-02 verification (covis inside `cross_validate`) | `CoVisModel` | "covis ~ ALS on warm users" |
| 2026-08-02 S (session replay, als-item-sim below popularity) | `CoVisModel` + an item-sim proxy | **REPLACE realtime with CoVis** |
| 2026-08-03 policy weight grid (+6.2% map@10, w=1.0) | `PolicyModel` + `CoVisModel` via `evaluate_e2e` | `BlendSettings` defaults `ALS_WEIGHT=1.0`, `COVIS_WEIGHT=1.0`, `RRF_K=60` - cited in `model.py:33-43` and `recommender_als.py:383-388` |
| 2026-08-13 `blend_all.py` (covis_only / als_covis ~2x production) | the **real prod artifacts** `covis_youtravel/20260812030329` and `als_youtravel/...`, fused in the runner "exactly as production would" | the current standing conclusion that `als_covis` is the right session path and today's production 70/30 blend is the worst of six options |

So: the *architecture* decision rests on the research implementation, and the
*final scorer choice* rests on the serving implementation. The journal itself
says the two protocols are not comparable ("not comparable to the 2026-08-03
grid's +6.2%"). The practical consequence for this proposal: **the blend weights
baked into `BlendSettings` were tuned on `CoVisModel` and are being served by
`RecommenderCoVis`**, whose graph differs per divergences 1 and 4. Nobody has
measured how much that matters. It is probably small - divergence 1 only affects
>100-interaction users, divergence 4 only affects exact ties - but "probably
small" is the honest statement, not "equivalent".

### 1.6 Neither composition implementation is a superset

`RecommenderALS`'s hand-rolled composition and `PolicyModel`'s generic one differ
in ways that matter:

| Capability | `RecommenderALS` | `PolicyModel` |
|---|---|---|
| Number of sources | fixed 2 (+ Popular as a routing fallback) | n-ary, from config |
| Live session as input | yes - `history` argument, the whole point | **no** - proxied from train interaction count (`policy/model.py:143`, admitted in its own docstring at `:20-25`) |
| Serves unknown user + session | yes - `covis_session` (`recommender_als.py:601-616`) | **structurally impossible** - `_recommend_cold` (`:180`) and `_recommend_u2i_warm` (`:194`) both route to `fallback_source` only |
| Overfetch before fusion | `top_n * 2` (`recommender_als.py:391`) | `k * overfetch`, default 3 |
| Session-strength weighting | none | `session_weight` tiers |
| Category share cap | none | yes |
| `filter_viewed` across sources | yes - re-excludes training-viewed after fusion (`:422`) | delegated to each source |
| Emits `Strategy` | yes | no |
| Publishes an artifact | yes | no |

The third row is the decisive one and it is a **structural** gap, not a
measurement caveat. In rectools terms production's most valuable segment -
"unknown user with a live session" - is a *cold* user, and `PolicyModel` sends
every cold user to the popularity fallback. `als_covis_blend` is reproducible in
`PolicyModel` only with the session proxy; `covis_session` is not reproducible at
all. A "just move serving to `PolicyModel`" proposal is therefore not a refactor
of unknown cost, it is blocked until `PolicyModel` grows an explicit-session
entry point.

### 1.7 One live migration debt, already incurred

Not caused by this proposal, but any config unification walks into it: the parent
repo's `app/src/settings.py` still imports `LighFMSettings` and `RandomSettings`
(deleted on this branch) and still passes the **flat** `SESSION_COVIS_ENABLED`,
`COVIS_*`, `BLEND_*` kwargs to `ALSSettings`. Verified: those kwargs are now
rejected outright.

```
ALSSettings(..., SESSION_COVIS_ENABLED=True, COVIS_TOP_K=100, ...)
-> ValidationError: 7 validation errors ... Extra inputs are not permitted
```

The nested-composition commit is therefore not yet deployable without a matching
parent-repo change. `__setstate__` covers *pickled* artifacts
(`model.py:83-120`, tested in `tests/test_config_composition.py`) but nothing
covers *constructor call sites*, which live outside this submodule.

---

## 2. Constraints, restated as things a proposal must survive

| # | Constraint | Concrete test |
|---|---|---|
| 1 | Triton wire contract fixed | `serving/config.pbtxt` inputs/outputs unchanged; `load_model` / `save_model_triton` still present; artifact layout `model.pkl` + `model.py` + `config.pbtxt` per version dir |
| 2 | `Strategy` strings published | all 13 values in `api/docs/DEBUG_INFO_CODEC.md` table 3 still emitted with identical spelling |
| 3 | Pickles keep loading | `recsys_config` unpickles; class identity and module path of every pickled class preserved, or a migration hook proven by test |
| 4 | Session semantics | the live-`history` path must stay live; no proposal may silently substitute the train-count proxy |
| 5 | Measured behaviour preservable | for each option, state whether it is bit-identical (refactor) or not (needs a measurement or A/B) |

A note on constraint 3 that shapes all three options: `model.pkl` is
`dill.dump(self.__dict__)`, and `load_model` does
`cls().__dict__.update(state_dict)` (`base.py:77-87`). The pickle therefore
contains the classes of *nested objects* (`recsys_config`, `covis`,
`model_hot_users`, `dataset`) but **not** the outer class - that comes from the
`serving/model.py` ladder. So renaming or moving `RecommenderALS` is survivable;
renaming or moving `RecommenderCoVis`, `ALSSettings`, `CoVisSettings`,
`PopularSettings` or `BlendSettings` is not, without a hook. This is the single
most important asymmetry for planning.

---

## 3. Options

### Option A - Do nothing structural; freeze the research hierarchy as an eval-only tool

**Target shape.** Unchanged. Document the split explicitly: `recommenders/` is
production, `models/` + `policy/` are the offline measurement toolkit, and they
are allowed to differ. Add the differential test that pins where they agree and
where they do not, so the divergence stops being folklore. Fix the determinism
defect (divergence 4) in `RecommenderCoVis` as an isolated bug fix.

**Layout.** No change.

**Changes needed.** Serving: none (or the one-line tie-break fix). Training:
none. Evaluation: none. Docs: this file plus a paragraph in the lib README.

**Constraints.** 1-4 trivially satisfied. 5: the tie-break fix is *not*
bit-identical - it changes which of several equally-co-occurring neighbours
survive `top_k`. Needs an offline re-measure on the existing next-item harness,
not a full A/B; or ship it behind a config flag defaulted off, which is
bit-identical.

**Migration order.** (1) commit this doc; (2) commit the differential test;
(3) optionally the tie-break fix.

**Rollback.** Every step is independent and revertible on its own.

**Honest case for it.** The research hierarchy costs ~380 lines and produces no
artifact, so its carrying cost is low. Its *value* is high and non-obviously
replaceable: the rectools-native shape is exactly what let CoVis be evaluated
inside stock `cross_validate` and what caught the "covis ~ ALS" question
(2026-08-02). Collapsing it into the serving hierarchy would mean giving up
`cross_validate` compatibility or re-implementing it. And the one thing that
actually hurt in production - the 70/30 unnormalised blend, worth ~2x map@10 per
2026-08-13 - is not a duplication problem at all. Option A does not fix the
disunity; it makes it deliberate and bounded.

**Honest case against.** The duplication is load-bearing for credibility: the
weights in `BlendSettings` were tuned on one implementation and are served by
another. Leaving that means every future offline number carries an unquantified
transfer error, and the next person to touch CoVis has to rediscover section 1.4.

---

### Option B - One algorithm, two thin shells (recommended)

**Target shape.** Extract the co-visitation algorithm into a single
parameterised kernel. `CoVisModel` becomes the rectools-facing shell over it;
`RecommenderCoVis` becomes the serving-facing shell over the same kernel, owning
only what is genuinely serving-specific: the external-string id space, history
parsing, event weights, `Strategy` emission and `save_model_triton`. Composition
and configuration are **not** touched.

**Layout sketch.**

```
smartrec_lib/
  models/
    covis_kernel.py      # NEW: pure functions, no id-space opinion, no framework
      build_cooccurrence(baskets, min_cooc, top_k, *, deterministic_ties) -> neighbours
      score_session(neighbours, seed, seed_weights, exclude, allowed, k, session_size)
    covis.py             # CoVisModel(ModelBase) -> kernel, internal int ids
  recommenders/
    recommender_covis.py # RecommenderCoVis(RecommenderModel) -> kernel, external str ids
                         # keeps: _parse_history, session_weights_enabled,
                         #        Strategy.MODEL_REALTIME_WARM_USERS, save_model_triton
```

The kernel is generic over the id type - it never converts ids, it only groups,
counts and sorts. That is what lets both shells share it while keeping their
different id spaces, which is the one difference that must NOT be unified
(section 4, trap 3).

Every divergence from section 1.4 becomes an explicit kernel parameter:

| Divergence | Kernel parameter | `CoVisModel` passes | `RecommenderCoVis` passes (B1, bit-compatible) |
|---|---|---|---|
| 1 basket cap | `fit_basket_size: int \| None` | `100` | `None` |
| 2 session cap | `session_size: int \| None` | `20` | `None` |
| 3 event weights | `seed_weights: Sequence[float] \| None` | `None` (all 1.0) | parsed weights when the flag is on |
| 4 tie-break | `deterministic_ties: bool` | `True` | `False` in B1, `True` in B2 |

**B1** is bit-identical for both hierarchies: same graphs, same scores, same
artifacts. **B2** is a follow-up commit that flips `deterministic_ties` to `True`
for serving, converging the last divergence.

**Changes in serving.** `RecommenderCoVis.train` and `.recommend` delegate to the
kernel. `recommend`'s signature, return type, `Strategy` value and
`neighbors` attribute layout (`Dict[str, List[Tuple[str, float]]]`) are
unchanged - which is what keeps existing `covis_youtravel` and
`als_covis_youtravel` pickles loading, because `neighbors` is the only state that
matters and its shape does not move. `serving/model.py` is untouched.
`RecommenderALS` is untouched (it calls `self.covis.recommend(...)`, whose
contract is preserved).

**Changes in training.** None. Same classes, same `train(dataset)`, same
`save_model_triton`.

**Changes in evaluation.** None for B1. `covis_scorer` keeps building a
`CoVisModel`. Optionally, a second scorer factory that builds a
`RecommenderCoVis` becomes trivial once the kernel is shared, which finally makes
"measure the thing production actually serves" a one-liner - that is the real
prize of this option.

**Constraints.** 1: untouched. 2: untouched. 3: `RecommenderCoVis` and
`CoVisSettings` keep their module paths and class names; the new module contains
only functions, which are never pickled (`dill.dump(self.__dict__)` stores
`neighbors`, `top_k`, `session_weights_enabled`, `recsys_config` - no callables).
Needs a test that an old `covis_youtravel` pickle round-trips; the existing
`test_legacy_covis_artifact_loads_and_recommends` already covers the shape.
4: untouched - the live-`history` path stays exactly where it is. 5: B1 is a
**refactor**, provably bit-identical, no measurement needed. B2 is a **behaviour
change** confined to tie-breaking; re-measure offline, or gate it.

**Migration order.**
1. Differential test pinning current agreement/divergence (this is the standalone
   first step - see section 5).
2. Add `covis_kernel.py` with the four parameters; port `CoVisModel` to it;
   assert the full existing suite is unchanged.
3. Port `RecommenderCoVis` to it with `deterministic_ties=False`, caps `None`.
   The differential test from step 1 is the proof of bit-compatibility.
4. (B2, separate commit, separately decidable) flip `deterministic_ties=True`,
   re-measure on the next-item harness, note it in EXPERIMENTS.md.

**Rollback.** Each of steps 2, 3, 4 is a single revertible commit that leaves the
tree working. Mid-way state (kernel exists, only `CoVisModel` uses it) is a
perfectly fine resting place: the duplication is not yet gone but the kernel is
already the canonical, tested implementation.

---

### Option C - Promote `PolicyModel` to a serving citizen

**Target shape.** `PolicyModel` gains an explicit-session entry point and a
serving adapter, and becomes the composition layer for a new artifact. The
hand-rolled composition inside `RecommenderALS` is eventually retired, but only
after the new artifact wins an A/B.

**Layout sketch.**

```
smartrec_lib/
  policy/
    model.py             # PolicyModel + NEW recommend_for_session(session, k, ...)
    session.py           # NEW: SessionSource protocol
                         #   score_session(session, seed_weights, k) -> [(item, score)]
                         #   implemented by CoVisModel; no-op for ALS/Popular/EASE
  recommenders/
    recommender_policy.py  # NEW RecommenderPolicy(RecommenderModel):
                           #   wraps a fitted PolicyModel + the pickled Dataset,
                           #   emits Strategy, implements save_model_triton
```

The key design insight that makes this tractable: in the current production
composition **only the CoVis source consumes the live session**. ALS and Popular
are session-independent - ALS scores from the user's stored embedding, Popular
from global counts. So a session-aware policy does not need every source to
become session-aware; it needs exactly one extra protocol method, implemented by
`CoVisModel` (which already has `recommend_for_session`, `models/covis.py:155`)
and defaulted to "no session opinion" for the rest.

**Changes in serving.** New `elif "policy" in model_name` branch in
`serving/model.py:126`, guarded to stay after the existing, more specific names.
New artifact name `policy_youtravel`. Existing branches untouched. The adapter
must map policy output onto the existing `Strategy` vocabulary rather than
inventing values - `als_covis_blend` for the fused hot+session case,
`covis_session` for non-hot+session, `model_cold_users` for the fallback,
`model_hot_users` for hot-without-session.

**Changes in training.** A new entry in the trainer's model map and a new config
in `app/src/settings.py`. `PolicyModel` needs `save_model_triton`; note it must
pickle the fitted `Dataset` (as `RecommenderALS` already does,
`recommender_als.py:166`) because every source's `recommend` requires a
`Dataset`. Artifact size and Triton RSS need checking - prod pkls are already
~1.85GB total per EXPERIMENTS.md 2026-08-04.

**Changes in evaluation.** This is where Option C pays off: the offline
`evaluate_e2e` numbers would finally be produced by the same object that serves,
closing the section 1.5 gap properly rather than approximately.

**Constraints.** 1: satisfied additively - new artifact, no existing one
rewritten. 2: satisfiable but requires care; the temptation to add new strategy
strings must be resisted, or the backend decodes `unknown`. 3: satisfied, because
nothing existing is renamed - a brand-new class is added. 4: **this is the whole
problem.** `PolicyModel` must gain a real session entry point, and
`_recommend_cold` / `_recommend_u2i_warm` must be able to route a cold user with
a session to the session source instead of the fallback. Without that,
`covis_session` is unrepresentable (section 1.6). 5: **not bit-identical, not
even approximately** - different overfetch (3 vs 2), session-strength tiers that
production does not have, per-source `filter_viewed` instead of the post-fusion
re-exclusion at `recommender_als.py:422`. This is an **A/B, not a refactor**, and
it should be a third arm alongside `als_youtravel` and `als_covis_youtravel`.

**Migration order.** (1) `SessionSource` protocol + `CoVisModel` implements it;
(2) `PolicyModel.recommend_for_session` + cold/warm-with-session routing, tested
offline; (3) `RecommenderPolicy` adapter + `save_model_triton`; (4) dev artifact,
flag off; (5) dev smoke via `_debug_info`; (6) prod artifact alongside the
others, flag off; (7) A/B. This is deliberately the phase table already in
`app/docs/SMARTREC_V2_RESEARCH.md:299-310`, which was written for exactly this.

**Rollback.** Clean at every phase because it is purely additive: delete the S3
directory, or leave the flag off. The point of no return is only step (8), when
`RecommenderALS`'s internal composition would be deleted - and that should not
happen until the A/B has concluded.

**Why not now.** Options A and B cost days and risk nothing. Option C costs
weeks, needs a live A/B to justify, and its main deliverable - a better session
path - has *already been obtained* by the cheap route (`als_covis_youtravel` is
in the prod bucket). The 2026-08-13 measurement says the outstanding win is to
make `als_covis` the default instead of the 70/30 blend, which needs no
unification at all.

---

## 4. Recommendation

**Do Option B (B1 first), and treat Option C as a separate, later, A/B-gated
project. Do not do the parts of C that look cheap.**

Reasoning:

1. The duplication that actually costs credibility is the *algorithm*, not the
   composition - because offline numbers were measured on one copy and production
   serves the other (section 1.5). Option B closes exactly that, and B1 closes it
   with a provable no-op.
2. The duplication that looks worst - composition written twice - is the one it
   would be wrong to collapse today. `PolicyModel` cannot express `covis_session`
   at all (section 1.6). "Just move to `PolicyModel`" would silently drop
   production's most valuable segment or replace a live session with a
   train-count proxy. That is a regression wearing a refactor's clothes.
3. Config duplication is real but is the least valuable to fix and the most
   dangerous to touch: those classes are pickled inside every artifact
   (constraint 3), and the branch has already incurred one un-migrated call-site
   break (section 1.7). Fix that debt before adding to it.
4. The largest available quality win requires no unification whatsoever
   (section 5, last paragraph).

**Cheapest first step, useful on its own even if nothing else is ever done:**
commit the differential/characterisation test for the two CoVis implementations
(done - `tests/test_covis_equivalence.py`, section 6). It is test-only, so it
cannot break serving, training or artifacts. Standalone value:

- it converts section 1.4 from prose into an executable spec, so the next person
  does not re-derive it;
- it proves the equivalence claim that any later unification depends on, *before*
  anyone starts the unification;
- it will fail loudly if someone "fixes" one implementation and drifts them
  further apart;
- it is the bit-compatibility proof for step 3 of Option B, so it is not throwaway
  work.

Second cheapest, also standalone: fix divergence 4. A production model whose
artifact depends on `PYTHONHASHSEED` is a defect independent of any unification
argument, and the fix is one sort key.

Finally, and outside the scope of this document but worth stating because it
dominates everything in it: per EXPERIMENTS.md 2026-08-13, the current default
`als_youtravel` session path (`0.7*als + 0.3*item_sim`,
`recommender_als.py:551`) is the **worst of six measured options**, and pure ALS
with no session term beats it by 55% map@10. Unifying the hierarchies changes no
user's feed. Switching the default session path does. If there is one thing to do
next, it is that, not this.

---

## 5. Non-goals and traps

Things that look like obvious wins and are not:

1. **"Delete `RecommenderCoVis`, it is a duplicate of `CoVisModel`."** It is not
   a duplicate: it is a different id space (external strings vs internal ints),
   plus `_parse_history`, plus event weights, plus `Strategy`, plus
   `save_model_triton`. Deleting it orphans the `covis_youtravel` artifact and the
   session layer inside `als_covis_youtravel`. Verified: it is also the *less
   correct* of the two (divergence 4), which makes the reflex to keep the
   "production" one and delete the "research" one exactly backwards.

2. **"Delete `PolicyModel` / `models/covis.py`, nothing uses them."** Verified
   that no artifact comes from them - but they are the substrate of
   `evaluation/`, which produced every architectural decision in EXPERIMENTS.md.
   Deleting them deletes the ability to evaluate anything inside stock
   `cross_validate`.

3. **Unifying the id space.** Tempting, since it is the biggest surface
   difference. Do not. `RecommenderCoVis.neighbors` uses external tour-id strings
   deliberately: the serving artifact carries no id map of its own, so external
   keys are what make it usable next to a separately-retrained ALS artifact and
   what let `RecommenderALS` pass raw session ids straight through. Internal ints
   would couple the artifact to a specific `Dataset` id map. The kernel in Option
   B is generic over id type precisely to avoid this trap.

4. **"Make `ALSSettings` a rectools `ModelConfig` so there is one config
   idiom."** `ALSSettings` is pickled inside every `model.pkl` in two S3 buckets.
   Changing its base class changes what pickle must reconstruct. `__setstate__`
   (`model.py:83`) handles field-shape changes, not base-class changes. If this is
   ever wanted, the migration is a new class plus a `__reduce__` hook on the old
   name, and it must be proven against a real prod pickle, not a synthetic one.

5. **"Have `RecommenderALS` call `PolicyModel` internally."** Would replace a
   working 2-source fusion with one that fetches `k*3` instead of `k*2`, applies
   session-strength tiers production has never had, and drops the post-fusion
   `filter_viewed` re-exclusion (`recommender_als.py:422`) that
   `test_hot_user_with_session_serves_blend` pins. Not behaviour-preserving; needs
   the same A/B as Option C, with none of Option C's upside.

6. **Renaming anything that appears in a pickle.** `RecommenderCoVis`,
   `ALSSettings`, `CoVisSettings`, `PopularSettings`, `BlendSettings` and every
   nested rectools model are reconstructed by module path from `model.pkl`. The
   outer recommender class is *not* (it comes from the `serving/model.py` ladder),
   so `RecommenderALS` is renameable and `RecommenderCoVis` is not - an asymmetry
   that is easy to get backwards.

7. **"`Strategy` should be a `str` Enum / models should return `.value`
   consistently."** Verified that `RecommenderPopular` returns the bare enum
   (`recommender_popular.py:96`) and `RecommenderALS` returns bare enums in the
   `no_strategy_...` branches, while everything else returns `.value`; pydantic
   coerces all of them to the same wire string, so the published contract is
   currently intact. Tidying this is safe but is *not* free of risk: it is one
   inattentive edit away from emitting `Strategy.MODEL_COLD_USERS` as the wire
   value and decoding as `unknown` in the backend. If touched, pin every branch
   with a test asserting the exact string.

8. **Reviving `MODEL_HOT_AND_COLD_USERS` or the reserved `popular` / `random`
   strategy ids.** No model emits them (`model.py:135-137`,
   DEBUG_INFO_CODEC.md notes). They are published contract and must stay in the
   enum, but they are not evidence of a missing model.

Explicit non-goals of this proposal: touching `api/`, `serving/config.pbtxt`,
`pixi.toml`/`pixi.lock`, the `feature/ease-model` branch, or the parent repo's
`app/src/settings.py` (which needs its own fix per section 1.7).

---

## 6. Reproducing the CoVis findings

The equivalence and the parameter divergences are committed as
`tests/test_covis_equivalence.py`:

```bash
cd app/smartrec/smartrec-lib
uv run --no-project --with-editable . --with pytest,pandas,rectools,implicit,scipy \
  python -c "import pytest,sys; sys.exit(pytest.main(['tests/','-q','-p','no:cacheprovider']))"
```

The hash-seed non-determinism (divergence 4) is deliberately **not** a committed
test: asserting non-determinism from inside one interpreter is not possible, and
a subprocess-based test would be slow and environment-sensitive. Reproduce it by
building an all-tied neighbour set (one item co-occurring exactly once with each
of N others, `top_k < N`), fitting `RecommenderCoVis`, and printing the kept
neighbours under several `PYTHONHASHSEED` values; `CoVisModel` on the same data
is invariant. The measured output is quoted in section 1.4.
