# smartrec-lib

Training and serving toolkit for recommendation models. Built on top of
`rectools` and `implicit`, with a shared API for training, metrics, and
exporting models to Triton.

The architecture contract — layers, frozen invariants, and where a new file
belongs — is in [../CLAUDE.md](../CLAUDE.md). Read it before adding a module.

## What's inside

- **`RecommenderModel`** base class: `train`, `recommend`, `calc_metrics`, plus
  (de)serialization to `model.pkl` from a local dir or S3 / Triton.
- **Models:**
  - `RecommenderALS` — main model (implicit ALS) with item-similarity, a nested
    Popular sub-model for cold users, and an optional CoVis session layer.
  - `RecommenderPopular`, `RecommenderEASE`, `RecommenderCoVis` — same interface.
- **Config objects:** `ALSSettings`, `PopularSettings`, `EASESettings`,
  `CoVisSettings`, `BlendSettings` — explicit hyperparameters.
- **`save_and_load_triton_models`** — upload/download weights to S3 and prepare
  the Triton layout (`config.pbtxt`, `model.py`, old-version cleanup).

## How recommendations work

`recommend()` returns items plus a `strategy` (see `Strategy` in `model.py`) that
says how they were produced:

- **hot** users (seen during training) → personalized ALS embeddings;
- **cold** users (unknown) → popular-items fallback; **warm** users sit in between;
- passing a real-time `history` of recent item IDs enriches the result on the fly
  (`*_realtime_*` strategies) without retraining.

In production the model runs inside Triton via the Python backend in
`smartrec_lib/serving/` (`model.py` + `config.pbtxt`), which loads the exported
`model.pkl`.

There is no incremental fit: models are always retrained in full via
`train(dataset)`.

## Quick start

```python
from datetime import datetime

import pandas as pd
from rectools.dataset import Dataset
from smartrec_lib.model import ALSSettings
from smartrec_lib.recommenders.recommender_als import RecommenderALS

interactions = pd.DataFrame({
    "user_id": [1, 1, 2],
    "item_id": [10, 20, 30],
    "weight": [1.0, 1.0, 1.0],
    "datetime": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03"]),
})

model = RecommenderALS(
    recsys_config=ALSSettings(
        ALS_FACTORS=64, ALS_ITERATIONS=15,
        ALS_REGULARIZATION_FACTOR=0.05, ALS_ALPHA=2,
    ),
    model_name="als_model",
    model_version=datetime.now().strftime("%Y%m%d%H%M%S"),
)
model.train(Dataset.construct(interactions))
recommendations = model.recommend(user_ids=1, top_n=10, filter_viewed=True)
```

## Export to Triton

```python
from pathy import Pathy

model.save_model_triton(base_s3_url=Pathy("s3://recsys-models"), num_to_keep=3)
```

Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (and
`S3_ENDPOINT` for Yandex Cloud) so `fsspec` can reach the bucket.

## Tests

```bash
cd app/smartrec/smartrec-lib
uv run pytest
```

## Development: lint and format

All code must pass the repo's formatters and linters BEFORE committing.
CI (`.github/workflows/smartrec-lib-test.yaml`) hard-fails on flake8
`E9,F63,F7,F82`; black config lives in `pyproject.toml` (line-length 120).

```bash
cd app/smartrec

# format (black, line-length 120 from pyproject.toml)
uv run black smartrec-lib/smartrec_lib smartrec-lib/tests

# lint - hard gate (CI fails on these)
uv run flake8 smartrec-lib --count --select=E9,F63,F7,F82 --show-source --ignore=F821

# lint - broad pass (keep it clean on new code)
uv run flake8 smartrec-lib --ignore=C901,F821 --count --max-complexity=10 --max-line-length=127

# tests
uv run pytest smartrec-lib/tests
```

Conventions: keep style-only changes in separate commits from logic changes;
new code must come in already formatted (no follow-up "apply black" commits).

## Layout

```
smartrec_lib/
  model.py          L0  contract: RecomItems, Strategy, *Settings (frozen by pickles)
  kernels/          L1  pure algorithms: cooccurrence, fusion, constraints
  recommenders/     L2  the servable models; module paths frozen by pickles
  serving/          L2  Triton python backend (model.py + config.pbtxt)
  research/         L3  rectools-native CoVisModel and PolicyModel
  evaluation/       L3  offline protocols
```

Imports flow strictly downward, and `recommenders/` never imports `research/`.
The rules, and the invariants that make the L2 paths frozen, are in
[../CLAUDE.md](../CLAUDE.md).

## Research components (rectools-native)

These follow the rectools `ModelBase` contract, so configs, save/load and
`cross_validate` come from the framework:

- `smartrec_lib.research.covis` - `CoVisModel`, session co-visitation; the
  online path (`recommend_for_session`) and the offline u2i path share one
  scoring routine (offline == online parity).
- `smartrec_lib.research.policy` - the serving policy as a regular rectools
  model: candidate sources (any rectools model config) -> weighted Reciprocal
  Rank Fusion -> category share cap; cold users fall back to the popularity
  source. Session source weight scales with session strength (one click is weak
  evidence - it must not turn the feed into a mono-category page).
- `smartrec_lib.evaluation` - three offline protocols: `evaluate_warm_cv`
  (accuracy + beyond-accuracy preset, ref model support), `evaluate_next_item`
  (session replay with `served_frac`), `evaluate_e2e` (whole system with cold
  users kept, hot/cold segments).

`recommenders/` keep serving production; migration is tracked in the parent
repo (`app/docs/SMARTREC_V2_RESEARCH.md`).

The co-visitation algorithm is written once, in `kernels/cooccurrence.py`. The
serving shell and `research/covis.py` differ only in the parameters they pass
(basket and session caps, per-seed event weights, tie determinism); which of
those differences are intentional is pinned by `tests/test_covis_equivalence.py`
and analysed in [docs/DESIGN_UNIFICATION.md](docs/DESIGN_UNIFICATION.md).
