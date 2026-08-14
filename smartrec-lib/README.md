# smartrec-lib

Training and serving toolkit for recommendation models. Built on top of
`rectools`, `implicit`, and `lightfm`, with a shared API for training, metrics,
and exporting models to Triton.

## What's inside

- **`RecommenderModel`** base class: `train`, `recommend`, `calc_metrics`, plus
  (de)serialization to `model.pkl` from a local dir or S3 / Triton.
- **Models:**
  - `RecommenderALS` — main model (implicit ALS) with item-similarity.
  - `RecommenderPopular`, `RecommenderRandom`, `RecommenderLightFM` — baselines
    behind the same interface.
- **Config objects:** `ALSSettings`, `PopularSettings`, `RandomSettings`,
  `LighFMSettings` — explicit hyperparameters.
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

> **Note:** `RecommenderALS.train_partial` (incremental fit) is **under
> development** and currently raises `NotImplementedError`. Use `train()` for
> full retraining.

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

## v2 core (rectools-native)

New-generation components live next to the legacy `recommenders/` package and
follow the rectools `ModelBase` contract, so configs, save/load and
`cross_validate` come from the framework:

- `smartrec_lib.models` - custom models missing from rectools. `CoVisModel`:
  session co-visitation; the online path (`recommend_for_session`) and the
  offline u2i path share one scoring routine (offline == online parity).
- `smartrec_lib.policy` - the serving policy as a regular rectools model:
  candidate sources (any rectools model config) -> weighted Reciprocal Rank
  Fusion -> category share cap; cold users fall back to the popularity source.
  Session source weight scales with session strength (one click is weak
  evidence - it must not turn the feed into a mono-category page).
- `smartrec_lib.evaluation` - three offline protocols: `evaluate_warm_cv`
  (accuracy + beyond-accuracy preset, ref model support), `evaluate_next_item`
  (session replay with `served_frac`), `evaluate_e2e` (whole system with cold
  users kept, hot/cold segments).

Legacy `recommenders/` stay untouched and keep serving production; migration is
tracked in the parent repo (`app/docs/SMARTREC_V2_RESEARCH.md`).
