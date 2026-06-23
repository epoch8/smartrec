# smartrec-lib

Training and serving toolkit for YouTravel recommendation models. Built on top of
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

> **Note:** `RecommenderALS.train_partial` (incremental fit) is **under
> development** and currently raises `NotImplementedError`. Use `train()` for
> full retraining.

## Quick start

```python
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
    model_name="als_youtravel",
    model_version="20241024",
)
model.train(Dataset.construct(interactions))
recommendations = model.recommend(user_ids=1, top_n=10, filter_viewed=True)
```

## Export to Triton

```python
from pathy import Pathy

model.save_model_triton(base_s3_url=Pathy("s3://youtravel-recsys"), num_to_keep=3)
```

Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (and
`S3_ENDPOINT` for Yandex Cloud) so `fsspec` can reach the bucket.

## Tests

```bash
cd app/smartrec/smartrec-lib
uv run pytest
```
