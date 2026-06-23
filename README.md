# smartrec

Internal ML library for the **YouTravel recommender system**. It gives a single
interface to train recommendation models and serve them from Triton Inference
Server.

Two packages:

| Package | What it does |
|---------|--------------|
| [`smartrec-lib`](smartrec-lib/) | Train models (ALS, Popular, Random, LightFM), compute metrics, export weights to S3 / Triton |
| [`smartrec-client`](smartrec-client/) | Lightweight Triton client to fetch recommendations over gRPC / HTTP |

## Where it is used

- **`youtravel-recsys/app`** trains models via `smartrec-lib` and writes them to
  S3; Triton serves from there.
- **`youtravel-recsys/api`** serves personalized feeds and queries Triton via
  `smartrec-client`.

It is vendored as a git submodule in both.

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
recs = model.recommend(user_ids=1, top_n=10, filter_viewed=True)
```

See the per-package docs: [smartrec-lib](smartrec-lib/README.md) ·
[smartrec-client](smartrec-client/README.md).
