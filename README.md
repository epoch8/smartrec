# smartrec

A general-purpose recommendation library. It provides a single interface to train
recommendation models and serve them from Triton Inference Server, independent of
any particular product or domain.

Two packages:

| Package | What it does |
|---------|--------------|
| [`smartrec-lib`](smartrec-lib/) | Train models (ALS, Popular, Random, LightFM), compute metrics, export weights to S3 / Triton |
| [`smartrec-client`](smartrec-client/) | Lightweight Triton client to fetch recommendations over gRPC / HTTP |

## Typical setup

- A **training job** uses `smartrec-lib` to fit models and write them to S3;
  Triton serves the latest version from there.
- An **application** fetches personalized recommendations from Triton via
  `smartrec-client`.

The library is domain-agnostic — plug it into any service. It is usually vendored
as a git submodule.

## Quick start

```python
from datetime import datetime

import pandas as pd
from rectools.dataset import Dataset
from smartrec_lib.model import ALSSettings, ModelSetSettings
from smartrec_lib.recommenders.recommender_als import RecommenderALS

interactions = pd.DataFrame({
    "user_id": [1, 1, 2],
    "item_id": [10, 20, 30],
    "weight": [1.0, 1.0, 1.0],
    "datetime": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03"]),
})

# A member takes its OWN leaf config; only the set takes ModelSetSettings.
model = RecommenderALS(
    recsys_config=ALSSettings(
        ALS_FACTORS=64, ALS_ITERATIONS=15,
        ALS_REGULARIZATION_FACTOR=0.05, ALS_ALPHA=2,
    ),
    model_name="als_model",
    model_version=datetime.now().strftime("%Y%m%d%H%M%S"),
)
model.train(Dataset.construct(interactions))
recs = model.recommend(user_ids=1, top_n=10, filter_viewed=True)
```

See the per-package docs: [smartrec-lib](smartrec-lib/README.md) ·
[smartrec-client](smartrec-client/README.md).
