# smartrec-client

Lightweight Python client for models served by **Triton Inference Server**.

## Features

- Picks gRPC or HTTP automatically from the URL scheme.
- Reads model metadata and builds input tensors from the Triton spec.
- Skips empty / `None` inputs and decodes `BYTES` outputs to Python strings.
- `recommendations_triton` helper returns recommendations in a single call.

## Install

```bash
cd app/smartrec/smartrec-client
uv pip install -e .
```

Requires Python >=3.10 (`tritonclient` is pulled in automatically).

## Usage

```python
from smartrec_client import recommendations_triton

result = recommendations_triton(
    triton_server_url="grpc://localhost:8001",
    model_name="als_model",
    user_ids="12345",
    top_n=10,
    filter_viewed=True,
    items_to_recommend=["501", "907"],  # optional
    history=["501", "907"],             # optional
)
print(result["model_version"], result["data"])
```

Make sure Triton is reachable at the given URL and the model is deployed.
