# smartrec-lib

Набор утилит для обучения и обслуживания рекомендательных моделей YouTravel. Библиотека построена поверх `rectools` и `implicit` и хранит общее API для тренировки, расчёта метрик и выгрузки моделей в Triton.

## Что реализовано
- Базовый класс `RecommenderModel` с методами `train`, `recommend`, `calc_metrics`, сериализацией в `model.pkl` и загрузкой из локального каталога или S3/Triton.
- Модели:
  - `RecommenderALS` — основная прод-модель, с пересчётом item-similarity и вложенными подмоделями (Popular для холодных, опционально CoVis для сессии).
  - `RecommenderPopular`, `RecommenderEASE`, `RecommenderCoVis` — модели с тем же единым интерфейсом.
- Конфиги `ALSSettings`, `PopularSettings`, `EASESettings`, `CoVisSettings`, `BlendSettings` для явного задания гиперпараметров.
- Модуль `save_and_load_triton_models` для загрузки/выгрузки весов в S3 и подготовки структуры Triton (`config.pbtxt`, `model.py`, очистка старых версий).

## Быстрый старт
```python
import pandas as pd
from rectools.dataset import Dataset
from smartrec_lib.model import ALSSettings
from smartrec_lib.recommenders.recommender_als import RecommenderALS

interactions = pd.DataFrame(
    {
        "user_id": [1, 1, 2],
        "item_id": [10, 20, 30],
        "weight": [1.0, 1.0, 1.0],
        "datetime": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03"]),
    }
)
dataset = Dataset.construct(interactions)
model = RecommenderALS(
    recsys_config=ALSSettings(
        ALS_FACTORS=64,
        ALS_ITERATIONS=15,
        ALS_REGULARIZATION_FACTOR=0.05,
        ALS_ALPHA=2,
    ),
    model_name="als_youtravel",
    model_version="20241024",
)
model.train(dataset)
recommendations = model.recommend(user_ids=1, top_n=10, filter_viewed=True)
```
Инкрементального дообучения нет: модели переобучаются целиком через `train(dataset)`.

## Выгрузка в Triton
```python
from pathy import Pathy

model.save_model_triton(
    base_s3_url=Pathy("s3://youtravel-recsys"),
    num_to_keep=3,
)
```
Установите переменные окружения `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` и при необходимости `S3_ENDPOINT`, чтобы `fsspec` смог подключиться к бакету.

## Тесты
В каталоге `smartrec-lib/tests` лежат smoke-тесты для ALS и остальных моделей. Запуск:
```bash
cd app/smartrec/smartrec-lib
uv run pytest tests
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

Having two hierarchies means the co-visitation algorithm exists twice
(`models/covis.py` and `recommenders/recommender_covis.py`). Where the two agree
and where they have drifted is pinned by `tests/test_covis_equivalence.py` and
analysed, with unification options, in
[docs/DESIGN_UNIFICATION.md](docs/DESIGN_UNIFICATION.md).
