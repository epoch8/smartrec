# smartrec-lib

Набор утилит для обучения и обслуживания рекомендательных моделей YouTravel. Библиотека построена поверх `rectools`, `implicit`, `lightfm` и хранит общее API для тренировки, расчёта метрик и выгрузки моделей в Triton.

## Что реализовано
- Базовый класс `RecommenderModel` с методами `train`, `recommend`, `calc_metrics`, сериализацией в `model.pkl` и загрузкой из локального каталога или S3/Triton.
- Модели:
  - `RecommenderALS` c поддержкой частичного дообучения (`train_partial`) и пересчётом item-similarity.
  - `RecommenderLightFM`, `RecommenderPopular`, `RecommenderRandom` — простейшие baseline'ы с единым интерфейсом.
- Конфиги `ALSSettings`, `LighFMSettings`, `PopularSettings`, `RandomSettings` для явного задания гиперпараметров.
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
Для инкрементального обновления передайте новый `Dataset` в `model.train_partial(...)`. Если найдены новые user/item ID, класс выполнит полный `fit`.

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
В каталоге `smartrec-lib/tests` лежат smoke-тесты для ALS (включая `train_partial`). Запуск:
```bash
cd app/smartrec/smartrec-lib
uv run pytest tests
```
