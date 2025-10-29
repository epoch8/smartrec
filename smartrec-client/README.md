# smartrec-client

Легковесный Python-клиент для взаимодействия с моделями, которые поднимает Triton Inference Server.

## Возможности
- Автоматически выбирает gRPC или HTTP клиент по схеме в `triton_server_url`.
- Загружает метаданные модели и формирует входные тензоры по описанию Triton.
- Пропускает пустые/`None` входы и приводит выводы к спискам Python, декодируя `BYTES` в строки.
- Предоставляет удобный хелпер `recommendations_triton` для получения рекомендаций в одном вызове.

## Установка
```bash
cd app/smartrec/smartrec-client
uv pip install -e .
```
Требуется Python ≥3.10 и установленный `tritonclient` (подтягивается автоматически).

## Пример использования
```python
import numpy as np
from smartrec_client import TritonModelClient, recommendations_triton

# Низкоуровневый клиент
client = TritonModelClient(url="grpc://localhost:8001", model_name="als_youtravel")
raw = client(
    user_ids=np.array(["12345"], dtype=np.object_),
    top_n=np.array([10], dtype=np.int32),
    filter_viewed=np.array([True], dtype=np.bool_),
)

# Хелпер с подготовкой входов
result = recommendations_triton(
    triton_server_url="grpc://localhost:8001",
    user_ids="12345",
    model_name="als_youtravel",
    top_n=10,
    filter_viewed=True,
    items_to_recommend=["501", "907"],
    history=["501", "907"],
)
print(result["model_version"], result["data"])
```
Перед запуском убедитесь, что Triton доступен по переданному адресу и модель задеплоена.
