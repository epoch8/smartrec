from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel
from enum import Enum



class RecomItem(BaseModel):
    item_id: str
    score: float


class RecommenderModel:
    model_class: str
    model_version: str

    def __init__(self) -> None:
        pass

    def save_model(self, save_dir: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load_model(cls, load_dir: str) -> "RecommenderModel":
        raise NotImplementedError()

    @classmethod
    def get_train_data(cls) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    def train(cls, train) -> "RecommenderModel":
        raise NotImplementedError()

    def status(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def recommend(
        self,
        user_id: str,
        user_history: List[str],
        top_n: int,
        filter_already_liked_items: bool,
        items: Optional[List[str]],
    ) -> Tuple[List[RecomItem], Dict]:
        raise NotImplementedError()

    def look_alike_items(
        self,
        item_id: str,
        top_n: int = 5,
        items: Optional[List[str]] = None,
    ) -> Tuple[List[RecomItem], Dict]:
        raise NotImplementedError()