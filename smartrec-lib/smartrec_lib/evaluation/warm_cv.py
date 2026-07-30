import typing as tp

import pandas as pd
from rectools.dataset import Dataset
from rectools.metrics import (
    MAP,
    MRR,
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    DebiasConfig,
    HitRate,
    MeanInvUserFreq,
    Precision,
    Recall,
    Serendipity,
)
from rectools.metrics.base import MetricAtK
from rectools.model_selection import TimeRangeSplitter, cross_validate
from rectools.models.base import ModelBase

DEBIAS = DebiasConfig(iqr_coef=1.5, random_state=42)


def default_metrics(k: int) -> tp.Dict[str, MetricAtK]:
    """Accuracy + beyond-accuracy preset shared by all offline reports."""
    return {
        f"map@{k}": MAP(k=k),
        f"map_debiased@{k}": MAP(k=k, debias_config=DEBIAS),
        f"ndcg@{k}": NDCG(k=k),
        f"mrr@{k}": MRR(k=k),
        f"hitrate@{k}": HitRate(k=k),
        f"precision@{k}": Precision(k=k),
        f"recall@{k}": Recall(k=k),
        f"miuf@{k}": MeanInvUserFreq(k=k),
        f"arp@{k}": AvgRecPopularity(k=k),
        f"coverage@{k}": CatalogCoverage(k=k),
        f"serendipity@{k}": Serendipity(k=k),
    }


def evaluate_warm_cv(
    dataset: Dataset,
    models: tp.Dict[str, ModelBase],
    k: int = 10,
    n_splits: int = 3,
    test_size: str = "4D",
    ref_model: tp.Optional[str] = None,
) -> pd.DataFrame:
    """
    Warm k-fold protocol (the historical baseline protocol of this project):
    TimeRangeSplitter, cold users and items filtered out. Directly comparable
    with previous ALS research numbers. Compare only within one run - the data
    window is anchored to now() upstream and drifts day to day.
    """
    splitter = TimeRangeSplitter(
        test_size=test_size,
        n_splits=n_splits,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=True,
    )
    result = cross_validate(
        dataset=dataset,
        splitter=splitter,
        models=models,
        metrics=default_metrics(k),
        k=k,
        filter_viewed=True,
        ref_models=[ref_model] if ref_model else None,
        validate_ref_models=bool(ref_model),
    )
    df = pd.DataFrame(result["metrics"])
    value_columns = [c for c in df.columns if c not in ("model", "i_split")]
    return df.groupby("model")[value_columns].mean().sort_values(f"map@{k}", ascending=False)
