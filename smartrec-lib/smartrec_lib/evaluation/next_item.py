import typing as tp

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MAP, Precision, Recall, calc_metrics
from rectools.model_selection import TimeRangeSplitter

from smartrec_lib.models.covis import CoVisModel

# Most-recent-first external session items + k -> ranked external items.
SessionScorer = tp.Callable[[tp.Sequence[tp.Any], int], tp.List[tp.Any]]


def _fold_frames(dataset: Dataset, train_ids, test_ids) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """Slice interactions by iloc indices and convert back to external ids."""
    def to_external(part: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                Columns.User: dataset.user_id_map.convert_to_external(part[Columns.User].values),
                Columns.Item: dataset.item_id_map.convert_to_external(part[Columns.Item].values),
                Columns.Weight: part[Columns.Weight].values,
                Columns.Datetime: part[Columns.Datetime].values,
            }
        )

    df = dataset.interactions.df
    return to_external(df.iloc[train_ids]), to_external(df.iloc[test_ids])


def evaluate_next_item(
    dataset: Dataset,
    scorer_factories: tp.Dict[str, tp.Callable[[Dataset], SessionScorer]],
    k: int = 10,
    n_splits: int = 3,
    test_size: str = "4D",
) -> pd.DataFrame:
    """
    Session / next-item replay for session models (CoVis, sequence models).
    Reuses the same TimeRangeSplitter folds as the warm protocol: the scorer is
    built on fold-train, the seed is the user's train history (most recent
    first), the target is their test items. `served_frac` is the share of test
    users the scorer could answer at all - the rest are routed to the fallback
    by the policy in production.

    Caveat (from the spec): next-item metrics systematically reward "show what
    was already clicked"; they measure signal strength, not the final feed. The
    aggressiveness of session personalization is decided by an online A/B.
    """
    splitter = TimeRangeSplitter(
        test_size=test_size,
        n_splits=n_splits,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=True,
    )
    metrics = {f"map@{k}": MAP(k=k), f"recall@{k}": Recall(k=k), f"precision@{k}": Precision(k=k)}
    rows: tp.List[tp.Dict[str, tp.Any]] = []
    for train_ids, test_ids, _info in splitter.split(dataset.interactions):
        train_ext, test_ext = _fold_frames(dataset, train_ids, test_ids)
        if train_ext.empty or test_ext.empty:
            continue
        fold_dataset = Dataset.construct(interactions_df=train_ext)
        history = (
            train_ext.sort_values(Columns.Datetime, ascending=False)
            .groupby(Columns.User)[Columns.Item]
            .agg(list)
            .to_dict()
        )
        test_users = test_ext[Columns.User].unique().tolist()
        for name, factory in scorer_factories.items():
            scorer = factory(fold_dataset)
            reco_rows: tp.List[tp.Tuple[tp.Any, tp.Any, int]] = []
            served = 0
            for user in test_users:
                session = history.get(user, [])
                ranked = scorer(session, k) if session else []
                if ranked:
                    served += 1
                for rank, item in enumerate(ranked, start=1):
                    reco_rows.append((user, item, rank))
            reco = pd.DataFrame(reco_rows, columns=[Columns.User, Columns.Item, Columns.Rank])
            if reco.empty:
                # calc_metrics chokes on an empty reco frame for some metrics; an
                # empty reco legitimately means "this scorer served nobody in this
                # fold", so score it as zeros rather than letting it raise.
                fold_metrics = {metric_name: 0.0 for metric_name in metrics}
            else:
                fold_metrics = calc_metrics(metrics, reco=reco, interactions=test_ext)
            fold_metrics["model"] = name
            fold_metrics["served_frac"] = served / max(len(test_users), 1)
            rows.append(fold_metrics)
    df = pd.DataFrame(rows)
    return df.groupby("model").mean(numeric_only=True).sort_values(f"map@{k}", ascending=False)


def covis_scorer(config: tp.Optional[dict] = None) -> tp.Callable[[Dataset], SessionScorer]:
    """Build a CoVis scorer on the fold dataset via the online session path."""

    def factory(fold_dataset: Dataset) -> SessionScorer:
        model = CoVisModel(**(config or {}))
        model.fit(fold_dataset)

        def score(session: tp.Sequence[tp.Any], k: int) -> tp.List[tp.Any]:
            return [item for item, _ in model.recommend_for_session(session, fold_dataset, k)]

        return score

    return factory


def popular_scorer(period_days: int = 14) -> tp.Callable[[Dataset], SessionScorer]:
    """Global popularity baseline under the SAME protocol (isolates session lift)."""

    def factory(fold_dataset: Dataset) -> SessionScorer:
        df = fold_dataset.interactions.df
        cutoff = df[Columns.Datetime].max() - pd.Timedelta(days=period_days)
        popularity = (
            df[df[Columns.Datetime] >= cutoff]
            .groupby(Columns.Item)[Columns.User]
            .nunique()
            .sort_values(ascending=False)
        )
        ranked_external = list(fold_dataset.item_id_map.convert_to_external(popularity.index.values))

        def score(session: tp.Sequence[tp.Any], k: int) -> tp.List[tp.Any]:
            seen = set(session)
            return [item for item in ranked_external if item not in seen][:k]

        return score

    return factory
