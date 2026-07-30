import typing as tp

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MAP, CoveredUsers, Recall, SufficientReco, UnrepeatedReco, calc_metrics
from rectools.model_selection import TimeRangeSplitter
from rectools.models.serialization import model_from_config

from smartrec_lib.evaluation.next_item import _fold_frames


def evaluate_e2e(
    dataset: Dataset,
    model_configs: tp.Dict[str, dict],
    k: int = 10,
    n_splits: int = 3,
    test_size: str = "4D",
) -> pd.DataFrame:
    """
    Whole-system protocol: cold users are KEPT in test folds, so the number
    answers "what does the feed look like for everyone", not "how good is the
    warm model". Segments: hot = test user has interactions in fold-train,
    cold = does not. The online-only segment "cold user with a live session"
    cannot be replayed from batch interactions - it is covered separately by
    the next-item protocol.

    Models are given as configs (data, not code) and re-fitted on every fold.

    The fold train dataset is built via `dataset.filter_interactions(...)`
    (the same call `cross_validate` uses), NOT `Dataset.construct` from an
    external-id frame - `Dataset.construct` would drop item/user features,
    which silently disables any policy behaviour that depends on them (e.g.
    PolicyModel's category share cap).

    Caveat: the "cold" segment (and, by extension, "all") only appears in
    folds that actually contain cold test users, so it can be averaged over
    fewer folds than "hot". The `n_folds` column (how many folds contributed
    to a (model, segment) average) makes this visible; `n_users` itself is
    also a per-fold mean, not a total across folds.
    """
    splitter = TimeRangeSplitter(
        test_size=test_size,
        n_splits=n_splits,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=False,
    )
    metrics = {
        f"map@{k}": MAP(k=k),
        f"recall@{k}": Recall(k=k),
        f"covered@{k}": CoveredUsers(k=k),
        f"sufficient@{k}": SufficientReco(k=k),
        f"unrepeated@{k}": UnrepeatedReco(k=k),
    }
    rows: tp.List[tp.Dict[str, tp.Any]] = []
    for train_ids, test_ids, _info in splitter.split(dataset.interactions):
        # Only used for the external test frame and the external train-user
        # set needed for hot/cold segmentation - NOT for building the fold
        # train dataset (that must go through filter_interactions to keep
        # item/user features, see the docstring above).
        train_ext, test_ext = _fold_frames(dataset, train_ids, test_ids)
        if train_ext.empty or test_ext.empty:
            continue
        fold_dataset = dataset.filter_interactions(
            row_indexes_to_keep=train_ids,
            keep_external_ids=True,
            keep_features_for_removed_entities=True,
        )
        train_users = set(train_ext[Columns.User])
        test_users = test_ext[Columns.User].unique().tolist()
        segments = {
            "all": set(test_users),
            "hot": {u for u in test_users if u in train_users},
            "cold": {u for u in test_users if u not in train_users},
        }
        for name, config in model_configs.items():
            model = model_from_config(config)
            model.fit(fold_dataset)
            reco = model.recommend(
                users=test_users,
                dataset=fold_dataset,
                k=k,
                filter_viewed=True,
                on_unsupported_targets="ignore",
            )
            for segment, users in segments.items():
                if not users:
                    continue
                seg_reco = reco[reco[Columns.User].isin(users)]
                seg_interactions = test_ext[test_ext[Columns.User].isin(users)]
                seg_metrics = calc_metrics(metrics, reco=seg_reco, interactions=seg_interactions)
                seg_metrics.update({"model": name, "segment": segment, "n_users": len(users)})
                rows.append(seg_metrics)
    df = pd.DataFrame(rows)
    grouped = df.groupby(["model", "segment"])
    result = grouped.mean(numeric_only=True)
    result["n_folds"] = grouped.size()
    return result
