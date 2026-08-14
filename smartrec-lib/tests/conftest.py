import pandas as pd
import pytest
from rectools import Columns
from rectools.dataset import Dataset

BASE = pd.Timestamp("2026-07-01")

# Co-occurrence counts this layout produces (per-user baskets):
#   (m1,m2)=2  (m2,m3)=2  (m1,m3)=1  (m2,pop1)=3  (m1,pop1)=2  (m3,pop1)=2
#   (t1,t2)=2  (t2,t3)=2  (t1,t3)=1  (t2,pop1)=3  (t1,pop1)=2  (t3,pop1)=2
# pop1 is the most popular item overall (6 users).
_ROWS = [
    ("u1", "m1", 1),
    ("u1", "m2", 2),
    ("u1", "pop1", 3),
    ("u2", "m1", 1),
    ("u2", "m2", 2),
    ("u2", "m3", 3),
    ("u2", "pop1", 4),
    ("u3", "m2", 2),
    ("u3", "m3", 3),
    ("u3", "pop1", 5),
    ("u4", "t1", 1),
    ("u4", "t2", 2),
    ("u4", "pop1", 6),
    ("u5", "t1", 2),
    ("u5", "t2", 3),
    ("u5", "t3", 4),
    ("u5", "pop1", 7),
    ("u6", "t2", 3),
    ("u6", "t3", 4),
    ("u6", "pop1", 8),
    ("u7", "t1", 8),  # cold-ish: single event on the last day (e2e cold segment)
]

ITEM_COUNTRY = {
    "m1": "maldives",
    "m2": "maldives",
    "m3": "maldives",
    "t1": "turkey",
    "t2": "turkey",
    "t3": "turkey",
    "pop1": "france",
}


def _interactions() -> pd.DataFrame:
    df = pd.DataFrame(_ROWS, columns=[Columns.User, Columns.Item, "day"])
    df[Columns.Weight] = 1.0
    df[Columns.Datetime] = df["day"].map(lambda d: BASE + pd.Timedelta(days=int(d)))
    return df[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]


@pytest.fixture
def interactions_df() -> pd.DataFrame:
    return _interactions()


@pytest.fixture
def dataset(interactions_df: pd.DataFrame) -> Dataset:
    return Dataset.construct(interactions_df=interactions_df)


@pytest.fixture
def dataset_with_features(interactions_df: pd.DataFrame) -> Dataset:
    item_features = pd.DataFrame(
        [(item, "tour_country_ru", country) for item, country in ITEM_COUNTRY.items()],
        columns=[Columns.Item, "feature", "value"],
    )
    return Dataset.construct(
        interactions_df=interactions_df,
        item_features_df=item_features,
        cat_item_features=["tour_country_ru"],
    )
