from rectools import Columns


def test_dataset_shape(dataset):
    assert dataset.user_id_map.size == 7
    assert dataset.item_id_map.size == 7


def test_dataset_with_features_has_categories(dataset_with_features):
    names = list(dataset_with_features.item_features.names)
    assert ("tour_country_ru", "maldives") in names
    assert ("tour_country_ru", "turkey") in names


def test_interactions_have_datetime(interactions_df):
    assert interactions_df[Columns.Datetime].notna().all()
