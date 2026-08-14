from rectools.models import PopularModel, RandomModel

from smartrec_lib.evaluation import evaluate_warm_cv


def test_warm_cv_returns_ranked_table(dataset):
    models = {
        "popular": PopularModel(),
        "random": RandomModel(random_state=42),
    }
    table = evaluate_warm_cv(dataset, models, k=3, n_splits=2, test_size="2D")
    assert set(table.index) == {"popular", "random"}
    assert "map@3" in table.columns
    assert "coverage@3" in table.columns
    assert "miuf@3" in table.columns
    # table is sorted by map desc
    assert table["map@3"].is_monotonic_decreasing


def test_warm_cv_ref_model_present_on_all_folds(dataset):
    models = {"popular": PopularModel(), "random": RandomModel(random_state=42)}
    table = evaluate_warm_cv(dataset, models, k=3, n_splits=2, test_size="2D", ref_model="popular")
    assert "popular" in table.index
