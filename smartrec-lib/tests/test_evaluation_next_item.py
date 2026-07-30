from smartrec_lib.evaluation import covis_scorer, evaluate_next_item, popular_scorer


def test_next_item_reports_both_scorers(dataset):
    table = evaluate_next_item(
        dataset,
        {"covis": covis_scorer({"min_cooc": 1}), "popular": popular_scorer(period_days=30)},
        k=3,
        n_splits=2,
        test_size="2D",
    )
    assert set(table.index) == {"covis", "popular"}
    assert "map@3" in table.columns
    assert "served_frac" in table.columns
    assert 0.0 <= table.loc["covis", "served_frac"] <= 1.0
    assert table.loc["popular", "served_frac"] == 1.0  # popularity always answers
