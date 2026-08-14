from smartrec_lib.kernels.constraints import apply_share_cap

CATS = {"m1": "mv", "m2": "mv", "m3": "mv", "m4": "mv", "t1": "tr", "t2": "tr"}


def test_mono_category_is_capped():
    ranked = ["m1", "m2", "m3", "t1", "t2", "m4"]
    result = apply_share_cap(ranked, CATS, k=4, max_share=0.5)  # cap = 2 per category
    assert result == ["m1", "m2", "t1", "t2"]  # third maldives item skipped, order kept


def test_backfill_prefers_full_page_over_strict_cap():
    # Only one category available: cap would leave the page short - backfill fills it.
    ranked = ["m1", "m2", "m3", "m4"]
    result = apply_share_cap(ranked, CATS, k=4, max_share=0.25)  # cap = 1
    assert result == ["m1", "m2", "m3", "m4"]  # order preserved, page full


def test_items_without_category_are_never_capped():
    ranked = ["x1", "x2", "x3"]
    result = apply_share_cap(ranked, CATS, k=3, max_share=0.34)
    assert result == ["x1", "x2", "x3"]


def test_relative_order_is_preserved():
    ranked = ["m1", "t1", "m2", "t2", "m3"]
    result = apply_share_cap(ranked, CATS, k=4, max_share=0.5)
    assert result == ["m1", "t1", "m2", "t2"]
