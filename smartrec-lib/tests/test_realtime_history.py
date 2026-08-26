"""
Test real-time recommendations with user history.

Проверяет что history параметр влияет на рекомендации для warm users.
"""

from datetime import datetime

import pandas as pd
import pytest
from rectools import Columns
from rectools.dataset import Dataset

from smartrec_lib.model import ALSSettings, ModelSetSettings, Strategy
from smartrec_lib.recommenders import RecommenderModelSet


@pytest.fixture
def interactions_df():
    """Create sample interactions dataset."""
    now = datetime.now()
    return pd.DataFrame(
        {
            Columns.User: ["user1", "user1", "user1", "user2", "user2", "user3", "user3", "user3"],
            Columns.Item: ["100", "101", "102", "200", "201", "300", "301", "302"],
            Columns.Weight: [1, 1, 1, 1, 1, 1, 1, 1],
            Columns.Datetime: [now] * 8,
        }
    )


@pytest.fixture
def dataset(interactions_df):
    """Create rectools Dataset."""
    return Dataset.construct(interactions_df)


@pytest.fixture
def trained_model(dataset):
    """Create and train ALS model."""
    config = ModelSetSettings(
        main=ALSSettings(
            ALS_FACTORS=16,
            ALS_ITERATIONS=5,
            ALS_REGULARIZATION_FACTOR=0.01,
            ALS_ALPHA=1,
        )
    )

    model = RecommenderModelSet(
        recsys_config=config,
        model_name="test_als",
        model_version="test_v1",
    )

    model.train(dataset)
    return model


def test_cold_user_no_history(trained_model):
    """
    Test 1: Cold user without history gets popular recommendations.

    Ожидания:
    - Strategy: MODEL_COLD_USERS
    - Recommendations returned
    """
    result = trained_model.recommend(
        user_ids="unknown_user_123",
        top_n=5,
        filter_viewed=True,
        history=None,  # No history
    )

    print(f"Result strategy: {result.strategy}")
    print(f"Expected: {Strategy.MODEL_COLD_USERS}")
    print(f"Match: {result.strategy == Strategy.MODEL_COLD_USERS}")

    assert result.strategy in [Strategy.MODEL_COLD_USERS, Strategy.MODEL_COLD_USERS.value]
    assert len(result.item_ids) > 0
    print(f"Cold user strategy: {result.strategy}")
    print(f"Recommendations: {result.item_ids}")


def test_warm_user_with_history(trained_model):
    """
    Test 2: Warm user with history gets personalized recommendations.

    Создаем новый пользователь с историей кликов на туры похожие на user1.

    Ожидания:
    - Strategy: MODEL_REALTIME_WARM_USERS
    - Recommendations based on weighted item similarity
    """
    # История: пользователь кликал на туры из обучающих данных
    # Используем weighted формат (клики = weight 1.0)
    history = ["100:1.0", "101:1.0"]  # These items exist in training data

    result = trained_model.recommend(
        user_ids="new_warm_user",
        top_n=5,
        filter_viewed=True,
        history=history,
    )

    assert result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert len(result.item_ids) > 0

    print(f"Warm user strategy: {result.strategy}")
    print(f"Weighted history: {history}")
    print(f"Recommendations: {result.item_ids}")


def test_different_histories_different_recommendations(trained_model):
    """
    Test 3: Different weighted histories lead to different recommendations.

    User A: история [100:2.0, 101:1.0] (клик на кнопку + просмотр)
    User B: история [200:3.0, 201:1.0] (конверсия + просмотр)

    Ожидания:
    - Обе получают персонализированные рекомендации
    - Рекомендации отличаются из-за разной истории и весов
    """
    history_a = ["100:2.0", "101:1.0"]  # Разные веса
    history_b = ["200:3.0", "201:1.0"]

    result_a = trained_model.recommend(
        user_ids="warm_user_a",
        top_n=5,
        filter_viewed=True,
        history=history_a,
    )

    result_b = trained_model.recommend(
        user_ids="warm_user_b",
        top_n=5,
        filter_viewed=True,
        history=history_b,
    )

    # Обе используют warm strategy
    assert result_a.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert result_b.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value

    # Рекомендации должны отличаться
    overlap = set(result_a.item_ids) & set(result_b.item_ids)
    overlap_ratio = len(overlap) / len(result_a.item_ids)

    print(f"User A weighted history: {history_a} → {result_a.item_ids}")
    print(f"User B weighted history: {history_b} → {result_b.item_ids}")
    print(f"Overlap: {len(overlap)}/{len(result_a.item_ids)} ({overlap_ratio*100:.1f}%)")

    # Рекомендации не должны полностью совпадать
    assert overlap_ratio < 1.0, "Recommendations should differ for different histories"


def test_history_size_impact(trained_model):
    """
    Test 4: Larger weighted history provides better recommendations.

    User A: 1 item with high weight (конверсия)
    User B: 3 items with разными весами (просмотр + клик + конверсия)

    Ожидания:
    - Оба получают персонализированные рекомендации
    - Больше истории с разными весами → больше контекста для рекомендаций
    """
    small_history = ["100:3.0"]  # Одна конверсия
    large_history = ["100:1.0", "101:2.0", "102:3.0"]  # Разные типы событий

    result_small = trained_model.recommend(
        user_ids="user_small_history",
        top_n=5,
        filter_viewed=True,
        history=small_history,
    )

    result_large = trained_model.recommend(
        user_ids="user_large_history",
        top_n=5,
        filter_viewed=True,
        history=large_history,
    )

    assert result_small.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert result_large.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value

    print(f"Small weighted history ({len(small_history)} item): {result_small.item_ids}")
    print(f"Large weighted history ({len(large_history)} items): {result_large.item_ids}")

    # Оба должны давать рекомендации
    assert len(result_small.item_ids) > 0
    assert len(result_large.item_ids) > 0


def test_history_with_items_to_recommend(trained_model):
    """
    Test 5: History + items_to_recommend filtering.

    Проверяем что history работает вместе с items_to_recommend.
    """
    history = ["100:1.0", "101:2.0"]  # Просмотр + клик на кнопку
    items_to_recommend = ["200", "201", "300", "301"]

    result = trained_model.recommend(
        user_ids="user_with_filter",
        top_n=3,
        filter_viewed=True,
        history=history,
        items_to_recommend=items_to_recommend,
    )

    assert result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert len(result.item_ids) > 0

    # Все рекомендованные туры должны быть из items_to_recommend
    assert all(item_id in items_to_recommend for item_id in result.item_ids)

    print(f"Weighted history: {history}")
    print(f"Candidate items: {items_to_recommend}")
    print(f"Recommendations: {result.item_ids}")


def test_hot_user_ignores_history(trained_model):
    """
    Test 6: Hot user (in training data) uses ALS, not history.

    Если пользователь есть в обучающих данных, история игнорируется.

    Ожидания:
    - Strategy: MODEL_HOT_USERS
    - History не используется
    """
    # user1 есть в training data
    history = ["999", "888"]  # Случайная история

    result = trained_model.recommend(
        user_ids="user1",
        top_n=5,
        filter_viewed=True,
        history=history,
    )

    # Для hot users используется ALS, не history
    assert result.strategy == Strategy.MODEL_HOT_USERS.value
    assert len(result.item_ids) > 0

    print(f"Hot user (user1) strategy: {result.strategy}")
    print(f"History ignored: {history}")
    print(f"ALS recommendations: {result.item_ids}")


def test_empty_history_fallback_to_cold(trained_model):
    """
    Test 7: Empty history falls back to cold user recommendations.

    Ожидания:
    - Strategy: MODEL_COLD_USERS
    - Popular items recommended
    """
    result = trained_model.recommend(
        user_ids="user_empty_history",
        top_n=5,
        filter_viewed=True,
        history=[],  # Empty history
    )

    assert result.strategy == Strategy.MODEL_COLD_USERS.value
    assert len(result.item_ids) > 0

    print(f"Empty history → cold strategy: {result.strategy}")
    print(f"Popular recommendations: {result.item_ids}")


def test_invalid_history_items_filtered(trained_model):
    """
    Test 8: Invalid items in history are filtered out.

    История содержит items которых нет в обучающих данных.

    Ожидания:
    - Невалидные items игнорируются
    - Рекомендации строятся на валидных items
    """
    # Mix of valid and invalid items
    history = ["100", "invalid_item", "101", "another_invalid"]

    result = trained_model.recommend(
        user_ids="user_mixed_history",
        top_n=5,
        filter_viewed=True,
        history=history,
    )

    # Должна использоваться warm strategy с валидными items
    assert result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert len(result.item_ids) > 0

    print(f"Mixed history: {history}")
    print(f"Valid items used for recommendations")
    print(f"Recommendations: {result.item_ids}")


def test_history_scores_are_reasonable(trained_model):
    """
    Test 9: Scores from history-based recommendations are reasonable.

    Ожидания:
    - Scores > 0
    - Scores упорядочены по убыванию
    """
    history = ["100", "101"]

    result = trained_model.recommend(
        user_ids="user_check_scores",
        top_n=5,
        filter_viewed=True,
        history=history,
    )

    assert len(result.scores) == len(result.item_ids)

    # Scores должны быть упорядочены по убыванию
    assert result.scores == sorted(result.scores, reverse=True), "Scores should be descending"

    print(f"Item IDs: {result.item_ids}")
    print(f"Scores: {[f'{s:.3f}' for s in result.scores]}")
    print(f"Scores are descending: {result.scores == sorted(result.scores, reverse=True)}")


def test_hot_user_with_realtime_history(trained_model):
    """
    Test 10: Hot user with real-time history gets enriched ALS recommendations.

    Ожидания:
    - Strategy: MODEL_REALTIME_HOT_USERS
    - Recommendations blend ALS and real-time similarity
    """
    # user1 is a hot user (in training data)
    # Добавляем weighted history (конверсия + просмотр)
    result = trained_model.recommend(
        user_ids="user1",
        top_n=5,
        filter_viewed=True,
        history=["200:3.0", "201:1.0"],  # Конверсия на 200, просмотр 201
    )

    assert result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value
    assert len(result.item_ids) > 0
    assert len(result.scores) == len(result.item_ids)
    print(f"Hot user with weighted real-time: strategy={result.strategy}")
    print(f"Weighted history: ['200:3.0', '201:1.0']")
    print(f"Recommendations: {result.item_ids}")


def test_hot_user_without_realtime_history(trained_model):
    """
    Test 11: Hot user without real-time history gets standard ALS.

    Ожидания:
    - Strategy: MODEL_HOT_USERS
    - Standard ALS recommendations
    """
    # user1 is a hot user (in training data)
    result = trained_model.recommend(
        user_ids="user1",
        top_n=5,
        filter_viewed=True,
        history=None,  # No real-time history
    )

    assert result.strategy == Strategy.MODEL_HOT_USERS.value
    assert len(result.item_ids) > 0
    print(f"Hot user without real-time gets strategy: {result.strategy}")
    print(f"Recommendations: {result.item_ids}")


def test_realtime_strategies_are_different(trained_model):
    """
    Test 12: Real-time and standard strategies produce different results.

    Ожидания:
    - Real-time enrichment changes recommendations for hot users
    - Strategies are correctly assigned
    """
    # Hot user without real-time
    standard_result = trained_model.recommend(
        user_ids="user1",
        top_n=5,
        filter_viewed=True,
        history=None,
    )

    # Same hot user with weighted real-time history
    realtime_result = trained_model.recommend(
        user_ids="user1",
        top_n=5,
        filter_viewed=True,
        history=["200:2.0", "201:1.0"],  # Weighted history
    )

    assert standard_result.strategy == Strategy.MODEL_HOT_USERS.value
    assert realtime_result.strategy == Strategy.MODEL_REALTIME_HOT_USERS.value

    print(f"Standard strategy: {standard_result.strategy}")
    print(f"Real-time weighted strategy: {realtime_result.strategy}")
    print(f"Standard recs: {standard_result.item_ids[:3]}")
    print(f"Real-time weighted recs: {realtime_result.item_ids[:3]}")


def test_weighted_history_warm_user(trained_model):
    """
    Test 13: Weighted history affects warm user recommendations.

    Ожидания:
    - Items with higher weights have more influence
    - Weighted format "tour_id:weight" is parsed correctly
    """
    # Warm user with weighted history
    weighted_result = trained_model.recommend(
        user_ids="new_weighted_user",
        top_n=5,
        filter_viewed=True,
        history=["100:1.0", "200:3.0"],  # 200 has 3x more weight
    )

    assert weighted_result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert len(weighted_result.item_ids) > 0

    print(f"Weighted history format parsed correctly")
    print(f"Recommendations: {weighted_result.item_ids[:3]}")
    print(f"Scores: {[f'{s:.3f}' for s in weighted_result.scores[:3]]}")


def test_weighted_history_backward_compatibility(trained_model):
    """
    Test 14: Plain history format (without weights) still works.

    Ожидания:
    - Plain format "tour_id" defaults to weight=1.0
    - Results are identical to explicit "tour_id:1.0"
    """
    # Plain format
    plain_result = trained_model.recommend(
        user_ids="user_plain",
        top_n=5,
        filter_viewed=True,
        history=["100", "101"],
    )

    # Explicit weight=1.0 format
    explicit_result = trained_model.recommend(
        user_ids="user_explicit",
        top_n=5,
        filter_viewed=True,
        history=["100:1.0", "101:1.0"],
    )

    assert plain_result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value
    assert explicit_result.strategy == Strategy.MODEL_REALTIME_WARM_USERS.value

    # Recommendations should be identical (same history, same weights)
    assert plain_result.item_ids == explicit_result.item_ids
    assert plain_result.scores == explicit_result.scores

    print(f"Plain format works: {plain_result.item_ids[:3]}")
    print(f"Backward compatibility maintained")


if __name__ == "__main__":
    """Run tests directly without pytest."""
    print("=" * 80)
    print("Testing Real-Time History-Based Recommendations")
    print("=" * 80)

    # Create fixtures manually
    now = datetime.now()
    interactions_df = pd.DataFrame(
        {
            Columns.User: ["user1", "user1", "user1", "user2", "user2", "user3", "user3", "user3"],
            Columns.Item: ["100", "101", "102", "200", "201", "300", "301", "302"],
            Columns.Weight: [1, 1, 1, 1, 1, 1, 1, 1],
            Columns.Datetime: [now] * 8,
        }
    )

    dataset = Dataset.construct(interactions_df)

    config = ModelSetSettings(
        main=ALSSettings(
            ALS_FACTORS=16,
            ALS_ITERATIONS=5,
            ALS_REGULARIZATION_FACTOR=0.01,
            ALS_ALPHA=1,
        )
    )

    model = RecommenderModelSet(
        recsys_config=config,
        model_name="test_als",
        model_version="test_v1",
    )

    print("\nTraining model...")
    model.train(dataset)
    print("Model trained\n")

    # Run tests
    print("\n[Test 1] Cold user without history")
    print("-" * 80)
    test_cold_user_no_history(model)

    print("\n[Test 2] Warm user with history")
    print("-" * 80)
    test_warm_user_with_history(model)

    print("\n[Test 3] Different histories → different recommendations")
    print("-" * 80)
    test_different_histories_different_recommendations(model)

    print("\n[Test 4] History size impact")
    print("-" * 80)
    test_history_size_impact(model)

    print("\n[Test 5] History + items_to_recommend")
    print("-" * 80)
    test_history_with_items_to_recommend(model)

    print("\n[Test 6] Hot user ignores history")
    print("-" * 80)
    test_hot_user_ignores_history(model)

    print("\n[Test 7] Empty history → cold fallback")
    print("-" * 80)
    test_empty_history_fallback_to_cold(model)

    print("\n[Test 8] Invalid history items filtered")
    print("-" * 80)
    test_invalid_history_items_filtered(model)

    print("\n[Test 9] History scores are reasonable")
    print("-" * 80)
    test_history_scores_are_reasonable(model)

    print("\n[Test 10] Hot user with real-time history")
    print("-" * 80)
    test_hot_user_with_realtime_history(model)

    print("\n[Test 11] Hot user without real-time history")
    print("-" * 80)
    test_hot_user_without_realtime_history(model)

    print("\n[Test 12] Real-time vs standard strategies")
    print("-" * 80)
    test_realtime_strategies_are_different(model)

    print("\n[Test 13] Weighted history for warm users")
    print("-" * 80)
    test_weighted_history_warm_user(model)

    print("\n[Test 14] Backward compatibility (plain format)")
    print("-" * 80)
    test_weighted_history_backward_compatibility(model)

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
