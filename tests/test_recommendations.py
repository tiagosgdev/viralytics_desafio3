"""
tests/test_recommendations.py
──────────────────────────────
Unit tests for the RecommendationEngine.
"""

import pytest
from src.recommendations.engine import RecommendationEngine


@pytest.fixture
def engine():
    return RecommendationEngine(top_k=5)


def test_empty_detection_returns_trending(engine):
    recs = engine.recommend([])
    assert len(recs) > 0
    assert all("reason" in r for r in recs)


def test_top_k_respected(engine):
    recs = engine.recommend(["short_sleeve_top"])
    assert len(recs) <= 5


def test_recommendation_has_required_fields(engine):
    recs = engine.recommend(["trousers"])
    for r in recs:
        assert "name"     in r
        assert "category" in r
        assert "price"    in r
        assert "reason"   in r
        assert "score"    in r


def test_no_duplicate_recommendations(engine):
    recs = engine.recommend(["short_sleeve_top", "trousers"])
    ids  = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))


def test_dress_suggests_outwear(engine):
    recs = engine.recommend(["short_sleeve_dress"])
    cats = {r["category"] for r in recs}
    # Should suggest outerwear to complement a dress
    outwear = {"short_sleeve_outwear", "long_sleeve_outwear"}
    assert cats & outwear


def test_exclude_detected_works(engine):
    detected = ["trousers"]
    recs = engine.recommend(detected, exclude_detected=True)
    rec_cats = {r["category"] for r in recs}
    assert "trousers" not in rec_cats


def test_multiple_detections(engine):
    recs = engine.recommend(["short_sleeve_top", "shorts", "vest"])
    assert len(recs) > 0
