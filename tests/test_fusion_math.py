from config import ARCFACE_WEIGHT, GEOMETRY_WEIGHT, IRIS_WEIGHT, MAHALANOBIS_WEIGHT
from fusion_math import arcface_similarity_percent, mahalanobis_evidence_percent, weighted_fusion_score


def test_weights_sum_to_one():
    assert abs(ARCFACE_WEIGHT + GEOMETRY_WEIGHT + MAHALANOBIS_WEIGHT + IRIS_WEIGHT - 1.0) < 1e-9


def test_locked_weights():
    assert ARCFACE_WEIGHT == 0.75
    assert GEOMETRY_WEIGHT == 0.15
    assert MAHALANOBIS_WEIGHT == 0.07
    assert IRIS_WEIGHT == 0.03


def test_arcface_similarity_mapping():
    assert arcface_similarity_percent(-1.0) == 0.0
    assert arcface_similarity_percent(0.0) == 50.0
    assert arcface_similarity_percent(1.0) == 100.0
    assert round(arcface_similarity_percent(0.32), 5) == 66.0


def test_mahalanobis_evidence_mapping():
    assert mahalanobis_evidence_percent(0.0) == 100.0
    assert round(mahalanobis_evidence_percent(1.5), 5) == 50.0
    assert mahalanobis_evidence_percent(3.0) == 0.0
    assert mahalanobis_evidence_percent(5.0) == 0.0


def test_probability_is_bounded():
    _, probability, _ = weighted_fusion_score(82, 80, 1.0, 75)
    assert 0.0 <= probability <= 1.0
