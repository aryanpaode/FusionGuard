from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    ARCFACE_WEIGHT,
    GEOMETRY_WEIGHT,
    IRIS_WEIGHT,
    LOGISTIC_CENTER,
    LOGISTIC_SCALE,
    MAHALANOBIS_WEIGHT,
    MAX_MAHALANOBIS_DISTANCE,
)


def arcface_similarity_percent(cosine_similarity: float) -> float:
    """Map cosine similarity [-1, 1] to an intuitive 0-100 evidence scale."""
    if not math.isfinite(cosine_similarity):
        return 0.0
    clipped = max(-1.0, min(1.0, cosine_similarity))
    return 50.0 * (clipped + 1.0)


def mahalanobis_evidence_percent(distance: float) -> float:
    if not math.isfinite(distance) or distance < 0:
        return 0.0
    if MAX_MAHALANOBIS_DISTANCE <= 0:
        return 0.0
    evidence = 100.0 * (1.0 - distance / MAX_MAHALANOBIS_DISTANCE)
    return max(0.0, min(100.0, evidence))


def weighted_fusion_score(
    arcface_percent: float,
    geometry_percent: float,
    mahalanobis_distance: float,
    iris_percent: float,
) -> tuple[float, float, float]:
    mahal_percent = mahalanobis_evidence_percent(mahalanobis_distance)
    weighted = (
        ARCFACE_WEIGHT * arcface_percent
        + GEOMETRY_WEIGHT * geometry_percent
        + MAHALANOBIS_WEIGHT * mahal_percent
        + IRIS_WEIGHT * iris_percent
    )
    linear = (weighted - LOGISTIC_CENTER) / LOGISTIC_SCALE
    linear = max(-500.0, min(500.0, linear))
    probability = 1.0 / (1.0 + math.exp(-linear))
    return float(weighted), float(probability), float(mahal_percent)


@dataclass(frozen=True)
class FusionWeights:
    arcface: float = ARCFACE_WEIGHT
    geometry: float = GEOMETRY_WEIGHT
    mahalanobis: float = MAHALANOBIS_WEIGHT
    iris: float = IRIS_WEIGHT

    def as_percent_text(self) -> str:
        return (
            f"ArcFace {self.arcface * 100:.0f}% + Geometry {self.geometry * 100:.0f}% + "
            f"Mahalanobis {self.mahalanobis * 100:.0f}% + Iris/LBP {self.iris * 100:.0f}%"
        )
