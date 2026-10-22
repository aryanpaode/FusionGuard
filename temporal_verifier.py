from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable


def strict_majority_threshold(usable_frames: int) -> int:
    if usable_frames <= 0:
        return 1
    return usable_frames // 2 + 1


def finalise_temporal_decision(frame_results: Iterable[dict]) -> dict:
    """Aggregate usable live-frame decisions.

    Each frame result is expected to already contain exactly one per-frame identity
    decision after best-angle-per-user comparison and ambiguity checking.
    Unknown/Uncertain frame votes count against a user's majority.
    """
    results = list(frame_results)
    usable = len(results)
    required = strict_majority_threshold(usable)

    identity_votes = [
        item.get("frame_identity")
        for item in results
        if item.get("frame_identity")
    ]
    counts = Counter(identity_votes)
    winner_name = None
    winner_votes = 0

    if counts:
        winner_name, winner_votes = counts.most_common(1)[0]

    if winner_name is not None and winner_votes >= required:
        supporting = [item for item in results if item.get("frame_identity") == winner_name]
        probabilities = [float(item["logistic_probability"]) for item in supporting]
        fusion_scores = [float(item["fusion_score"]) for item in supporting]
        margins = [float(item.get("identity_margin") or 0.0) for item in supporting]
        return {
            "decision": "Authorised",
            "winner_name": winner_name,
            "winner_votes": winner_votes,
            "usable_frames": usable,
            "required_votes": required,
            "median_confidence": median(probabilities),
            "median_fusion_score": median(fusion_scores),
            "median_identity_margin": median(margins),
            "supporting_frames": supporting,
            "vote_counts": dict(counts),
        }

    # If multiple enrolled identities received positive frame votes but none reached
    # a strict majority, the session is ambiguous rather than a clean unknown.
    positive_identities = [name for name, count in counts.items() if count > 0]
    decision = "Uncertain" if len(positive_identities) >= 2 else "Unknown"

    # A single identity receiving some positive votes but not a strict majority is
    # also uncertain: evidence exists, but it is not temporally consistent enough.
    if len(positive_identities) == 1 and winner_votes > 0:
        decision = "Uncertain"

    all_probabilities = [float(item.get("logistic_probability", 0.0)) for item in results]
    all_scores = [float(item.get("fusion_score", 0.0)) for item in results]

    return {
        "decision": decision,
        "winner_name": winner_name,
        "winner_votes": winner_votes,
        "usable_frames": usable,
        "required_votes": required,
        "median_confidence": median(all_probabilities) if all_probabilities else 0.0,
        "median_fusion_score": median(all_scores) if all_scores else 0.0,
        "median_identity_margin": 0.0,
        "supporting_frames": [],
        "vote_counts": dict(counts),
    }
