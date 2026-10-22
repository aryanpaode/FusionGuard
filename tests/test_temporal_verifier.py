from temporal_verifier import finalise_temporal_decision, strict_majority_threshold


def frame(identity, probability, fusion=85.0, margin=6.0):
    return {
        "frame_identity": identity,
        "candidate_name": identity or "Best Candidate",
        "candidate_user_id": 1 if identity else None,
        "logistic_probability": probability,
        "fusion_score": fusion,
        "identity_margin": margin,
    }


def test_strict_majority_thresholds():
    assert strict_majority_threshold(3) == 2
    assert strict_majority_threshold(4) == 3
    assert strict_majority_threshold(5) == 3
    assert strict_majority_threshold(6) == 4


def test_four_of_five_authorises_and_uses_median_supporting_confidence():
    results = [frame("Aryan", .90), frame("Aryan", .80), frame(None, .20), frame("Aryan", .70), frame("Aryan", .60)]
    final = finalise_temporal_decision(results)
    assert final["decision"] == "Authorised"
    assert final["winner_name"] == "Aryan"
    assert final["winner_votes"] == 4
    assert final["required_votes"] == 3
    assert abs(final["median_confidence"] - .75) < 1e-9


def test_unknown_votes_count_against_majority():
    results = [frame("Aryan", .9), frame("Aryan", .8), frame(None, .2), frame(None, .2), frame(None, .1)]
    final = finalise_temporal_decision(results)
    assert final["decision"] == "Uncertain"
    assert final["winner_votes"] == 2
    assert final["required_votes"] == 3


def test_conflicting_identities_without_majority_are_uncertain():
    results = [frame("Aryan", .85), frame("Aryan", .82), frame("Aaryan", .80), frame("Aaryan", .77), frame(None, .20)]
    final = finalise_temporal_decision(results)
    assert final["decision"] == "Uncertain"


def test_all_unknown_is_unknown():
    results = [frame(None, .2), frame(None, .1), frame(None, .25)]
    final = finalise_temporal_decision(results)
    assert final["decision"] == "Unknown"
