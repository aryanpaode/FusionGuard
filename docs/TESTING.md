# Testing

FusionGuard includes 13 unit tests for fusion mathematics, temporal voting and guest-key generation. The suite exercises the existing Python logic without starting the application server, connecting to MySQL or capturing webcam images.

## Running the unit suite

From the repository root, in an activated Python environment:

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

[`requirements-test.txt`](../requirements-test.txt) contains `pytest` and `python-dotenv`. These dependencies are sufficient for the current logic tests. The full application dependencies are listed separately in [`requirements.txt`](../requirements.txt).

The tests assume the supplied recognition defaults, including fusion weights of `0.75 / 0.15 / 0.07 / 0.03` and a Mahalanobis ceiling of `3.0`. Environment variables and a local `.env` file can override configuration values, so a test environment should retain those defaults.

## Test coverage

| Test file | Count | Checks |
| --- | ---: | --- |
| [`test_fusion_math.py`](../tests/test_fusion_math.py) | 5 | Weights sum to one; exact configured weights; ArcFace cosine-to-percentage mapping; Mahalanobis distance-to-evidence mapping; logistic output remains between zero and one. |
| [`test_temporal_verifier.py`](../tests/test_temporal_verifier.py) | 5 | Strict-majority thresholds; four-of-five authorisation with the median supporting-frame score; unknown votes count against a majority; conflicting identities without a majority produce `Uncertain`; all-unknown frames produce `Unknown`. |
| [`test_guest_keys.py`](../tests/test_guest_keys.py) | 3 | Generated keys contain exactly eight digits; the hash differs from plaintext and has the expected digest length; a sample of generated keys is not constant. |
| **Total** | **13** | |

The guest-key tests cover formatting, hashing and basic generation behaviour. Database-backed expiry, revocation and usage-limit handling require integration checks. The generation test does not establish a uniqueness or randomness guarantee.

## Recorded validation

The validation recorded on **22 October 2026** used **Python 3.12.14**, `pytest 9.1.1` and `python-dotenv 1.2.4` for the pure-logic suite.

| Check | Recorded result |
| --- | --- |
| Existing unit tests | **13 passed; 0 failures, 0 errors, 0 skipped** |
| Python syntax | All 15 Python files passed syntax parsing. |
| Browser JavaScript syntax | `static/app.js` passed the Node.js syntax check. |
| Included Mahalanobis model | Required arrays were present, dimensionally consistent and finite. |

No Flask server, MySQL service, webcam capture, ArcFace inference or other ML integration was run as part of this validation. Syntax and model-structure checks establish only the properties listed above.

The full application setup documented in [Local deployment](LOCAL_DEPLOYMENT.md) uses **Python 3.11**. The successful Python 3.12.14 logic run does not establish compatibility of the complete MediaPipe, TensorFlow, DeepFace and database stack on that interpreter.

## Application integration checks

The following checks remain separate from the unit-suite result. They require the full application environment, MySQL and, where relevant, a webcam and enrolled test users.

| Area | Integration check |
| --- | --- |
| Database and administrator access | Initialise the schema, create the configured administrator, log in and verify persistence across application restarts. |
| Enrolment | Capture all four views, exercise quality rejection, store templates and verify active/inactive user handling. |
| Model preparation | Load ArcFace weights, extract templates and confirm that the configured Mahalanobis model is used. |
| Webcam and liveness | Verify camera permission, consecutive readiness probes, separate liveness capture and blink/head-motion outcomes. |
| Recognition sessions | Exercise `Authorised`, `Unknown`, `Uncertain`, retry and `Suspected Spoof` paths with appropriate test inputs. |
| Monitoring controls | Verify the ten-second cooldown, continuing preview, **Resume Now** and **Stop Monitoring**. |
| Incidents and evidence | Check alert records, browser tones, evidence links, frame details, review notes and incident-log deletion behaviour. |
| Guest access | Check valid, premature, expired, revoked, exhausted and unknown keys, including access history and rejected-attempt incidents. |

Recognition accuracy, false acceptance/rejection rates, threshold calibration and presentation-attack resistance require a defined biometric evaluation dataset and protocol. The included unit suite provides no measured claim for those outcomes.

## Related documentation

- [Project overview](../README.md)
- [Local deployment](LOCAL_DEPLOYMENT.md)
- [Architecture](ARCHITECTURE.md)
- [Model files](../models/README.md)
