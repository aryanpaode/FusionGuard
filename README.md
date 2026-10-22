# FusionGuard

FusionGuard is a local entrance-monitoring application built with Flask and MySQL. It combines ArcFace face embeddings with periocular geometry, Mahalanobis distance evidence and local binary pattern texture features, then combines decisions across several webcam frames. The application supports enrolled users, temporary guest access and administrator review of recognition evidence.

**Author:** Aryan Paode · C033  
**Course:** Software Engineering · BTI-CE, Semester 9 · MPSTME

## Features

- **Four-view enrolment:** front, slight left, slight right and slight up captures, with checks for face presence, sharpness, brightness and face size. Administrators can activate or deactivate enrolled users.
- **Webcam monitoring:** consecutive usable-frame probes, a separate blink or head-motion liveness challenge, a recognition burst and a cooldown between completed sessions.
- **Multimodal recognition:** ArcFace-led weighted fusion, individual match gates, comparison against each user's best enrolment view and a strict majority across usable frames.
- **Administrator dashboard:** password-hashed login, session-based access, dashboard counts and filtered recognition history.
- **Incident review:** automatically refreshed incident records, captured evidence, component scores, frame-by-frame results, review status and notes. Selected incident logs can be deleted while the associated recognition session and captured evidence remain available.
- **Alerts and audit records:** browser alert tones for unknown, uncertain and suspected-spoof recognition results, persistent alert records and an audit trail of administrator actions.
- **Temporary guest keys:** random eight-digit keys with validity periods, usage limits and revocation. MySQL stores a peppered SHA-256 digest and the last four digits. Guest access attempts are logged; rejected attempts also create incidents and alert records.

## Recognition pipeline

![Recognition state diagram](docs/diagrams/recognition-state.png)

Monitoring begins after three consecutive probes pass the image-quality checks. The browser captures four liveness frames and then a separate burst of five recognition frames. The server evaluates the liveness challenge before comparing identities. Recognition requires at least three usable recognition frames; failed quality checks or feature-extraction errors remove a frame from the voting set.

For each usable frame, the recognition engine compares all enrolment views of every active user. It keeps the highest-scoring view for each user and ranks those per-user candidates.

### Weighted fusion

| Component | Weight | Evidence |
| --- | ---: | --- |
| ArcFace | 0.75 | Cosine similarity between normalised complete-face embeddings |
| Periocular geometry | 0.15 | Similarity of landmark-derived eye-region measurements |
| Mahalanobis evidence | 0.07 | Distance-based evidence from normalised geometry |
| Iris/LBP | 0.03 | Local binary pattern histogram similarity over the periocular crop |

The implementation maps ArcFace cosine similarity to a percentage scale, multiplies geometry and LBP cosine similarities by 100, and converts Mahalanobis distance to evidence using the configured distance ceiling. With the default ceiling of 3.0:

```text
A = 50 × (ArcFace cosine similarity + 1)
M = clip(100 × (1 − Mahalanobis distance / 3), 0, 100)
S = 0.75A + 0.15G + 0.07M + 0.03I
match score = 1 / (1 + exp(−(S − 68) / 4))
```

Here, `G` is the geometry similarity and `I` is the LBP histogram similarity, each multiplied by 100. The logistic layer maps the weighted score to a value between 0 and 1; this value is a configured match score, not a measured probability of correct identification.

The leading candidate receives an identity vote only when all applicable gates pass:

| Gate | Default requirement |
| --- | --- |
| ArcFace cosine distance | At most 0.68 |
| Mahalanobis distance | At most 3.0 |
| Logistic match score | At least 0.50 |
| Lead over the second-ranked user | At least 3 fusion-score points when another user is enrolled |

### Temporal decision

An identity must receive `floor(n / 2) + 1` votes across `n` usable recognition frames. Unknown and ambiguous frames remain in the denominator, so they count against a candidate's majority. For an authorised session, the reported confidence is the median logistic match score of the winning identity's supporting frames.

| Outcome | Meaning |
| --- | --- |
| `Authorised` | One enrolled identity has a strict majority of usable-frame votes. |
| `Unknown` | No enrolled identity receives a valid identity vote. |
| `Uncertain` | At least one identity receives a vote, but no identity achieves a strict majority. |
| `Suspected Spoof` | The liveness challenge fails before identity comparison. |

A completed session starts a ten-second cooldown. The webcam preview remains active, and the dashboard provides **Resume Now** and **Stop Monitoring** controls.

The defaults above are defined in [`config.py`](config.py) and can be overridden through environment variables. The default face sharpness check rejects a Laplacian variance below 3 and displays a softness warning below 18.

## Technology and modules

| Area | Implementation |
| --- | --- |
| Web application | Flask, Jinja2 templates and plain JavaScript |
| Webcam capture and alerts | Browser MediaDevices API and Web Audio API |
| Face features | DeepFace ArcFace with TensorFlow/tf-keras |
| Image processing and landmarks | OpenCV and MediaPipe |
| Numerical features and fusion | NumPy, SciPy and scikit-image |
| Persistence | MySQL with mysql-connector-python |
| Unit tests | pytest |

| Module | Responsibility |
| --- | --- |
| [`app.py`](app.py) | Routes, enrolment, recognition orchestration, incident review and guest access |
| [`fusionguard_multimodal.py`](fusionguard_multimodal.py) | Alignment, feature extraction, template serialisation and pairwise comparison |
| [`fusion_math.py`](fusion_math.py) | Evidence mappings, weighted score and logistic transform |
| [`temporal_verifier.py`](temporal_verifier.py) | Strict-majority voting and supporting-frame medians |
| [`quality.py`](quality.py) | Face presence, sharpness, brightness and size checks |
| [`liveness.py`](liveness.py) | Eye-aspect and head-motion checks across separate liveness frames |
| [`guest_keys.py`](guest_keys.py) | Guest-key generation, hashing and validity checks |
| [`db.py`](db.py) | Database initialisation, queries, records and audit logging |
| [`config.py`](config.py) | Environment configuration and runtime defaults |

MySQL stores users, feature templates, recognition sessions, frame results, incidents, alerts, guest keys, guest access logs and audit records. Captured images are stored under `static/uploads/`. The existing `recognition_frame_results.cnn_similarity` field stores ArcFace similarity.

The included [`mahalanobis_model_full_500.npz`](mahalanobis_model_full_500.npz) contains nine-dimensional `mean` and `std` arrays and an inverse covariance matrix. The recognition module uses these arrays for geometry-distance calculation. `MAHALANOBIS_MODEL_PATH` can select another file; a missing model file activates the identity-covariance fallback implemented in the module.

## Local runtime

The documented application environment is 64-bit Python 3.11 with MySQL 8 and a browser with webcam access. [Local deployment](docs/LOCAL_DEPLOYMENT.md) covers environment setup, database configuration and model preparation. [`.env.example`](.env.example) lists the connection settings, administrator credentials, application secrets and recognition parameters.

After the environment is configured, the main commands are:

```sh
python -m pip install -r requirements.txt
python prepare_arcface.py
python app.py
```

An optional [reference constraint set](constraints-python311.txt) is available for the Python 3.11 environment; it is not a complete dependency lock.

`prepare_arcface.py` loads the ArcFace model; DeepFace may download its weights on the first run. Starting `app.py` initialises the database and creates the configured administrator account when needed; first-time creation requires `FG_ADMIN_PASSWORD`. It then starts Flask's local development server at **http://127.0.0.1:5000**. The guest-entry page is `/guest`, and `/health` reports application and MySQL status.

The normal workflow is administrator login, four-view enrolment and **Start Monitoring**. Guest keys are managed from the dashboard and entered on the guest page.

## Repository structure

| Path | Contents |
| --- | --- |
| `templates/` | Login, administrator dashboard and guest-entry pages |
| `static/` | Browser code, stylesheet and empty runtime upload directories |
| `tests/` | Fusion, temporal-voting and guest-key unit tests |
| `tools/` | Mahalanobis model inspection utility |
| `models/` | Documentation of model assets and their storage |
| `docs/` | Architecture, local deployment, testing and the recognition state diagram |
| `database.sql` | MySQL schema without user records |
| `.env.example` | Configuration names, public defaults and empty credential fields |

## Testing and evaluation

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The existing suite contains 13 unit tests, all of which passed the recorded [logic validation](docs/TESTING.md). The suite covers fusion weights and evidence mappings, temporal majority decisions, supporting-frame confidence aggregation, and guest-key formatting and hashing. These logic tests run without starting MySQL or using a webcam. [Testing](docs/TESTING.md) describes the checks and their coverage; [architecture](docs/ARCHITECTURE.md) describes the application components and data flow.

FusionGuard is an academic software prototype. The liveness check is a blink/head-motion heuristic, and end-to-end recognition accuracy, false acceptance/rejection rates and presentation-attack resistance require a separate biometric evaluation. The application records software authorisation decisions and local alerts; physical lock control and remote notification services are outside its current implementation.
