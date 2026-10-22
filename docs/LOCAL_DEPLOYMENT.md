# Local deployment

FusionGuard runs as a local Flask application with a browser webcam and MySQL. The supplied entry point binds to `127.0.0.1:5000` with Flask debugging enabled. This setup is intended for a single-machine academic demonstration.

## Prerequisites

- Python 3.11, 64-bit.
- MySQL Server 8, running locally or at the configured database address.
- A browser with webcam permission and a working camera.
- Internet access for dependency installation and the first ArcFace model download.

## Install the environment

Run these commands from the project root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Keep the environment active when using the Python commands or batch launchers. On other supported platforms, create and activate a Python 3.11 virtual environment using the platform's equivalent commands.

The optional `constraints-python311.txt` supplies reference versions for selected numerical, TensorFlow and OpenCV dependencies:

```powershell
python -m pip install -r requirements.txt -c constraints-python311.txt
```

These constraints are reference pins, not a validated complete environment lock. They supplement `requirements.txt`.

## Configure `.env`

| Setting | Purpose |
|---|---|
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD` | MySQL connection details |
| `MYSQL_DATABASE` | Database to initialize and use |
| `FG_ADMIN_USERNAME`, `FG_ADMIN_PASSWORD` | First administrator account; set a unique private password before startup |
| `FLASK_SECRET_KEY` | Private random value used to sign browser sessions |
| `FG_GUEST_KEY_PEPPER` | Separate private random value used when hashing guest keys |

The empty secret fields in `.env` require private values. The command `python -c "import secrets; print(secrets.token_hex(32))"` generates a random value; separate runs provide distinct session and guest-key secrets. The `.env` file remains private. A stable guest-key pepper across restarts allows previously issued keys to remain verifiable.

The MySQL account needs the permissions used by initialization: creating the database and tables, adding compatibility columns, and reading/writing application records. The initializer preserves existing records. A new administrator is inserted only when the configured username is absent; changing `FG_ADMIN_PASSWORD` does **not** reset an existing account's password.

## Prepare models and start

With MySQL running:

```powershell
python prepare_arcface.py
python tools/check_model.py
python app.py
```

ArcFace is loaded through DeepFace. Its first preparation may download `arcface_weights.h5`; subsequent runs use the cached weights. No enrolment data is needed for this preparation.

The included `mahalanobis_model_full_500.npz` is loaded from the project root unless `MAHALANOBIS_MODEL_PATH` specifies another file. The check utility examines the standard root path. If the model is absent, the application uses an identity-covariance demonstration fallback; results from that fallback are not equivalent to results using the supplied model.

Use `python app.py` for first startup: it initializes the schema and administrator before launching Flask. Importing the app through another server command does not execute that initialization block. `database.sql` is a schema reference; manual import is not required for normal startup and does not create the administrator.

Open <http://127.0.0.1:5000>, sign in, allow camera access, and capture all four enrolment views before starting recognition. Guest-key entry is available at <http://127.0.0.1:5000/guest>. The `/health` endpoint reports Flask/MySQL status; it does not validate ArcFace readiness or recognition accuracy.

## Tests and existing enrolments

```powershell
python -m pytest -q
```

The packaged tests cover fusion mathematics, temporal voting and guest-key helpers without requiring a webcam or running MySQL. They do not measure biometric accuracy.

An existing database containing MobileNetV2 templates needs migration before ArcFace recognition. Back up the database and retain the original private enrolment images, then run:

```powershell
python rebuild_arcface_templates.py --dry-run
python rebuild_arcface_templates.py
```

The dry run checks image paths but also invokes database initialization, which may create schema, compatibility columns or the first administrator. It does not rewrite feature templates. The full rebuild creates a timestamped enrolment-table backup and replaces feature blobs while preserving user/sample identifiers and image paths. A fresh installation can enrol new users directly.
