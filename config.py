from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Flask / security
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "fusionguard-v3-change-before-deployment")
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "fusionguard_multimodal")

# First-run administrator
DEFAULT_ADMIN_USERNAME = os.getenv("FG_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("FG_ADMIN_PASSWORD", "")

# Eight-digit guest key hashing pepper
GUEST_KEY_PEPPER = os.getenv("FG_GUEST_KEY_PEPPER", "fusionguard-demo-pepper")

# FusionGuard V3.5 identity model
IDENTITY_MODEL_NAME = "ArcFace"
ARCFACE_WEIGHT = float(os.getenv("FUSION_ARCFACE_WEIGHT", os.getenv("FUSION_CNN_WEIGHT", "0.75")))
GEOMETRY_WEIGHT = float(os.getenv("FUSION_GEOMETRY_WEIGHT", "0.15"))
MAHALANOBIS_WEIGHT = float(os.getenv("FUSION_MAHALANOBIS_WEIGHT", "0.07"))
IRIS_WEIGHT = float(os.getenv("FUSION_IRIS_WEIGHT", "0.03"))

# DeepFace currently uses a pre-tuned ArcFace cosine-distance threshold of 0.68.
# FusionGuard keeps this explicit so it can later be calibrated on local data.
ARCFACE_COSINE_DISTANCE_THRESHOLD = float(os.getenv("FUSION_ARCFACE_COSINE_THRESHOLD", "0.68"))
MIN_ARCFACE_SIMILARITY = float(os.getenv("FUSION_MIN_ARCFACE_SIMILARITY", str(100.0 - 50.0 * ARCFACE_COSINE_DISTANCE_THRESHOLD)))

# Initial fusion calibration for ArcFace. ArcFace's own distance gate remains the
# primary deep-identity gate; these values only map the four-source weighted score
# through the existing logistic layer. Recalibrate later using genuine/impostor data.
LOGISTIC_CENTER = float(os.getenv("FUSION_LOGISTIC_CENTER", "68.0"))
LOGISTIC_SCALE = float(os.getenv("FUSION_LOGISTIC_SCALE", "4.0"))
MATCH_THRESHOLD = float(os.getenv("FUSION_MATCH_THRESHOLD", "0.50"))
MAX_MAHALANOBIS_DISTANCE = float(os.getenv("FUSION_MAX_MAHALANOBIS", "3.0"))
MIN_IDENTITY_MARGIN = float(os.getenv("FUSION_MIN_IDENTITY_MARGIN", "3.0"))

# Temporal recognition
RECOGNITION_FRAME_COUNT = int(os.getenv("FG_RECOGNITION_FRAMES", "5"))
MIN_USABLE_RECOGNITION_FRAMES = int(os.getenv("FG_MIN_USABLE_FRAMES", "3"))
LIVENESS_FRAME_COUNT = int(os.getenv("FG_LIVENESS_FRAMES", "4"))
COOLDOWN_SECONDS = int(os.getenv("FG_COOLDOWN_SECONDS", "10"))
STABLE_PROBE_COUNT = int(os.getenv("FG_STABLE_PROBE_COUNT", "3"))

# Quality thresholds
MIN_BLUR_VARIANCE = float(os.getenv("FG_MIN_BLUR_VARIANCE", "3.0"))
BLUR_WARNING_VARIANCE = float(os.getenv("FG_BLUR_WARNING_VARIANCE", "18.0"))
MIN_BRIGHTNESS = float(os.getenv("FG_MIN_BRIGHTNESS", "45.0"))
MAX_BRIGHTNESS = float(os.getenv("FG_MAX_BRIGHTNESS", "215.0"))
MIN_FACE_WIDTH_RATIO = float(os.getenv("FG_MIN_FACE_WIDTH_RATIO", "0.22"))

ANGLE_FIELDS = {
    "front": "Front View",
    "left": "Slight Left",
    "right": "Slight Right",
    "up": "Slight Up",
}

UPLOAD_ROOT = BASE_DIR / "static" / "uploads"
ENROLLMENT_ROOT = UPLOAD_ROOT / "enrollment"
LIVE_ROOT = UPLOAD_ROOT / "live"
INCIDENT_ROOT = UPLOAD_ROOT / "incidents"
MAHALANOBIS_MODEL_PATH = Path(os.getenv("MAHALANOBIS_MODEL_PATH", str(BASE_DIR / "mahalanobis_model_full_500.npz")))

for directory in (ENROLLMENT_ROOT, LIVE_ROOT, INCIDENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

weights_sum = ARCFACE_WEIGHT + GEOMETRY_WEIGHT + MAHALANOBIS_WEIGHT + IRIS_WEIGHT
if abs(weights_sum - 1.0) > 1e-9:
    raise ValueError("FusionGuard V3.5 modality weights must add to 1.0.")
if not 0.0 < ARCFACE_COSINE_DISTANCE_THRESHOLD < 2.0:
    raise ValueError("FUSION_ARCFACE_COSINE_THRESHOLD must be between 0 and 2.")
if LOGISTIC_SCALE <= 0:
    raise ValueError("FUSION_LOGISTIC_SCALE must be greater than zero.")
if RECOGNITION_FRAME_COUNT < 3:
    raise ValueError("At least 3 recognition frames are required for temporal voting.")
