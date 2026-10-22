from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from config import BLUR_WARNING_VARIANCE, MAX_BRIGHTNESS, MIN_BLUR_VARIANCE, MIN_BRIGHTNESS, MIN_FACE_WIDTH_RATIO

_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.45)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_frame(image: np.ndarray) -> dict:
    """Evaluate webcam usability without over-rejecting normal consumer webcams.

    V3.3 measures sharpness primarily inside the detected face rather than across the
    whole scene. A mildly soft frame is returned as a warning, not a hard rejection;
    recognition confidence and temporal voting still provide the actual identity gate.
    """
    if image is None or image.size == 0:
        return {"ok": False, "reason": "Empty image", "warning": "", "face_detected": False}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_blur_variance = _laplacian_variance(gray)
    brightness = float(np.mean(gray))
    height, width = image.shape[:2]
    results = _face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return {
            "ok": False,
            "reason": "No face detected",
            "warning": "",
            "face_detected": False,
            "blur_variance": round(full_blur_variance, 2),
            "face_blur_variance": 0.0,
            "brightness": round(brightness, 2),
            "face_width_ratio": 0.0,
        }

    landmarks = results.multi_face_landmarks[0].landmark
    xs = [float(lm.x) for lm in landmarks]
    ys = [float(lm.y) for lm in landmarks]
    face_width_ratio = max(xs) - min(xs)

    x0 = max(0, int(min(xs) * width))
    x1 = min(width, int(max(xs) * width))
    y0 = max(0, int(min(ys) * height))
    y1 = min(height, int(max(ys) * height))

    # Add a small margin so hairline/eyes/nose/mouth edges contribute to the score.
    pad_x = int(max(4, (x1 - x0) * 0.12))
    pad_y = int(max(4, (y1 - y0) * 0.12))
    x0 = max(0, x0 - pad_x)
    x1 = min(width, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(height, y1 + pad_y)
    face_gray = gray[y0:y1, x0:x1]
    face_blur_variance = _laplacian_variance(face_gray)

    reasons = []
    warnings = []

    # A hard blur rejection is deliberately conservative. Ordinary webcam softness is
    # advisory because the multimodal matcher and temporal vote remain stricter gates.
    if face_blur_variance < MIN_BLUR_VARIANCE:
        reasons.append("face is too blurred for reliable capture")
    elif face_blur_variance < BLUR_WARNING_VARIANCE:
        warnings.append("image is slightly soft")

    if brightness < MIN_BRIGHTNESS:
        reasons.append("lighting is too dark")
    if brightness > MAX_BRIGHTNESS:
        reasons.append("lighting is too bright")
    if face_width_ratio < MIN_FACE_WIDTH_RATIO:
        reasons.append("face is too small")

    return {
        "ok": not reasons,
        "reason": "Good quality" if not reasons else ", ".join(reasons),
        "warning": ", ".join(warnings),
        "face_detected": True,
        "blur_variance": round(full_blur_variance, 2),
        "face_blur_variance": round(face_blur_variance, 2),
        "brightness": round(brightness, 2),
        "face_width_ratio": round(float(face_width_ratio), 3),
        "image_width": width,
        "image_height": height,
    }
