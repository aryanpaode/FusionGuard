from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm

_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.45)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _point(landmarks, index: int, width: int, height: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x * width, lm.y * height], dtype=np.float64)


def _ear(landmarks, indices, width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = [_point(landmarks, idx, width, height) for idx in indices]
    horizontal = norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((norm(p2 - p6) + norm(p3 - p5)) / (2.0 * horizontal))


def assess_liveness(images: list[np.ndarray]) -> dict:
    """Simple blink/head-motion challenge over frames separate from recognition frames."""
    ears = []
    nose_offsets = []
    valid_frames = 0

    for image in images:
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        results = _face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            continue
        valid_frames += 1
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = _ear(landmarks, LEFT_EYE, width, height)
        right_ear = _ear(landmarks, RIGHT_EYE, width, height)
        ears.append((left_ear + right_ear) / 2.0)

        left_eye_center = (_point(landmarks, 33, width, height) + _point(landmarks, 133, width, height)) / 2.0
        right_eye_center = (_point(landmarks, 362, width, height) + _point(landmarks, 263, width, height)) / 2.0
        eye_mid = (left_eye_center + right_eye_center) / 2.0
        inter_eye = norm(right_eye_center - left_eye_center)
        nose = _point(landmarks, 1, width, height)
        offset = (nose[0] - eye_mid[0]) / inter_eye if inter_eye else 0.0
        nose_offsets.append(float(offset))

    if valid_frames < 3:
        return {
            "passed": False,
            "reason": "Not enough liveness frames contained a detectable face.",
            "valid_frames": valid_frames,
            "blink_detected": False,
            "head_movement_detected": False,
        }

    ear_range = max(ears) - min(ears) if ears else 0.0
    blink_detected = bool(ears and min(ears) < 0.23 and ear_range >= 0.045)
    head_range = max(nose_offsets) - min(nose_offsets) if nose_offsets else 0.0
    head_movement_detected = head_range >= 0.16
    passed = blink_detected or head_movement_detected

    return {
        "passed": passed,
        "reason": "Liveness challenge passed." if passed else "Blink or turn your head slightly and try again.",
        "valid_frames": valid_frames,
        "blink_detected": blink_detected,
        "head_movement_detected": head_movement_detected,
        "ear_range": round(float(ear_range), 4),
        "head_motion_range": round(float(head_range), 4),
    }
