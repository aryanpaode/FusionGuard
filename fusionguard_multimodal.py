"""FusionGuard V3.5 multimodal recognition engine with ArcFace identity embeddings.

Identity evidence:
    ArcFace 75% + periocular geometry 15% + Mahalanobis-derived evidence 7%
    + iris/LBP 3% -> logistic sigmoid.

ArcFace is provided through DeepFace and is loaded lazily. Existing MobileNetV2
feature blobs are deliberately rejected; use rebuild_arcface_templates.py after
restoring the original enrolment images so legacy identities are not silently
compared in incompatible embedding spaces.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, mahalanobis
from skimage.feature import local_binary_pattern
from config import ARCFACE_COSINE_DISTANCE_THRESHOLD, IDENTITY_MODEL_NAME, MAHALANOBIS_MODEL_PATH, MATCH_THRESHOLD, MAX_MAHALANOBIS_DISTANCE
from fusion_math import arcface_similarity_percent, weighted_fusion_score

ImageInput = Union[str, Path, np.ndarray]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.50)


def load_image(image_input: ImageInput) -> np.ndarray:
    image = image_input.copy() if isinstance(image_input, np.ndarray) else cv2.imread(str(image_input))
    if image is None or image.size == 0:
        raise ValueError(f"Could not load image: {image_input}")
    return image


def landmark_point(landmarks, index: int, width: int, height: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)


def align_complete_face(image: np.ndarray) -> tuple[np.ndarray, object]:
    height, width = image.shape[:2]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face detected. Keep the complete face inside the frame.")

    landmarks = results.multi_face_landmarks[0].landmark
    left_eye_center = (landmark_point(landmarks, 33, width, height) + landmark_point(landmarks, 133, width, height)) / 2.0
    right_eye_center = (landmark_point(landmarks, 362, width, height) + landmark_point(landmarks, 263, width, height)) / 2.0
    eye_vector = right_eye_center - left_eye_center
    angle = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
    center = tuple(((left_eye_center + right_eye_center) / 2.0).tolist())

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    aligned_results = face_mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    if not aligned_results.multi_face_landmarks:
        raise ValueError("Face alignment failed. Retake the image with less head tilt.")
    return aligned, aligned_results.multi_face_landmarks[0].landmark


def crop_complete_face(aligned_image: np.ndarray, landmarks) -> np.ndarray:
    height, width = aligned_image.shape[:2]
    x_values = np.array([landmark.x * width for landmark in landmarks], dtype=np.float64)
    y_values = np.array([landmark.y * height for landmark in landmarks], dtype=np.float64)
    min_x, max_x = float(x_values.min()), float(x_values.max())
    min_y, max_y = float(y_values.min()), float(y_values.max())
    detected_width, detected_height = max_x - min_x, max_y - min_y
    x_padding = 0.18 * detected_width
    top_padding = 0.28 * detected_height
    bottom_padding = 0.14 * detected_height
    x1, x2 = max(0, int(min_x - x_padding)), min(width, int(max_x + x_padding))
    y1, y2 = max(0, int(min_y - top_padding)), min(height, int(max_y + bottom_padding))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid complete-face crop.")
    full_face = aligned_image[y1:y2, x1:x2]
    if full_face.shape[0] < 90 or full_face.shape[1] < 90:
        raise ValueError("Face is too small. Move closer to the webcam.")
    return full_face


def get_angle_degrees(point1: np.ndarray, point2: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0])))


def extract_periocular_geometry(aligned_image: np.ndarray, landmarks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = aligned_image.shape[:2]

    def point(index: int) -> np.ndarray:
        return landmark_point(landmarks, index, width, height)

    left_eye = [33, 133, 159, 145, 468]
    right_eye = [362, 263, 386, 374, 473]
    left_width = norm(point(left_eye[0]) - point(left_eye[1]))
    left_height = norm(point(left_eye[2]) - point(left_eye[3]))
    right_width = norm(point(right_eye[0]) - point(right_eye[1]))
    right_height = norm(point(right_eye[2]) - point(right_eye[3]))
    left_iris_inner = norm(point(468) - point(left_eye[0]))
    left_iris_outer = norm(point(468) - point(left_eye[1]))
    right_iris_inner = norm(point(473) - point(right_eye[0]))
    right_iris_outer = norm(point(473) - point(right_eye[1]))
    iris_ratio_left = left_iris_inner / left_iris_outer if left_iris_outer else 0.0
    iris_ratio_right = right_iris_inner / right_iris_outer if right_iris_outer else 0.0
    average_iris_ratio = (iris_ratio_left + iris_ratio_right) / 2.0
    average_height_width_ratio = ((left_height / left_width if left_width else 0.0) + (right_height / right_width if right_width else 0.0)) / 2.0
    left_cross_denominator = norm(point(left_eye[1]) - point(left_eye[3]))
    right_cross_denominator = norm(point(right_eye[1]) - point(right_eye[3]))
    left_cross = norm(point(left_eye[0]) - point(left_eye[2])) / left_cross_denominator if left_cross_denominator else 0.0
    right_cross = norm(point(right_eye[0]) - point(right_eye[2])) / right_cross_denominator if right_cross_denominator else 0.0
    average_cross_ratio = (left_cross + right_cross) / 2.0
    slant_angle_difference = abs(get_angle_degrees(point(left_eye[0]), point(left_eye[1])) - get_angle_degrees(point(right_eye[0]), point(right_eye[1])))
    inner_distance = norm(point(left_eye[0]) - point(right_eye[0]))
    outer_distance = norm(point(left_eye[1]) - point(right_eye[1]))
    center_ratio = inner_distance / outer_distance if outer_distance else 0.0

    eye_indices = [33, 133, 159, 145, 468, 362, 263, 386, 374, 473, 105, 334]
    eye_x = [point(index)[0] for index in eye_indices]
    eye_y = [point(index)[1] for index in eye_indices]
    horizontal_padding = max(12, int(0.08 * outer_distance))
    vertical_padding = max(22, int(0.13 * outer_distance))
    min_x, max_x = max(0, int(min(eye_x) - horizontal_padding)), min(width, int(max(eye_x) + horizontal_padding))
    min_y, max_y = max(0, int(min(eye_y) - vertical_padding)), min(height, int(max(eye_y) + vertical_padding))
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Periocular crop is outside the image.")
    eye_crop = aligned_image[min_y:max_y, min_x:max_x]
    if eye_crop.size == 0:
        raise ValueError("Periocular region could not be extracted.")

    eye_y_alignment = abs(point(left_eye[0])[1] - point(right_eye[0])[1])
    left_brow, right_brow = point(105), point(334)
    left_eye_center = (point(left_eye[0]) + point(left_eye[1])) / 2.0
    right_eye_center = (point(right_eye[0]) + point(right_eye[1])) / 2.0
    left_brow_gap = abs(left_brow[1] - left_eye_center[1])
    right_brow_gap = abs(right_brow[1] - right_eye_center[1])
    average_brow_gap = (left_brow_gap + right_brow_gap) / 2.0
    inter_brow_distance = norm(left_brow - right_brow)
    inter_brow_ratio = inter_brow_distance / inner_distance if inner_distance else 0.0

    raw_geometry = np.array([
        outer_distance, center_ratio, average_height_width_ratio, average_cross_ratio,
        average_iris_ratio, slant_angle_difference, eye_y_alignment, average_brow_gap,
        inter_brow_ratio,
    ], dtype=np.float64)
    scale_sensitive = np.array([outer_distance, eye_y_alignment, average_brow_gap], dtype=np.float64)
    normalized_scale_values = scale_sensitive / inner_distance if inner_distance else scale_sensitive
    earlier_geometry_definition = np.concatenate((normalized_scale_values, raw_geometry[2:]))
    geometry_norm = norm(earlier_geometry_definition)
    normalized_geometry = earlier_geometry_definition / geometry_norm if geometry_norm else earlier_geometry_definition
    raw_with_inner_distance = np.append(raw_geometry, inner_distance)
    return eye_crop, normalized_geometry, raw_with_inner_distance


def _deepface_module():
    try:
        from deepface import DeepFace
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ArcFace dependency is not installed. Run: python -m pip install deepface==0.0.101 tf-keras"
        ) from exc
    return DeepFace


def warmup_arcface() -> None:
    """Load ArcFace once (and download its weights on the first run if needed)."""
    DeepFace = _deepface_module()
    DeepFace.build_model(task="facial_recognition", model_name="ArcFace")


def get_arcface_embedding(full_face: np.ndarray) -> np.ndarray:
    DeepFace = _deepface_module()
    try:
        representations = DeepFace.represent(
            img_path=full_face,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="ArcFace",
            l2_normalize=True,
        )
    except Exception as exc:
        raise RuntimeError(f"ArcFace embedding failed: {exc}") from exc
    if not representations:
        raise RuntimeError("ArcFace did not return an embedding.")
    embedding = np.asarray(representations[0]["embedding"], dtype=np.float64).reshape(-1)
    embedding_norm = norm(embedding)
    if embedding_norm == 0:
        raise RuntimeError("ArcFace returned a zero-length embedding.")
    return embedding / embedding_norm

def get_iris_lbp_histogram(eye_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    radius, number_of_points = 1, 8
    lbp = local_binary_pattern(resized, number_of_points, radius, method="uniform")
    histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(0, number_of_points + 3), range=(0, number_of_points + 2))
    histogram = histogram.astype(np.float64)
    histogram /= histogram.sum() + 1e-6
    return histogram


def extract_template(image_input: ImageInput) -> Dict[str, np.ndarray]:
    image = load_image(image_input)
    aligned_image, landmarks = align_complete_face(image)
    full_face = crop_complete_face(aligned_image, landmarks)
    eye_crop, geometry, raw_geometry = extract_periocular_geometry(aligned_image, landmarks)
    return {
        "arcface_embedding": get_arcface_embedding(full_face),
        "geometry": geometry,
        "raw_geometry": raw_geometry,
        "iris_histogram": get_iris_lbp_histogram(eye_crop),
    }


def serialize_template(template: Dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, identity_model=np.array([IDENTITY_MODEL_NAME]), arcface_embedding=template["arcface_embedding"], geometry=template["geometry"], raw_geometry=template["raw_geometry"], iris_histogram=template["iris_histogram"])
    return buffer.getvalue()


def deserialize_template(blob: bytes) -> Dict[str, np.ndarray]:
    buffer = io.BytesIO(blob)
    with np.load(buffer, allow_pickle=False) as stored:
        if "arcface_embedding" not in stored.files:
            if "cnn_embedding" in stored.files:
                raise ValueError(
                    "Legacy MobileNetV2 enrolment template detected. Restore the original enrolment images and run python rebuild_arcface_templates.py before recognition."
                )
            raise ValueError("Unsupported enrolment template format.")
        return {
            "arcface_embedding": stored["arcface_embedding"],
            "geometry": stored["geometry"],
            "raw_geometry": stored["raw_geometry"],
            "iris_histogram": stored["iris_histogram"],
        }


def safe_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    if norm(vector1) == 0 or norm(vector2) == 0:
        return 0.0
    return float(np.clip(1.0 - cosine(vector1, vector2), -1.0, 1.0))


def _normalise_raw_geometry(raw_geometry: np.ndarray) -> np.ndarray:
    inner_distance = raw_geometry[-1]
    return raw_geometry[:-1] / inner_distance if inner_distance else raw_geometry[:-1]


@lru_cache(maxsize=1)
def _load_mahalanobis_model():
    if not MAHALANOBIS_MODEL_PATH.exists():
        return None
    with np.load(MAHALANOBIS_MODEL_PATH, allow_pickle=False) as model:
        return model["mean"].copy(), model["inv_cov"].copy(), model["std"].copy()


def calculate_mahalanobis_distance(raw_geometry1: np.ndarray, raw_geometry2: np.ndarray) -> tuple[float, str]:
    normalized1 = _normalise_raw_geometry(raw_geometry1)
    normalized2 = _normalise_raw_geometry(raw_geometry2)
    model = _load_mahalanobis_model()

    if model is None:
        # Identity covariance is a valid Mahalanobis form and keeps the software
        # demonstrable when the user's separately trained NPZ is not packaged.
        delta = normalized1 - normalized2
        distance = float(np.sqrt(np.dot(delta, delta)))
        return distance, "identity-covariance fallback"

    mean_vector, inverse_covariance, standard_deviation = model
    if normalized1.shape != mean_vector.shape or normalized2.shape != mean_vector.shape:
        raise ValueError("Mahalanobis model dimensionality does not match the stored geometry template.")
    safe_std = np.where(standard_deviation == 0, 1.0, standard_deviation)
    centered1 = (normalized1 - mean_vector) / safe_std
    centered2 = (normalized2 - mean_vector) / safe_std
    return float(mahalanobis(centered1, centered2, inverse_covariance)), "trained model"


def compare_templates(enrolled_template: Dict[str, np.ndarray], live_template: Dict[str, np.ndarray]) -> Dict[str, object]:
    arcface_cosine = safe_cosine_similarity(enrolled_template["arcface_embedding"], live_template["arcface_embedding"])
    arcface_distance = 1.0 - arcface_cosine
    arcface_percent = arcface_similarity_percent(arcface_cosine)
    geometry1 = enrolled_template["geometry"] - np.mean(enrolled_template["geometry"])
    geometry2 = live_template["geometry"] - np.mean(live_template["geometry"])
    geometry_similarity = safe_cosine_similarity(geometry1, geometry2)
    mahalanobis_distance, mahalanobis_source = calculate_mahalanobis_distance(enrolled_template["raw_geometry"], live_template["raw_geometry"])
    iris_similarity = safe_cosine_similarity(enrolled_template["iris_histogram"], live_template["iris_histogram"])

    fusion_score, probability, mahalanobis_evidence = weighted_fusion_score(
        arcface_percent,
        geometry_similarity * 100.0,
        mahalanobis_distance,
        iris_similarity * 100.0,
    )

    mahalanobis_pass = mahalanobis_distance <= MAX_MAHALANOBIS_DISTANCE
    logistic_pass = probability >= MATCH_THRESHOLD
    arcface_pass = arcface_distance <= ARCFACE_COSINE_DISTANCE_THRESHOLD
    decision = "Match" if logistic_pass and mahalanobis_pass and arcface_pass else "No Match"

    return {
        "identity_model": IDENTITY_MODEL_NAME,
        "arcface_similarity": round(arcface_percent, 3),
        "arcface_cosine": round(arcface_cosine, 6),
        "arcface_distance": round(arcface_distance, 6),
        "arcface_threshold": ARCFACE_COSINE_DISTANCE_THRESHOLD,
        # Compatibility alias for the existing recognition_frame_results.cnn_similarity column.
        "cnn_similarity": round(arcface_percent, 3),
        "geometry_similarity": round(geometry_similarity * 100.0, 3),
        "mahalanobis_distance": round(mahalanobis_distance, 3),
        "mahalanobis_evidence": round(mahalanobis_evidence, 3),
        "mahalanobis_source": mahalanobis_source,
        "iris_similarity": round(iris_similarity * 100.0, 3),
        "fusion_score": round(fusion_score, 3),
        "logistic_probability": round(probability, 6),
        "mahalanobis_pass": mahalanobis_pass,
        "arcface_pass": arcface_pass,
        "decision": decision,
    }
