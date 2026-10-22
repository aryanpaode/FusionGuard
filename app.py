from __future__ import annotations

import base64
import json
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime
from decimal import Decimal
from functools import wraps
from pathlib import Path

import cv2
import mysql.connector
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from mysql.connector import Error
from werkzeug.security import check_password_hash

import db
from config import (
    ANGLE_FIELDS,
    COOLDOWN_SECONDS,
    ENROLLMENT_ROOT,
    INCIDENT_ROOT,
    LIVENESS_FRAME_COUNT,
    LIVE_ROOT,
    MATCH_THRESHOLD,
    ARCFACE_COSINE_DISTANCE_THRESHOLD,
    IDENTITY_MODEL_NAME,
    MAX_MAHALANOBIS_DISTANCE,
    MIN_ARCFACE_SIMILARITY,
    MIN_IDENTITY_MARGIN,
    MIN_USABLE_RECOGNITION_FRAMES,
    RECOGNITION_FRAME_COUNT,
    SECRET_KEY,
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_SAMESITE,
    STABLE_PROBE_COUNT,
)
from fusion_math import FusionWeights
from fusionguard_multimodal import compare_templates, deserialize_template, extract_template, serialize_template
from guest_keys import create_guest_key, verify_guest_key
from liveness import assess_liveness
from quality import evaluate_frame
from temporal_verifier import finalise_temporal_decision

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = SESSION_COOKIE_HTTPONLY
app.config["SESSION_COOKIE_SAMESITE"] = SESSION_COOKIE_SAMESITE
# Four Base64 webcam enrolment images can exceed Werkzeug's conservative form limit.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 20 * 1024 * 1024


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Administrator login required."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def decode_data_url(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid webcam image data.")
    header, encoded = data_url.split(",", 1)
    if "image/" not in header:
        raise ValueError("Captured data is not an image.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Webcam image could not be decoded.") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Webcam image could not be read by OpenCV.")
    return image


def save_image(image: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise ValueError("Image could not be saved.")


def relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(Path(__file__).resolve().parent).as_posix()


def web_static_path(path: str | None) -> str | None:
    if not path:
        return None
    return path[7:] if path.startswith("static/") else path


def _json_safe(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _incident_api_row(row: dict) -> dict:
    item = _json_safe(dict(row))
    evidence_path = web_static_path(item.get("evidence_path"))
    item["evidence_url"] = url_for("static", filename=evidence_path) if evidence_path else None
    return item


def _cooldown_remaining() -> int:
    until = float(session.get("recognition_cooldown_until", 0.0) or 0.0)
    remaining = max(0.0, until - time.time())
    return int(np.ceil(remaining)) if remaining else 0


def _start_cooldown() -> None:
    session["recognition_cooldown_until"] = time.time() + COOLDOWN_SECONDS


def _rank_frame_against_users(live_template: dict, enrolled_samples: list[dict], frame_index: int) -> dict:
    results_by_user: dict[int, dict] = {}

    for sample in enrolled_samples:
        enrolled_template = deserialize_template(sample["feature_template"])
        scores = compare_templates(enrolled_template, live_template)
        candidate = {**sample, **scores}
        current_best = results_by_user.get(sample["user_id"])
        if current_best is None or candidate["fusion_score"] > current_best["fusion_score"]:
            results_by_user[sample["user_id"]] = candidate

    if not results_by_user:
        raise ValueError("No active enrolment templates are available.")

    ranked = sorted(results_by_user.values(), key=lambda item: item["fusion_score"], reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    identity_margin = best["fusion_score"] - runner_up["fusion_score"] if runner_up is not None else 100.0
    margin_pass = runner_up is None or identity_margin >= MIN_IDENTITY_MARGIN
    frame_match = best["decision"] == "Match" and margin_pass
    if frame_match:
        rejection_reason = None
    elif not best.get("arcface_pass", False):
        rejection_reason = f"ArcFace cosine distance {best['arcface_distance']:.3f} exceeds {ARCFACE_COSINE_DISTANCE_THRESHOLD:.3f}."
    elif not best.get("mahalanobis_pass", False):
        rejection_reason = f"Mahalanobis distance {best['mahalanobis_distance']:.3f} exceeds {MAX_MAHALANOBIS_DISTANCE:.3f}."
    elif float(best.get("logistic_probability", 0.0)) < MATCH_THRESHOLD:
        rejection_reason = f"Fusion confidence {float(best.get('logistic_probability', 0.0)) * 100:.1f}% is below {MATCH_THRESHOLD * 100:.1f}%."
    elif not margin_pass:
        rejection_reason = f"Identity margin {identity_margin:.3f} is below {MIN_IDENTITY_MARGIN:.3f}."
    else:
        rejection_reason = "Frame did not satisfy the multimodal match gates."

    return {
        "frame_index": frame_index,
        "candidate_user_id": best["user_id"],
        "candidate_name": best["user_name"],
        "best_angle_label": best["angle_label"],
        "arcface_similarity": best["arcface_similarity"],
        "arcface_distance": best["arcface_distance"],
        # Compatibility alias retained for the existing MySQL frame-results column.
        "cnn_similarity": best["arcface_similarity"],
        "geometry_similarity": best["geometry_similarity"],
        "mahalanobis_distance": best["mahalanobis_distance"],
        "mahalanobis_evidence": best["mahalanobis_evidence"],
        "mahalanobis_source": best["mahalanobis_source"],
        "iris_similarity": best["iris_similarity"],
        "fusion_score": best["fusion_score"],
        "logistic_probability": best["logistic_probability"],
        "identity_margin": round(float(identity_margin), 3),
        "runner_up_name": runner_up["user_name"] if runner_up else None,
        "runner_up_score": runner_up["fusion_score"] if runner_up else None,
        "frame_decision": "Match" if frame_match else ("Ambiguous" if best["decision"] == "Match" else "Unknown"),
        "frame_identity": best["user_name"] if frame_match else None,
        "rejection_reason": rejection_reason,
    }


def _result_payload(result: dict, frame_results: list[dict], liveness: dict, session_id: int, evidence_path: str | None, incident_id: int | None) -> dict:
    supporting = result.get("supporting_frames") or []
    reference_frames = supporting if supporting else frame_results

    def med(key: str) -> float:
        values = [float(item.get(key, 0.0)) for item in reference_frames]
        return round(float(np.median(values)), 3) if values else 0.0

    angle_counts = Counter(item.get("best_angle_label") for item in supporting if item.get("best_angle_label"))
    best_angle = angle_counts.most_common(1)[0][0] if angle_counts else (frame_results[0].get("best_angle_label") if frame_results else None)

    return {
        "ok": True,
        "session_id": session_id,
        "decision": result["decision"],
        "winner_name": result.get("winner_name"),
        "winner_votes": result.get("winner_votes", 0),
        "usable_frames": result.get("usable_frames", 0),
        "required_votes": result.get("required_votes", 0),
        "median_confidence": round(float(result.get("median_confidence", 0.0)), 6),
        "median_fusion_score": round(float(result.get("median_fusion_score", 0.0)), 3),
        "median_identity_margin": round(float(result.get("median_identity_margin", 0.0)), 3),
        "identity_model": IDENTITY_MODEL_NAME,
        "best_angle_label": best_angle,
        "arcface_similarity": med("arcface_similarity"),
        "arcface_distance": med("arcface_distance"),
        "geometry_similarity": med("geometry_similarity"),
        "mahalanobis_distance": med("mahalanobis_distance"),
        "mahalanobis_evidence": med("mahalanobis_evidence"),
        "iris_similarity": med("iris_similarity"),
        "liveness": liveness,
        "vote_counts": result.get("vote_counts", {}),
        "evidence_path": web_static_path(evidence_path),
        "incident_id": incident_id,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "alert": result["decision"] in {"Unknown", "Uncertain", "Suspected Spoof"},
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        admin = db.fetch_admin_by_username(username)
        if admin and check_password_hash(admin["password_hash"], password):
            session.clear()
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            db.audit(admin["id"], "login", "Administrator logged in")
            return redirect(url_for("dashboard"))
        flash("Invalid administrator username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    admin_id = session.get("admin_id")
    if admin_id:
        db.audit(admin_id, "logout", "Administrator logged out")
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    decision_filter = request.args.get("decision", "").strip()
    name_filter = request.args.get("name", "").strip()
    incident_filter = request.args.get("incident_status", "").strip()

    return render_template(
        "dashboard.html",
        users=db.fetch_users(include_inactive=True),
        sessions=db.fetch_sessions(limit=75, decision=decision_filter, name=name_filter),
        incidents=db.fetch_incidents(status=incident_filter),
        guest_keys=db.fetch_guest_keys(),
        guest_access_logs=db.fetch_guest_access_logs(),
        metrics=db.dashboard_metrics(),
        decision_filter=decision_filter,
        name_filter=name_filter,
        incident_filter=incident_filter,
        angle_fields=ANGLE_FIELDS,
        recognition_frame_count=RECOGNITION_FRAME_COUNT,
        liveness_frame_count=LIVENESS_FRAME_COUNT,
        stable_probe_count=STABLE_PROBE_COUNT,
        cooldown_seconds=COOLDOWN_SECONDS,
        fusion_weights=FusionWeights().as_percent_text(),
        identity_model=IDENTITY_MODEL_NAME,
        max_mahalanobis=MAX_MAHALANOBIS_DISTANCE,
        min_arcface=MIN_ARCFACE_SIMILARITY,
        arcface_distance_threshold=ARCFACE_COSINE_DISTANCE_THRESHOLD,
        min_margin=MIN_IDENTITY_MARGIN,
    )


@app.route("/enrol", methods=["POST"])
@login_required
def enrol():
    name = request.form.get("name", "").strip()
    role = request.form.get("role", "Authorised User").strip() or "Authorised User"
    if not name:
        flash("Enter the authorised user's name.")
        return redirect(url_for("dashboard"))

    captured = {key: request.form.get(f"{key}_image", "").strip() for key in ANGLE_FIELDS}
    missing = [ANGLE_FIELDS[key] for key, value in captured.items() if not value]
    if missing:
        flash("Capture all four enrolment views: " + ", ".join(missing))
        return redirect(url_for("dashboard"))

    folder = ENROLLMENT_ROOT / f"{uuid.uuid4().hex[:10]}_{datetime.now():%Y%m%d_%H%M%S}"
    samples = []
    try:
        for key, label in ANGLE_FIELDS.items():
            image = decode_data_url(captured[key])
            quality = evaluate_frame(image)
            if not quality["ok"]:
                raise ValueError(f"{label} quality rejected: {quality['reason']}")
            image_path = folder / f"{key}.jpg"
            save_image(image, image_path)
            template = extract_template(image)
            samples.append({
                "angle_key": key,
                "angle_label": label,
                "image_path": relative_path(image_path),
                "feature_blob": serialize_template(template),
                "identity_model": IDENTITY_MODEL_NAME,
            })
        user_id = db.insert_user_with_samples(name, role, samples)
        db.audit(session.get("admin_id"), "enrol_user", f"Enrolled {name} as user #{user_id} with four views")
        flash(f"{name} was enrolled successfully using four webcam angles.")
    except Error as exc:
        shutil.rmtree(folder, ignore_errors=True)
        flash("MySQL enrolment failed: " + str(exc))
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        flash("Enrolment rejected: " + str(exc))
    return redirect(url_for("dashboard"))


@app.route("/user/<int:user_id>/toggle", methods=["POST"])
@login_required
def toggle_user(user_id: int):
    active = request.form.get("active") == "1"
    db.set_user_active(user_id, active)
    db.audit(session.get("admin_id"), "user_status", f"User #{user_id} active={active}")
    flash("Authorised-user status updated.")
    return redirect(url_for("dashboard"))


@app.route("/api/probe", methods=["POST"])
@login_required
def probe():
    payload = request.get_json(silent=True) or {}
    try:
        image = decode_data_url(payload.get("image", ""))
        quality = evaluate_frame(image)
        return jsonify({"ok": True, "ready": bool(quality["ok"]), "quality": quality})
    except Exception as exc:
        return jsonify({"ok": False, "ready": False, "error": str(exc)}), 400


@app.route("/api/monitor/recognize", methods=["POST"])
@login_required
def monitor_recognize():
    remaining = _cooldown_remaining()
    if remaining:
        return jsonify({"ok": False, "cooldown": True, "remaining": remaining, "error": "Recognition cooldown is active."}), 429

    payload = request.get_json(silent=True) or {}
    liveness_data = payload.get("liveness_frames") or []
    recognition_data = payload.get("recognition_frames") or []
    if len(liveness_data) < 3:
        return jsonify({"ok": False, "error": "At least three separate liveness frames are required."}), 400
    if len(recognition_data) < MIN_USABLE_RECOGNITION_FRAMES:
        return jsonify({"ok": False, "error": "Not enough recognition frames were submitted."}), 400

    try:
        liveness_images = [decode_data_url(item) for item in liveness_data]
        liveness = assess_liveness(liveness_images)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Liveness processing failed: {exc}"}), 400

    if not liveness["passed"]:
        evidence_file = INCIDENT_ROOT / f"spoof_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        save_image(liveness_images[0], evidence_file)
        result = {
            "identity_model": IDENTITY_MODEL_NAME,
            "decision": "Suspected Spoof",
            "winner_name": None,
            "winner_user_id": None,
            "winner_votes": 0,
            "usable_frames": 0,
            "required_votes": 0,
            "median_confidence": 0.0,
            "median_fusion_score": 0.0,
            "median_identity_margin": 0.0,
            "supporting_frames": [],
            "vote_counts": {},
        }
        session_id = db.insert_recognition_session(result, relative_path(evidence_file), liveness, COOLDOWN_SECONDS)
        incident_id = db.create_incident(session_id, "Liveness challenge failed; possible presentation/spoof attempt.", "high", relative_path(evidence_file))
        db.insert_alert(incident_id, "security", "FusionGuard raised a suspected-spoof alert.")
        db.audit(session.get("admin_id"), "recognition", f"Session #{session_id}: Suspected Spoof")
        _start_cooldown()
        return jsonify(_result_payload(result, [], liveness, session_id, relative_path(evidence_file), incident_id))

    enrolled_samples = db.fetch_enrollment_templates()
    if not enrolled_samples:
        return jsonify({"ok": False, "error": "No active authorised users are enrolled."}), 400
    legacy_samples = [sample for sample in enrolled_samples if str(sample.get("identity_model") or "").lower() != IDENTITY_MODEL_NAME.lower()]
    if legacy_samples:
        return jsonify({
            "ok": False,
            "template_upgrade_required": True,
            "error": f"{len(legacy_samples)} enrolment template(s) still use the legacy MobileNetV2 embedding. Copy the original enrolment images into static/uploads/enrollment and run: python rebuild_arcface_templates.py",
        }), 409
    try:
        # Fail fast before collecting a five-frame burst if a blob is mislabeled/corrupt.
        for sample in enrolled_samples:
            deserialize_template(sample["feature_template"])
    except Exception as exc:
        return jsonify({"ok": False, "template_upgrade_required": True, "error": str(exc)}), 409

    frame_results = []
    saved_paths: list[Path] = []
    quality_rejections = []

    for index, data_url in enumerate(recognition_data, start=1):
        try:
            image = decode_data_url(data_url)
            quality = evaluate_frame(image)
            if not quality["ok"]:
                quality_rejections.append({"frame": index, "reason": quality["reason"]})
                continue
            image_path = LIVE_ROOT / f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:7]}_f{index}.jpg"
            save_image(image, image_path)
            saved_paths.append(image_path)
            live_template = extract_template(image)
            frame_results.append(_rank_frame_against_users(live_template, enrolled_samples, index))
        except Exception as exc:
            quality_rejections.append({"frame": index, "reason": str(exc)})

    if len(frame_results) < MIN_USABLE_RECOGNITION_FRAMES:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        return jsonify({
            "ok": False,
            "retry": True,
            "error": f"Only {len(frame_results)} usable recognition frames were obtained; {MIN_USABLE_RECOGNITION_FRAMES} are required.",
            "rejections": quality_rejections,
        }), 422

    result = finalise_temporal_decision(frame_results)
    result["identity_model"] = IDENTITY_MODEL_NAME
    if not result.get("winner_name"):
        nearest = Counter(item.get("candidate_name") for item in frame_results if item.get("candidate_name"))
        if nearest:
            result["winner_name"] = nearest.most_common(1)[0][0]
    if result["decision"] == "Authorised":
        supporting = result["supporting_frames"]
        result["winner_user_id"] = supporting[0]["candidate_user_id"] if supporting else None
    else:
        result["winner_user_id"] = None

    evidence_path = relative_path(saved_paths[0]) if saved_paths else None
    session_id = db.insert_recognition_session(result, evidence_path, liveness, COOLDOWN_SECONDS)
    db.insert_frame_results(session_id, frame_results)

    incident_id = None
    if result["decision"] == "Unknown":
        incident_id = db.create_incident(session_id, "No enrolled identity achieved a strict temporal majority.", "high", evidence_path)
        db.insert_alert(incident_id, "unauthorised", "Unknown person detected at the monitored entrance.")
    elif result["decision"] == "Uncertain":
        incident_id = db.create_incident(session_id, "Conflicting or insufficiently consistent identity votes across live frames.", "medium", evidence_path)
        db.insert_alert(incident_id, "uncertain", "Ambiguous recognition requires administrator review.")

    db.audit(session.get("admin_id"), "recognition", f"Session #{session_id}: {result['decision']}; votes={result.get('vote_counts', {})}")
    _start_cooldown()
    return jsonify(_result_payload(result, frame_results, liveness, session_id, evidence_path, incident_id))


@app.route("/api/cooldown/resume", methods=["POST"])
@login_required
def resume_now():
    session["recognition_cooldown_until"] = 0.0
    return jsonify({"ok": True})


@app.route("/api/incidents", methods=["GET"])
@login_required
def api_incidents():
    status = request.args.get("status", "").strip()
    if status not in {"", "open", "investigating", "reviewed", "closed", "false_alarm"}:
        status = ""
    rows = [_incident_api_row(row) for row in db.fetch_incidents(status=status)]
    return jsonify({"ok": True, "incidents": rows, "open_incidents": db.count_open_incidents()})


@app.route("/api/incidents/<int:incident_id>", methods=["GET"])
@login_required
def api_incident_details(incident_id: int):
    detail = db.fetch_incident_details(incident_id)
    if not detail:
        return jsonify({"ok": False, "error": "Incident was not found."}), 404
    payload = _json_safe(detail)
    evidence_path = web_static_path(payload["incident"].get("evidence_path") or payload["incident"].get("session_evidence_path"))
    payload["incident"]["evidence_url"] = url_for("static", filename=evidence_path) if evidence_path else None
    return jsonify({"ok": True, **payload})


@app.route("/api/incidents/<int:incident_id>/review", methods=["POST"])
@login_required
def api_review_incident(incident_id: int):
    payload = request.get_json(silent=True) or {}
    status = str(payload.get("status", "reviewed")).strip()
    if status not in {"open", "investigating", "reviewed", "closed", "false_alarm"}:
        return jsonify({"ok": False, "error": "Invalid incident status."}), 400
    notes = str(payload.get("notes", "")).strip()
    db.review_incident(incident_id, session["admin_id"], status, notes)
    db.audit(session.get("admin_id"), "incident_review", f"Incident #{incident_id} -> {status}")
    return jsonify({"ok": True})


@app.route("/api/incidents/delete", methods=["POST"])
@login_required
def api_delete_incidents():
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("incident_ids") or []
    try:
        incident_ids = [int(value) for value in raw_ids]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Incident IDs must be integers."}), 400
    deleted = db.delete_incidents(incident_ids)
    db.audit(session.get("admin_id"), "incident_delete", f"Deleted {deleted} selected incident log(s) during testing cleanup")
    return jsonify({"ok": True, "deleted": deleted, "open_incidents": db.count_open_incidents()})


@app.route("/incident/<int:incident_id>/review", methods=["POST"])
@login_required
def review_incident(incident_id: int):
    status = request.form.get("status", "reviewed").strip()
    if status not in {"open", "investigating", "reviewed", "closed", "false_alarm"}:
        status = "reviewed"
    notes = request.form.get("notes", "").strip()
    db.review_incident(incident_id, session["admin_id"], status, notes)
    db.audit(session.get("admin_id"), "incident_review", f"Incident #{incident_id} -> {status}")
    flash("Incident review saved.")
    return redirect(url_for("dashboard"))


@app.route("/guest-key/create", methods=["POST"])
@login_required
def create_guest_key_route():
    guest_name = request.form.get("guest_name", "").strip()
    start_text = request.form.get("valid_from", "").strip()
    end_text = request.form.get("valid_until", "").strip()
    max_uses = int(request.form.get("max_uses", "1") or 1)
    if not guest_name or not start_text or not end_text:
        flash("Guest name, start time and expiry time are required.")
        return redirect(url_for("dashboard"))
    try:
        valid_from = datetime.fromisoformat(start_text)
        valid_until = datetime.fromisoformat(end_text)
        key_id, plaintext = create_guest_key(guest_name, valid_from, valid_until, max_uses, session.get("admin_id"))
        session["new_guest_key"] = plaintext
        db.audit(session.get("admin_id"), "guest_key_create", f"Guest key #{key_id} created for {guest_name}; plaintext shown once")
        flash(f"Guest key created for {guest_name}. Copy it now; only the hash is stored after this page.")
    except Exception as exc:
        flash("Guest-key creation failed: " + str(exc))
    return redirect(url_for("dashboard") + "#guest-keys")


@app.route("/guest-key/<int:key_id>/revoke", methods=["POST"])
@login_required
def revoke_guest_key_route(key_id: int):
    db.revoke_guest_key(key_id)
    db.audit(session.get("admin_id"), "guest_key_revoke", f"Revoked guest key #{key_id}")
    flash("Guest key revoked.")
    return redirect(url_for("dashboard") + "#guest-keys")


@app.route("/guest", methods=["GET", "POST"])
def guest_access():
    result = None
    if request.method == "POST":
        result = verify_guest_key(request.form.get("guest_key", ""))
        if result.get("authorised"):
            db.audit(None, "guest_access", f"Guest authorised: {result.get('guest_name') or 'Unnamed guest'}")
        else:
            reason = result.get("reason") or "Guest key rejected."
            guest_name = result.get("guest_name") or "Unknown guest"
            incident_id = db.create_incident(None, f"Rejected guest-key attempt ({guest_name}): {reason}", "medium", None)
            db.insert_alert(incident_id, "guest_access", f"Guest access rejected: {reason}")
            db.audit(None, "guest_access_rejected", f"Incident #{incident_id}: {reason}")
            result["incident_id"] = incident_id
    return render_template("guest.html", result=result)


@app.route("/health")
def health():
    mysql_status = "disconnected"
    try:
        connection = db.get_connection()
        if connection.is_connected():
            mysql_status = "connected"
        connection.close()
    except Error:
        pass
    return {
        "application": "FusionGuard V3.5 ArcFace",
        "identity_model": IDENTITY_MODEL_NAME,
        "flask": "running",
        "mysql": mysql_status,
        "fusion": FusionWeights().as_percent_text(),
        "mahalanobis_ceiling": MAX_MAHALANOBIS_DISTANCE,
        "temporal_decision": "best angle per user per frame -> strict majority -> median supporting-frame confidence",
        "cooldown_seconds": COOLDOWN_SECONDS,
    }


@app.context_processor
def inject_helpers():
    return {"static_path": web_static_path, "new_guest_key": session.pop("new_guest_key", None)}


if __name__ == "__main__":
    try:
        db.initialise_database()
    except Error as exc:
        print("FusionGuard could not initialise MySQL.")
        print("Start MySQL Server and verify MYSQL_USER / MYSQL_PASSWORD.")
        print(f"Details: {exc}")
        raise SystemExit(1)

    app.run(host="127.0.0.1", port=5000, debug=True)
