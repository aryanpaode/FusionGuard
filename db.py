from __future__ import annotations

from datetime import datetime
from typing import Optional

import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash

from config import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER

# V3.2 schema compatibility: foreign keys created by this initializer are left unnamed
# so MySQL generates table-specific names and cannot collide with legacy FusionGuard constraints.


def get_connection(database: Optional[str] = MYSQL_DATABASE):
    settings = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
    }
    if database:
        settings["database"] = database
    return mysql.connector.connect(**settings)


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        """,
        (MYSQL_DATABASE, table_name, column_name),
    )
    return bool(cursor.fetchone()[0])


def _column_type(cursor, table_name: str, column_name: str) -> str:
    cursor.execute(
        """
        SELECT COLUMN_TYPE
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        """,
        (MYSQL_DATABASE, table_name, column_name),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Expected column {table_name}.{column_name} was not found.")
    return str(row[0])


def _primary_key_column(cursor, table_name: str) -> str:
    cursor.execute(
        """
        SELECT k.COLUMN_NAME
        FROM information_schema.table_constraints t
        JOIN information_schema.key_column_usage k
          ON t.CONSTRAINT_NAME=k.CONSTRAINT_NAME
         AND t.TABLE_SCHEMA=k.TABLE_SCHEMA
         AND t.TABLE_NAME=k.TABLE_NAME
        WHERE t.TABLE_SCHEMA=%s AND t.TABLE_NAME=%s AND t.CONSTRAINT_TYPE='PRIMARY KEY'
        ORDER BY k.ORDINAL_POSITION
        LIMIT 1
        """,
        (MYSQL_DATABASE, table_name),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Table {table_name} does not have a primary key.")
    return str(row[0])


def _ensure_column(cursor, table_name: str, column_name: str, definition: str) -> None:
    if not _column_exists(cursor, table_name, column_name):
        cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` {definition}")


def _ensure_legacy_incident_columns(cursor) -> None:
    # Older FusionGuard increments used a smaller incident table. Keep its data
    # and add only the V3 fields that are missing instead of dropping/recreating it.
    _ensure_column(cursor, "incidents", "session_id", "BIGINT NULL")
    _ensure_column(cursor, "incidents", "reason", "VARCHAR(255) NOT NULL DEFAULT 'Legacy incident'")
    _ensure_column(cursor, "incidents", "severity", "VARCHAR(20) NOT NULL DEFAULT 'medium'")
    _ensure_column(cursor, "incidents", "status", "VARCHAR(30) NOT NULL DEFAULT 'open'")
    _ensure_column(cursor, "incidents", "evidence_path", "VARCHAR(500) NULL")
    _ensure_column(cursor, "incidents", "reviewed_by", "INT NULL")
    _ensure_column(cursor, "incidents", "reviewed_at", "DATETIME NULL")
    _ensure_column(cursor, "incidents", "notes", "TEXT NULL")
    _ensure_column(cursor, "incidents", "created_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")


def initialise_database() -> None:
    connection = get_connection(database=None)
    cursor = connection.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        connection.commit()
    finally:
        cursor.close()
        connection.close()

    connection = get_connection()
    cursor = connection.cursor()
    try:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS administrators (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(80) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS authorised_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(120) NOT NULL UNIQUE,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS enrollment_samples (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                angle_key VARCHAR(20) NOT NULL,
                angle_label VARCHAR(50) NOT NULL,
                image_path VARCHAR(500) NOT NULL,
                feature_template LONGBLOB NOT NULL,
                identity_model VARCHAR(40) NOT NULL DEFAULT 'ArcFace',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES authorised_users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                CONSTRAINT uq_user_angle UNIQUE (user_id, angle_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS recognition_sessions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                matched_user_id INT NULL,
                best_candidate_name VARCHAR(120) NULL,
                decision VARCHAR(40) NOT NULL,
                identity_model VARCHAR(40) NOT NULL DEFAULT 'ArcFace',
                usable_frames INT NOT NULL DEFAULT 0,
                winning_votes INT NOT NULL DEFAULT 0,
                required_votes INT NOT NULL DEFAULT 0,
                median_confidence DECIMAL(10,6) NOT NULL DEFAULT 0,
                median_fusion_score DECIMAL(8,3) NOT NULL DEFAULT 0,
                median_identity_margin DECIMAL(8,3) NOT NULL DEFAULT 0,
                evidence_path VARCHAR(500) NULL,
                liveness_result VARCHAR(40) NOT NULL,
                liveness_details TEXT NULL,
                cooldown_seconds INT NOT NULL DEFAULT 10,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (matched_user_id) REFERENCES authorised_users(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS recognition_frame_results (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id BIGINT NOT NULL,
                frame_index INT NOT NULL,
                candidate_user_id INT NULL,
                candidate_name VARCHAR(120) NULL,
                best_angle_label VARCHAR(50) NULL,
                frame_decision VARCHAR(40) NOT NULL,
                cnn_similarity DECIMAL(8,3) NOT NULL DEFAULT 0,
                arcface_distance DECIMAL(10,6) NULL,
                geometry_similarity DECIMAL(8,3) NOT NULL DEFAULT 0,
                mahalanobis_distance DECIMAL(10,3) NOT NULL DEFAULT 0,
                mahalanobis_evidence DECIMAL(8,3) NOT NULL DEFAULT 0,
                iris_similarity DECIMAL(8,3) NOT NULL DEFAULT 0,
                fusion_score DECIMAL(8,3) NOT NULL DEFAULT 0,
                logistic_probability DECIMAL(10,6) NOT NULL DEFAULT 0,
                identity_margin DECIMAL(8,3) NULL,
                rejection_reason VARCHAR(255) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES recognition_sessions(id) ON UPDATE CASCADE ON DELETE CASCADE,
                FOREIGN KEY (candidate_user_id) REFERENCES authorised_users(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id BIGINT NULL,
                reason VARCHAR(255) NOT NULL,
                severity VARCHAR(20) NOT NULL DEFAULT 'medium',
                status VARCHAR(30) NOT NULL DEFAULT 'open',
                evidence_path VARCHAR(500) NULL,
                reviewed_by INT NULL,
                reviewed_at DATETIME NULL,
                notes TEXT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES recognition_sessions(id) ON UPDATE CASCADE ON DELETE SET NULL,
                FOREIGN KEY (reviewed_by) REFERENCES administrators(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS guest_keys (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                key_hash CHAR(64) NOT NULL UNIQUE,
                key_last4 CHAR(4) NOT NULL,
                guest_name VARCHAR(120) NOT NULL,
                valid_from DATETIME NOT NULL,
                valid_until DATETIME NOT NULL,
                max_uses INT NOT NULL DEFAULT 1,
                uses INT NOT NULL DEFAULT 0,
                revoked BOOLEAN NOT NULL DEFAULT FALSE,
                created_by INT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES administrators(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS guest_access_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                guest_key_id BIGINT NULL,
                guest_name VARCHAR(120) NULL,
                provided_last4 CHAR(4) NULL,
                decision VARCHAR(40) NOT NULL,
                reason VARCHAR(255) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (guest_key_id) REFERENCES guest_keys(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                admin_id INT NULL,
                action VARCHAR(120) NOT NULL,
                details TEXT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES administrators(id) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """,
        ]
        for statement in statements:
            cursor.execute(statement)

        # V3 can be opened on top of older FusionGuard databases. In those
        # versions the incident primary key was commonly INT (and in some builds
        # named incident_id) rather than the new BIGINT id. MySQL requires a
        # foreign-key child column to match the parent type exactly, so derive
        # alert_logs.incident_id from the real existing parent schema.
        _ensure_legacy_incident_columns(cursor)
        incident_pk = _primary_key_column(cursor, "incidents")
        incident_pk_type = _column_type(cursor, "incidents", incident_pk)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS alert_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                incident_id {incident_pk_type} NULL,
                alert_type VARCHAR(40) NOT NULL,
                status VARCHAR(30) NOT NULL DEFAULT 'raised',
                message VARCHAR(500) NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (incident_id) REFERENCES incidents(`{incident_pk}`) ON UPDATE CASCADE ON DELETE SET NULL
            )
            """
        )

        if not _column_exists(cursor, "authorised_users", "role"):
            cursor.execute("ALTER TABLE authorised_users ADD COLUMN role VARCHAR(80) NOT NULL DEFAULT 'Authorised User' AFTER name")
        _ensure_column(cursor, "enrollment_samples", "identity_model", "VARCHAR(40) NOT NULL DEFAULT 'MobileNetV2'")
        _ensure_column(cursor, "recognition_sessions", "identity_model", "VARCHAR(40) NOT NULL DEFAULT 'MobileNetV2'")
        _ensure_column(cursor, "recognition_frame_results", "arcface_distance", "DECIMAL(10,6) NULL")
        _ensure_column(cursor, "recognition_frame_results", "rejection_reason", "VARCHAR(255) NULL")

        cursor.execute("SELECT id FROM administrators WHERE username=%s", (DEFAULT_ADMIN_USERNAME,))
        if cursor.fetchone() is None:
            if not DEFAULT_ADMIN_PASSWORD:
                raise ValueError("FG_ADMIN_PASSWORD must be set before creating the first administrator.")
            cursor.execute(
                "INSERT INTO administrators (username, password_hash) VALUES (%s, %s)",
                (DEFAULT_ADMIN_USERNAME, generate_password_hash(DEFAULT_ADMIN_PASSWORD)),
            )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def fetch_admin_by_username(username: str):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM administrators WHERE username=%s AND active=TRUE", (username,))
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def fetch_users(include_inactive: bool = True):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        where = "" if include_inactive else "WHERE u.active=TRUE"
        cursor.execute(
            f"""
            SELECT u.id, u.name, u.role, u.active, u.created_at, COUNT(s.id) AS sample_count
            FROM authorised_users u
            LEFT JOIN enrollment_samples s ON s.user_id=u.id
            {where}
            GROUP BY u.id, u.name, u.role, u.active, u.created_at
            ORDER BY u.created_at DESC
            """
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def fetch_enrollment_templates():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT s.id AS sample_id, s.user_id, u.name AS user_name, s.angle_key, s.angle_label, s.image_path, s.feature_template, s.identity_model
            FROM enrollment_samples s
            INNER JOIN authorised_users u ON u.id=s.user_id
            WHERE u.active=TRUE
            ORDER BY u.id, s.id
            """
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def insert_user_with_samples(name: str, role: str, samples: list[dict]) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        connection.start_transaction()
        cursor.execute("INSERT INTO authorised_users (name, role) VALUES (%s, %s)", (name, role))
        user_id = cursor.lastrowid
        for sample in samples:
            cursor.execute(
                """
                INSERT INTO enrollment_samples (user_id, angle_key, angle_label, image_path, feature_template, identity_model)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, sample["angle_key"], sample["angle_label"], sample["image_path"], sample["feature_blob"], sample.get("identity_model", "ArcFace")),
            )
        connection.commit()
        return int(user_id)
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def set_user_active(user_id: int, active: bool) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE authorised_users SET active=%s WHERE id=%s", (active, user_id))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def insert_recognition_session(result: dict, evidence_path: str | None, liveness: dict, cooldown_seconds: int) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO recognition_sessions (
                matched_user_id, best_candidate_name, decision, identity_model, usable_frames, winning_votes,
                required_votes, median_confidence, median_fusion_score, median_identity_margin,
                evidence_path, liveness_result, liveness_details, cooldown_seconds
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                result.get("winner_user_id") if result.get("decision") == "Authorised" else None,
                result.get("winner_name"),
                result.get("decision", "Unknown"),
                result.get("identity_model", "ArcFace"),
                result.get("usable_frames", 0),
                result.get("winner_votes", 0),
                result.get("required_votes", 0),
                result.get("median_confidence", 0.0),
                result.get("median_fusion_score", 0.0),
                result.get("median_identity_margin", 0.0),
                evidence_path,
                "passed" if liveness.get("passed") else "failed",
                str(liveness),
                cooldown_seconds,
            ),
        )
        session_id = cursor.lastrowid
        connection.commit()
        return int(session_id)
    finally:
        cursor.close()
        connection.close()


def insert_frame_results(session_id: int, frame_results: list[dict]) -> None:
    if not frame_results:
        return
    connection = get_connection()
    cursor = connection.cursor()
    try:
        for frame in frame_results:
            cursor.execute(
                """
                INSERT INTO recognition_frame_results (
                    session_id, frame_index, candidate_user_id, candidate_name, best_angle_label,
                    frame_decision, cnn_similarity, arcface_distance, geometry_similarity, mahalanobis_distance,
                    mahalanobis_evidence, iris_similarity, fusion_score, logistic_probability,
                    identity_margin, rejection_reason
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    session_id,
                    frame.get("frame_index", 0),
                    frame.get("candidate_user_id"),
                    frame.get("candidate_name"),
                    frame.get("best_angle_label"),
                    frame.get("frame_decision", "Unknown"),
                    frame.get("arcface_similarity", frame.get("cnn_similarity", 0.0)),
                    frame.get("arcface_distance"),
                    frame.get("geometry_similarity", 0.0),
                    frame.get("mahalanobis_distance", 0.0),
                    frame.get("mahalanobis_evidence", 0.0),
                    frame.get("iris_similarity", 0.0),
                    frame.get("fusion_score", 0.0),
                    frame.get("logistic_probability", 0.0),
                    frame.get("identity_margin"),
                    frame.get("rejection_reason"),
                ),
            )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def create_incident(session_id: int | None, reason: str, severity: str, evidence_path: str | None) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO incidents (session_id, reason, severity, evidence_path) VALUES (%s,%s,%s,%s)",
            (session_id, reason, severity, evidence_path),
        )
        incident_id = cursor.lastrowid
        connection.commit()
        return int(incident_id)
    finally:
        cursor.close()
        connection.close()


def insert_alert(incident_id: int | None, alert_type: str, message: str) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO alert_logs (incident_id, alert_type, message) VALUES (%s,%s,%s)", (incident_id, alert_type, message))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def fetch_sessions(limit: int = 50, decision: str = "", name: str = ""):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        clauses, params = [], []
        if decision:
            clauses.append("decision=%s")
            params.append(decision)
        if name:
            clauses.append("best_candidate_name LIKE %s")
            params.append(f"%{name}%")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        cursor.execute(
            f"""
            SELECT id, best_candidate_name, decision, identity_model, usable_frames, winning_votes, required_votes,
                   median_confidence, median_fusion_score, median_identity_margin, evidence_path,
                   liveness_result, cooldown_seconds, created_at
            FROM recognition_sessions
            {where}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            tuple(params),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def fetch_incidents(status: str = ""):
    """Return incident rows enriched with the linked recognition-session summary.

    This keeps the dashboard useful without forcing an administrator to open a
    second table just to understand why an incident was created.
    """
    connection = get_connection()
    metadata_cursor = connection.cursor()
    cursor = connection.cursor(dictionary=True)
    try:
        incident_pk = _primary_key_column(metadata_cursor, "incidents")
        id_projection = "i.id" if incident_pk == "id" else f"i.`{incident_pk}` AS id"
        where = "WHERE i.status=%s" if status else ""
        params = (status,) if status else ()
        cursor.execute(
            f"""
            SELECT {id_projection}, i.session_id, i.reason, i.severity, i.status, i.evidence_path,
                   i.reviewed_by, i.reviewed_at, i.notes, i.created_at, a.username AS reviewer,
                   s.decision AS session_decision, s.identity_model, s.best_candidate_name, s.usable_frames,
                   s.winning_votes, s.required_votes, s.median_confidence,
                   s.median_fusion_score, s.median_identity_margin, s.liveness_result,
                   COALESCE(al.alert_count, 0) AS alert_count
            FROM incidents i
            LEFT JOIN administrators a ON a.id=i.reviewed_by
            LEFT JOIN recognition_sessions s ON s.id=i.session_id
            LEFT JOIN (
                SELECT incident_id, COUNT(*) AS alert_count
                FROM alert_logs
                GROUP BY incident_id
            ) al ON al.incident_id=i.`{incident_pk}`
            {where}
            ORDER BY i.created_at DESC
            """,
            params,
        )
        return cursor.fetchall()
    finally:
        metadata_cursor.close()
        cursor.close()
        connection.close()


def fetch_incident_details(incident_id: int):
    """Return one incident with its recognition session, frame evidence and alerts."""
    connection = get_connection()
    metadata_cursor = connection.cursor()
    cursor = connection.cursor(dictionary=True)
    try:
        incident_pk = _primary_key_column(metadata_cursor, "incidents")
        id_projection = "i.id" if incident_pk == "id" else f"i.`{incident_pk}` AS id"
        cursor.execute(
            f"""
            SELECT {id_projection}, i.session_id, i.reason, i.severity, i.status, i.evidence_path,
                   i.reviewed_by, i.reviewed_at, i.notes, i.created_at, a.username AS reviewer,
                   s.best_candidate_name, s.decision AS session_decision, s.identity_model, s.usable_frames,
                   s.winning_votes, s.required_votes, s.median_confidence,
                   s.median_fusion_score, s.median_identity_margin, s.evidence_path AS session_evidence_path,
                   s.liveness_result, s.liveness_details, s.cooldown_seconds, s.created_at AS session_created_at
            FROM incidents i
            LEFT JOIN administrators a ON a.id=i.reviewed_by
            LEFT JOIN recognition_sessions s ON s.id=i.session_id
            WHERE i.`{incident_pk}`=%s
            """,
            (incident_id,),
        )
        incident = cursor.fetchone()
        if not incident:
            return None

        frames = []
        if incident.get("session_id") is not None:
            cursor.execute(
                """
                SELECT frame_index, candidate_name, best_angle_label, frame_decision,
                       cnn_similarity, arcface_distance, geometry_similarity, mahalanobis_distance,
                       mahalanobis_evidence, iris_similarity, fusion_score,
                       logistic_probability, identity_margin, rejection_reason, created_at
                FROM recognition_frame_results
                WHERE session_id=%s
                ORDER BY frame_index ASC
                """,
                (incident["session_id"],),
            )
            frames = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, alert_type, status, message, created_at
            FROM alert_logs
            WHERE incident_id=%s
            ORDER BY created_at ASC, id ASC
            """,
            (incident_id,),
        )
        alerts = cursor.fetchall()
        return {"incident": incident, "frames": frames, "alerts": alerts}
    finally:
        metadata_cursor.close()
        cursor.close()
        connection.close()


def count_open_incidents() -> int:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM incidents WHERE status IN ('open','investigating')")
        return int(cursor.fetchone()[0])
    finally:
        cursor.close()
        connection.close()


def delete_incidents(incident_ids: list[int]) -> int:
    """Delete selected incident/alert log rows for controlled testing cleanup.

    Recognition sessions and image evidence are intentionally retained so deleting
    an incident cannot silently destroy the underlying biometric test record.
    """
    ids = sorted({int(value) for value in incident_ids if int(value) > 0})
    if not ids:
        return 0
    connection = get_connection()
    cursor = connection.cursor()
    try:
        connection.start_transaction()
        incident_pk = _primary_key_column(cursor, "incidents")
        placeholders = ",".join(["%s"] * len(ids))
        cursor.execute(f"DELETE FROM alert_logs WHERE incident_id IN ({placeholders})", tuple(ids))
        cursor.execute(f"DELETE FROM incidents WHERE `{incident_pk}` IN ({placeholders})", tuple(ids))
        deleted = int(cursor.rowcount)
        connection.commit()
        return deleted
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

def review_incident(incident_id: int, admin_id: int, status: str, notes: str) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        incident_pk = _primary_key_column(cursor, "incidents")
        cursor.execute(
            f"UPDATE incidents SET status=%s, notes=%s, reviewed_by=%s, reviewed_at=%s WHERE `{incident_pk}`=%s",
            (status, notes, admin_id, datetime.now(), incident_id),
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def dashboard_metrics() -> dict:
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        metrics = {}
        cursor.execute("SELECT COUNT(*) AS c FROM authorised_users WHERE active=TRUE")
        metrics["active_users"] = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM recognition_sessions")
        metrics["sessions"] = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM incidents WHERE status IN ('open','investigating')")
        metrics["open_incidents"] = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM guest_keys WHERE revoked=FALSE AND valid_from<=NOW() AND valid_until>=NOW() AND uses<max_uses")
        metrics["active_guest_keys"] = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM guest_access_logs WHERE DATE(created_at)=CURDATE()")
        metrics["guest_accesses_today"] = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) AS c FROM guest_access_logs WHERE decision='Guest Rejected' AND DATE(created_at)=CURDATE()")
        metrics["guest_rejections_today"] = cursor.fetchone()["c"]
        return metrics
    finally:
        cursor.close()
        connection.close()


def audit(admin_id: int | None, action: str, details: str = "") -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO audit_logs (admin_id, action, details) VALUES (%s,%s,%s)", (admin_id, action, details))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def guest_key_hash_exists(key_hash: str) -> bool:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id FROM guest_keys WHERE key_hash=%s", (key_hash,))
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        connection.close()


def insert_guest_key(key_hash: str, key_last4: str, guest_name: str, valid_from: datetime, valid_until: datetime, max_uses: int, admin_id: int | None) -> int:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO guest_keys (key_hash, key_last4, guest_name, valid_from, valid_until, max_uses, created_by)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (key_hash, key_last4, guest_name, valid_from, valid_until, max_uses, admin_id),
        )
        key_id = cursor.lastrowid
        connection.commit()
        return int(key_id)
    finally:
        cursor.close()
        connection.close()


def fetch_guest_key_by_hash(key_hash: str):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM guest_keys WHERE key_hash=%s", (key_hash,))
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def record_guest_access(key_id: int | None, guest_name: str | None, last4: str | None, decision: str, reason: str) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO guest_access_logs (guest_key_id, guest_name, provided_last4, decision, reason) VALUES (%s,%s,%s,%s,%s)",
            (key_id, guest_name, last4, decision, reason),
        )
        if decision == "Guest Authorised" and key_id is not None:
            cursor.execute("UPDATE guest_keys SET uses=uses+1 WHERE id=%s", (key_id,))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def fetch_guest_keys():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT id, key_last4, guest_name, valid_from, valid_until, max_uses, uses, revoked, created_at
            FROM guest_keys ORDER BY created_at DESC LIMIT 50
            """
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def fetch_guest_access_logs(limit: int = 50):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT id, guest_key_id, guest_name, provided_last4, decision, reason, created_at
            FROM guest_access_logs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def revoke_guest_key(key_id: int) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE guest_keys SET revoked=TRUE WHERE id=%s", (key_id,))
        connection.commit()
    finally:
        cursor.close()
        connection.close()
