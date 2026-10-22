CREATE DATABASE IF NOT EXISTS fusionguard_multimodal CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE fusionguard_multimodal;

CREATE TABLE IF NOT EXISTS administrators (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS authorised_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL UNIQUE,
    role VARCHAR(80) NOT NULL DEFAULT 'Authorised User',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

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
);

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
);

CREATE TABLE IF NOT EXISTS recognition_frame_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    frame_index INT NOT NULL,
    candidate_user_id INT NULL,
    candidate_name VARCHAR(120) NULL,
    best_angle_label VARCHAR(50) NULL,
    frame_decision VARCHAR(40) NOT NULL,
    cnn_similarity DECIMAL(8,3) NOT NULL DEFAULT 0, -- legacy column name; stores ArcFace similarity for V3.5+
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
);

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
);

CREATE TABLE IF NOT EXISTS alert_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    incident_id BIGINT NULL,
    alert_type VARCHAR(40) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'raised',
    message VARCHAR(500) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON UPDATE CASCADE ON DELETE SET NULL
);

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
);

CREATE TABLE IF NOT EXISTS guest_access_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    guest_key_id BIGINT NULL,
    guest_name VARCHAR(120) NULL,
    provided_last4 CHAR(4) NULL,
    decision VARCHAR(40) NOT NULL,
    reason VARCHAR(255) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (guest_key_id) REFERENCES guest_keys(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    admin_id INT NULL,
    action VARCHAR(120) NOT NULL,
    details TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES administrators(id) ON UPDATE CASCADE ON DELETE SET NULL
);
