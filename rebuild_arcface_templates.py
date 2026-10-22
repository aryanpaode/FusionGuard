from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import db
from config import BASE_DIR, IDENTITY_MODEL_NAME
from fusionguard_multimodal import extract_template, serialize_template, warmup_arcface


def resolve_image_path(stored_path: str) -> Path:
    path = Path(stored_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild existing FusionGuard enrolment feature blobs with ArcFace without changing user IDs or image files."
    )
    parser.add_argument("--user-id", type=int, action="append", help="Only rebuild this user ID. Repeat for multiple users.")
    parser.add_argument("--dry-run", action="store_true", help="Validate image paths and show planned work without changing MySQL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db.initialise_database()

    connection = db.get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        params = []
        where = ""
        if args.user_id:
            placeholders = ",".join(["%s"] * len(args.user_id))
            where = f"WHERE s.user_id IN ({placeholders})"
            params.extend(args.user_id)

        cursor.execute(
            f"""
            SELECT s.id AS sample_id, s.user_id, u.name AS user_name, s.angle_label,
                   s.image_path, s.identity_model
            FROM enrollment_samples s
            JOIN authorised_users u ON u.id=s.user_id
            {where}
            ORDER BY s.user_id, s.id
            """,
            tuple(params),
        )
        samples = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    if not samples:
        print("No enrolment samples were found for the selected users.")
        return 1

    missing = []
    print("Enrolment images to rebuild:")
    for sample in samples:
        image_path = resolve_image_path(sample["image_path"])
        exists = image_path.exists()
        print(
            f"  sample #{sample['sample_id']:>3} | user #{sample['user_id']:<3} | "
            f"{sample['user_name']:<24} | {sample['angle_label']:<12} | "
            f"{'OK' if exists else 'MISSING'} | {image_path}"
        )
        if not exists:
            missing.append(image_path)

    if missing:
        print("\nNo database changes were made.")
        print("Restore/copy the old enrolment folder so every image_path above exists, then rerun this script.")
        return 2

    if args.dry_run:
        print("\nDry run complete. All selected enrolment images exist; no MySQL changes were made.")
        return 0

    print("\nLoading ArcFace. The first run may download ArcFace weights...")
    warmup_arcface()

    rebuilt = []
    for index, sample in enumerate(samples, start=1):
        image_path = resolve_image_path(sample["image_path"])
        print(f"[{index}/{len(samples)}] ArcFace template: {sample['user_name']} - {sample['angle_label']}")
        template = extract_template(image_path)
        rebuilt.append((serialize_template(template), sample["sample_id"]))

    backup_table = f"enrollment_samples_backup_{datetime.now():%Y%m%d_%H%M%S}"
    connection = db.get_connection()
    cursor = connection.cursor()
    try:
        connection.start_transaction()
        cursor.execute(f"CREATE TABLE `{backup_table}` LIKE enrollment_samples")
        cursor.execute(f"INSERT INTO `{backup_table}` SELECT * FROM enrollment_samples")
        for blob, sample_id in rebuilt:
            cursor.execute(
                "UPDATE enrollment_samples SET feature_template=%s, identity_model=%s WHERE id=%s",
                (blob, IDENTITY_MODEL_NAME, sample_id),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    print(f"\nCompleted: {len(rebuilt)} enrolment template(s) rebuilt with ArcFace.")
    print(f"Backup table created: {backup_table}")
    print("User IDs, user names, sample IDs, angle labels and image paths were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
