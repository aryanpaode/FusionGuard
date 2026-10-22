from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from config import GUEST_KEY_PEPPER


def hash_guest_key(key: str) -> str:
    material = f"{GUEST_KEY_PEPPER}:{key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def generate_numeric_key() -> str:
    return f"{secrets.randbelow(100_000_000):08d}"


def create_guest_key(guest_name: str, valid_from: datetime, valid_until: datetime, max_uses: int, admin_id: int | None) -> tuple[int, str]:
    from db import guest_key_hash_exists, insert_guest_key

    if valid_until <= valid_from:
        raise ValueError("Guest-key expiry must be after its start time.")
    if max_uses < 1:
        raise ValueError("Guest-key max uses must be at least 1.")

    for _ in range(30):
        plaintext = generate_numeric_key()
        digest = hash_guest_key(plaintext)
        if not guest_key_hash_exists(digest):
            key_id = insert_guest_key(digest, plaintext[-4:], guest_name, valid_from, valid_until, max_uses, admin_id)
            return key_id, plaintext
    raise RuntimeError("Could not generate a unique guest key after repeated attempts.")


def verify_guest_key(plaintext: str, now: datetime | None = None) -> dict:
    from db import fetch_guest_key_by_hash, record_guest_access

    now = now or datetime.now()
    key = plaintext.strip()
    if len(key) != 8 or not key.isdigit():
        record_guest_access(None, None, key[-4:] if key else None, "Guest Rejected", "Key must contain exactly eight digits.")
        return {"authorised": False, "reason": "Enter the eight-digit numeric guest key.", "guest_name": None, "key_id": None}

    row = fetch_guest_key_by_hash(hash_guest_key(key))
    if row is None:
        record_guest_access(None, None, key[-4:], "Guest Rejected", "Unknown key")
        return {"authorised": False, "reason": "Guest key was not recognised.", "guest_name": None, "key_id": None}

    reason = None
    if row["revoked"]:
        reason = "Guest key has been revoked."
    elif now < row["valid_from"]:
        reason = "Guest key is not active yet."
    elif now > row["valid_until"]:
        reason = "Guest key has expired."
    elif row["uses"] >= row["max_uses"]:
        reason = "Guest key has reached its usage limit."

    if reason:
        record_guest_access(row["id"], row["guest_name"], key[-4:], "Guest Rejected", reason)
        return {"authorised": False, "reason": reason, "guest_name": row["guest_name"], "key_id": row["id"]}

    record_guest_access(row["id"], row["guest_name"], key[-4:], "Guest Authorised", "Valid time-bound guest key")
    return {
        "authorised": True,
        "reason": "Guest access authorised.",
        "guest_name": row["guest_name"],
        "remaining_uses": row["max_uses"] - row["uses"] - 1,
    }
