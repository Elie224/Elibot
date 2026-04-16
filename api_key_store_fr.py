from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path


STORE_PATH = Path("data/security/api_keys_store.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_store() -> dict:
    if not STORE_PATH.exists():
        return {"keys": []}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"keys": []}


def _write_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_key(raw_key: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", raw_key.encode("utf-8"), salt, 120_000)
    return dk.hex()


def create_api_key(role: str, label: str = "") -> dict:
    if role not in {"basic", "advanced", "admin"}:
        raise ValueError("invalid role")

    raw_key = "elibot_" + secrets.token_urlsafe(28)
    salt_hex = secrets.token_hex(16)
    key_hash = _hash_key(raw_key, salt_hex)
    key_id = str(uuid.uuid4())

    record = {
        "id": key_id,
        "role": role,
        "label": label.strip(),
        "salt": salt_hex,
        "hash": key_hash,
        "active": True,
        "created_at": _now_iso(),
        "revoked_at": None,
        "last_used_at": None,
    }

    store = _read_store()
    keys = store.get("keys", [])
    keys.append(record)
    store["keys"] = keys
    _write_store(store)

    return {
        "id": key_id,
        "role": role,
        "label": record["label"],
        "created_at": record["created_at"],
        "api_key": raw_key,
    }


def verify_api_key(raw_key: str) -> dict | None:
    if not raw_key:
        return None

    store = _read_store()
    keys = store.get("keys", [])
    changed = False

    for rec in keys:
        if not rec.get("active", False):
            continue
        salt_hex = rec.get("salt") or ""
        expected = rec.get("hash") or ""
        if not salt_hex or not expected:
            continue

        candidate = _hash_key(raw_key, salt_hex)
        if secrets.compare_digest(candidate, expected):
            rec["last_used_at"] = _now_iso()
            changed = True
            if changed:
                _write_store(store)
            return {
                "id": rec.get("id"),
                "role": rec.get("role", "basic"),
                "label": rec.get("label", ""),
                "source": "store",
            }

    return None


def list_api_keys(include_inactive: bool = False) -> list[dict]:
    store = _read_store()
    out = []
    for rec in store.get("keys", []):
        if (not include_inactive) and (not rec.get("active", False)):
            continue
        out.append(
            {
                "id": rec.get("id"),
                "role": rec.get("role", "basic"),
                "label": rec.get("label", ""),
                "active": rec.get("active", False),
                "created_at": rec.get("created_at"),
                "revoked_at": rec.get("revoked_at"),
                "last_used_at": rec.get("last_used_at"),
            }
        )
    return out


def revoke_api_key(key_id: str) -> bool:
    if not key_id:
        return False

    store = _read_store()
    keys = store.get("keys", [])
    changed = False

    for rec in keys:
        if rec.get("id") == key_id and rec.get("active", False):
            rec["active"] = False
            rec["revoked_at"] = _now_iso()
            changed = True
            break

    if changed:
        _write_store(store)
    return changed


def bootstrap_from_env(env_key_role_map: dict[str, str]) -> int:
    """Persist env keys as hashed records once for future rotations.

    Returns number of inserted records.
    """
    if not env_key_role_map:
        return 0

    store = _read_store()
    keys = store.get("keys", [])

    # Build quick lookup by (role, hash) to avoid duplicates.
    existing_pairs = set()
    for rec in keys:
        role = rec.get("role", "basic")
        h = rec.get("hash", "")
        if role and h:
            existing_pairs.add((role, h))

    inserted = 0
    for raw_key, role in env_key_role_map.items():
        salt_hex = secrets.token_hex(16)
        key_hash = _hash_key(raw_key, salt_hex)
        if (role, key_hash) in existing_pairs:
            continue

        keys.append(
            {
                "id": str(uuid.uuid4()),
                "role": role,
                "label": "bootstrapped_env",
                "salt": salt_hex,
                "hash": key_hash,
                "active": True,
                "created_at": _now_iso(),
                "revoked_at": None,
                "last_used_at": None,
            }
        )
        existing_pairs.add((role, key_hash))
        inserted += 1

    if inserted:
        store["keys"] = keys
        _write_store(store)

    return inserted
