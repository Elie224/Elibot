from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_MEMORY_PATH = Path("data/memory/task_memory.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_store() -> dict[str, Any]:
    if not TASK_MEMORY_PATH.exists():
        return {}
    try:
        return json.loads(TASK_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(payload: dict[str, Any]) -> None:
    TASK_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_MEMORY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_tasks(session_id: str, include_done: bool = True) -> list[dict[str, Any]]:
    store = _read_store()
    items = store.get(session_id, []) if isinstance(store.get(session_id, []), list) else []
    if include_done:
        return items
    return [x for x in items if x.get("status") != "done"]


def upsert_task(session_id: str, title: str, steps: list[str] | None = None) -> dict[str, Any]:
    store = _read_store()
    items = store.get(session_id, []) if isinstance(store.get(session_id, []), list) else []

    task = {
        "task_id": str(uuid.uuid4()),
        "title": (title or "Tache technique").strip()[:180],
        "steps": [s.strip() for s in (steps or []) if str(s).strip()][:20],
        "done_steps": [],
        "remaining_steps": len([s for s in (steps or []) if str(s).strip()]),
        "status": "active",
        "errors": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    items.append(task)
    store[session_id] = items
    _write_store(store)
    return task


def record_progress(
    session_id: str,
    task_id: str,
    done_step: str | None = None,
    error: str | None = None,
    mark_done: bool = False,
) -> dict[str, Any]:
    store = _read_store()
    items = store.get(session_id, []) if isinstance(store.get(session_id, []), list) else []

    target = None
    for item in items:
        if item.get("task_id") == task_id:
            target = item
            break

    if target is None:
        raise ValueError("task not found")

    if done_step:
        step = done_step.strip()
        if step and step not in target.get("done_steps", []):
            target.setdefault("done_steps", []).append(step)

    if error:
        target.setdefault("errors", []).append({"at": _now_iso(), "message": str(error)[:300]})

    total_steps = len(target.get("steps", []))
    done_count = len(target.get("done_steps", []))
    target["remaining_steps"] = max(0, total_steps - done_count)

    if mark_done or (total_steps > 0 and done_count >= total_steps):
        target["status"] = "done"
    elif target.get("errors"):
        target["status"] = "blocked"
    else:
        target["status"] = "active"

    target["updated_at"] = _now_iso()
    _write_store(store)
    return target
