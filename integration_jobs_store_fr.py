from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_iso_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return 0.0


def save_jobs(path: Path, jobs: dict[str, dict[str, Any]], updated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": updated_at,
        "jobs": jobs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jobs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    jobs = raw.get("jobs")
    if not isinstance(jobs, dict):
        return {}
    return jobs


def purge_expired_jobs(jobs: dict[str, dict[str, Any]], ttl_seconds: int, now_ts: float | None = None) -> int:
    current = now_ts if now_ts is not None else time.time()
    ttl = max(60, int(ttl_seconds))
    removed = 0

    for job_id, job in list(jobs.items()):
        status = str(job.get("status") or "")
        if status in {"queued", "running"}:
            continue

        finished_ts = parse_iso_ts(job.get("finished_at"))
        if not finished_ts:
            finished_ts = parse_iso_ts(job.get("created_at"))

        if finished_ts and (current - finished_ts) > ttl:
            jobs.pop(job_id, None)
            removed += 1

    return removed


def count_jobs_by_status(jobs: dict[str, dict[str, Any]]) -> dict[str, int]:
    queued = 0
    running = 0
    finished = 0
    for job in jobs.values():
        status = str(job.get("status") or "")
        if status == "queued":
            queued += 1
        elif status == "running":
            running += 1
        elif status in {"done", "error"}:
            finished += 1
    return {
        "queued": queued,
        "running": running,
        "finished": finished,
        "total": queued + running + finished,
    }


def pending_for_principal(jobs: dict[str, dict[str, Any]], principal: str) -> int:
    total = 0
    for job in jobs.values():
        if job.get("principal") != principal:
            continue
        status = str(job.get("status") or "")
        if status in {"queued", "running"}:
            total += 1
    return total
