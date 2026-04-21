import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_text(path: Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines <= 0:
        return lines
    return lines[-max_lines:]


def _seconds_since(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime))


def _classify_cause(
    weekly_report: dict[str, Any] | None,
    heartbeat_age_s: float | None,
    heartbeat_stale_s: int,
    interrupted_report: dict[str, Any] | None,
    runner_stderr_tail: list[str],
) -> tuple[str, list[str]]:
    hints: list[str] = []
    probable = "unknown"

    failed_step = None
    rc = None
    if weekly_report:
        failed_step = weekly_report.get("failed_step")
        for step in weekly_report.get("steps", []):
            if step.get("name") == failed_step:
                rc = step.get("returncode")
                stderr_hint = step.get("stderr")
                if stderr_hint:
                    hints.append(str(stderr_hint))
                break

    if interrupted_report and interrupted_report.get("status") == "interrupted":
        probable = "graceful_interruption"
        hints.append(f"training_interrupted_reason={interrupted_report.get('reason')}")

    if rc == 4294967295:
        probable = "external_termination"
        hints.append("windows_returncode_4294967295")

    if heartbeat_age_s is not None and heartbeat_age_s > heartbeat_stale_s:
        probable = "training_freeze_or_stall"
        hints.append(f"heartbeat_stale_seconds={round(heartbeat_age_s, 1)}")

    if failed_step:
        hints.append(f"failed_step={failed_step}")

    if runner_stderr_tail:
        hints.append("runner_stderr_present")

    return probable, hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a failed weekly run and produce post-mortem JSON")
    parser.add_argument("--weekly-report", default="reports/weekly_train_runner_report.json")
    parser.add_argument("--runner-stderr", default="reports/weekly_train_runner_stderr.log")
    parser.add_argument("--runner-stdout", default="reports/weekly_train_runner_stdout.log")
    parser.add_argument("--lock-file", default="reports/.weekly_dual_lane.lock")
    parser.add_argument("--heartbeat-file", default="reports/training_heartbeat.json")
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=300)
    parser.add_argument("--interrupted-report", default="reports/training_interrupted.json")
    parser.add_argument("--out-json", default="reports/failed_run_analysis.json")
    parser.add_argument("--tail-lines", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weekly_path = Path(args.weekly_report)
    stderr_path = Path(args.runner_stderr)
    stdout_path = Path(args.runner_stdout)
    lock_path = Path(args.lock_file)
    heartbeat_path = Path(args.heartbeat_file)
    interrupted_path = Path(args.interrupted_report)

    weekly_report = _read_json(weekly_path)
    interrupted_report = _read_json(interrupted_path)
    heartbeat_report = _read_json(heartbeat_path)
    lock_report = _read_json(lock_path)

    stderr_tail = _tail_text(stderr_path, args.tail_lines)
    stdout_tail = _tail_text(stdout_path, args.tail_lines)

    heartbeat_age_s = _seconds_since(heartbeat_path)

    probable_cause, hints = _classify_cause(
        weekly_report=weekly_report,
        heartbeat_age_s=heartbeat_age_s,
        heartbeat_stale_s=args.heartbeat_stale_seconds,
        interrupted_report=interrupted_report,
        runner_stderr_tail=stderr_tail,
    )

    failed_step = weekly_report.get("failed_step") if weekly_report else None
    run_started = weekly_report.get("started_at") if weekly_report else None
    run_finished = weekly_report.get("finished_at") if weekly_report else None

    payload: dict[str, Any] = {
        "generated_at": _now_iso(),
        "status": "analysis_complete",
        "probable_cause": probable_cause,
        "hints": hints,
        "failed_step": failed_step,
        "run_started_at": run_started,
        "run_finished_at": run_finished,
        "files_seen": {
            "weekly_report": str(weekly_path),
            "runner_stderr": str(stderr_path),
            "runner_stdout": str(stdout_path),
            "lock_file": str(lock_path),
            "heartbeat_file": str(heartbeat_path),
            "interrupted_report": str(interrupted_path),
        },
        "observability": {
            "heartbeat_age_seconds": heartbeat_age_s,
            "heartbeat_stale_threshold_seconds": args.heartbeat_stale_seconds,
            "heartbeat": heartbeat_report,
            "lock": lock_report,
            "interrupted": interrupted_report,
        },
        "runner_log_tails": {
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
        },
        "suggestions": [
            "Verify no concurrent training process is active before starting weekly run.",
            "If probable cause is external_termination, check user session close, watchdog, and antivirus events.",
            "If heartbeat is stale, inspect GPU/CPU pressure and memory exhaustion around failure timestamp.",
            "Relaunch question lane evaluation/promotion on current stable model to keep challenger governance moving.",
        ],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": payload["status"],
                "probable_cause": probable_cause,
                "out_json": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
