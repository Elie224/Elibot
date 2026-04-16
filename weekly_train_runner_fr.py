import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly training orchestration for Elibot")

    parser.add_argument("--python-exec", default=sys.executable)

    # Step 1: feeding pipeline
    parser.add_argument("--chat-log", default="data/logs/elibot_chat_events.jsonl")
    parser.add_argument("--audit-log", default="data/logs/elibot_audit.jsonl")
    parser.add_argument("--signature-dataset", default="data/processed/chatbot_train_fr_signature_v2_domain.csv")
    parser.add_argument("--feedback-out", default="reports/derived/chatbot_train_fr_feedback_weekly.csv")
    parser.add_argument("--bundle-out", default="reports/derived/chatbot_train_fr_weekly_bundle.csv")
    parser.add_argument("--feeding-report", default="reports/feeding_pipeline_report.json")
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--min-quality-score", type=float, default=0.65)
    parser.add_argument("--max-feedback-rows", type=int, default=30000)
    parser.add_argument("--max-signature-rows", type=int, default=15000)

    # Step 2: training
    parser.add_argument("--base-model", default="models/chatbot-fr-flan-t5-small-v2-signature")
    parser.add_argument("--output-model", default="models/chatbot-fr-flan-t5-small-weekly")
    parser.add_argument("--train-max-samples", type=int, default=20000)
    parser.add_argument("--train-max-eval-samples", type=int, default=1200)
    parser.add_argument("--train-epochs", type=float, default=0.8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-grad-accum", type=int, default=2)
    parser.add_argument("--train-learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-log-steps", type=int, default=50)

    # Step 3: evaluation
    parser.add_argument("--eval-samples", type=int, default=150)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-max-new-tokens", type=int, default=96)
    parser.add_argument("--eval-out-json", default="reports/eval_weekly_latest.json")
    parser.add_argument("--eval-out-csv", default="reports/eval_weekly_latest_samples.csv")

    parser.add_argument("--pipeline-seed", type=int, default=42)
    parser.add_argument("--run-report", default="reports/weekly_train_runner_report.json")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def _run_command(cmd: list[str], dry_run: bool) -> dict[str, Any]:
    cmd_display = " ".join(shlex.quote(x) for x in cmd)
    if dry_run:
        return {
            "command": cmd,
            "command_display": cmd_display,
            "ok": True,
            "returncode": 0,
            "stdout": "[dry-run] skipped",
            "stderr": "",
            "duration_s": 0.0,
        }

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - started

    return {
        "command": cmd,
        "command_display": cmd_display,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "duration_s": round(duration, 3),
    }


def main() -> None:
    args = parse_args()

    run_started = _now_iso()
    report: dict[str, Any] = {
        "started_at": run_started,
        "dry_run": bool(args.dry_run),
        "steps": [],
        "status": "running",
    }

    step1 = [
        args.python_exec,
        "continuous_learning_pipeline_fr.py",
        "--chat-log",
        args.chat_log,
        "--audit-log",
        args.audit_log,
        "--signature-dataset",
        args.signature_dataset,
        "--out-feedback",
        args.feedback_out,
        "--out-bundle",
        args.bundle_out,
        "--out-report",
        args.feeding_report,
        "--min-response-chars",
        str(args.min_response_chars),
        "--min-quality-score",
        str(args.min_quality_score),
        "--max-feedback-rows",
        str(args.max_feedback_rows),
        "--max-signature-rows",
        str(args.max_signature_rows),
        "--seed",
        str(args.pipeline_seed),
    ]
    if args.allow_empty:
        step1.append("--allow-empty")

    step2 = [
        args.python_exec,
        "train_chatbot_fr.py",
        "--data-file",
        args.bundle_out,
        "--model-name",
        args.base_model,
        "--output-dir",
        args.output_model,
        "--max-train-samples",
        str(args.train_max_samples),
        "--max-eval-samples",
        str(args.train_max_eval_samples),
        "--epochs",
        str(args.train_epochs),
        "--batch-size",
        str(args.train_batch_size),
        "--grad-accum",
        str(args.train_grad_accum),
        "--learning-rate",
        str(args.train_learning_rate),
        "--warmup-ratio",
        str(args.train_warmup_ratio),
        "--max-grad-norm",
        str(args.train_max_grad_norm),
        "--log-steps",
        str(args.train_log_steps),
    ]

    step3 = [
        args.python_exec,
        "evaluate_model_fr.py",
        "--model-dir",
        args.output_model,
        "--data-file",
        args.bundle_out,
        "--samples",
        str(args.eval_samples),
        "--seed",
        str(args.eval_seed),
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
        "--out-json",
        args.eval_out_json,
        "--out-csv",
        args.eval_out_csv,
    ]

    for name, cmd in [
        ("feeding_pipeline", step1),
        ("training", step2),
        ("evaluation", step3),
    ]:
        result = _run_command(cmd, dry_run=bool(args.dry_run))
        result["name"] = name
        report["steps"].append(result)
        if not result["ok"]:
            report["status"] = "failed"
            report["failed_step"] = name
            break
        if name == "feeding_pipeline" and not args.dry_run:
            bundle_path = Path(args.bundle_out)
            if not bundle_path.exists():
                report["status"] = "failed"
                report["failed_step"] = "feeding_pipeline"
                report["failure_reason"] = f"bundle_not_found:{bundle_path}"
                break

    if report.get("status") == "running":
        report["status"] = "ok"

    report["finished_at"] = _now_iso()

    out = Path(args.run_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "report": str(out),
        "dry_run": report["dry_run"],
        "steps": [{"name": s["name"], "ok": s["ok"], "returncode": s["returncode"]} for s in report["steps"]],
    }, ensure_ascii=False))

    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
