import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weekly continual-learning dataset bundle for Elibot")
    parser.add_argument("--chat-log", default="data/logs/elibot_chat_events.jsonl")
    parser.add_argument("--audit-log", default="data/logs/elibot_audit.jsonl")
    parser.add_argument("--signature-dataset", default="data/processed/chatbot_train_fr_signature_v2_domain.csv")
    parser.add_argument("--out-feedback", default="data/processed/chatbot_train_fr_feedback_weekly.csv")
    parser.add_argument("--out-bundle", default="data/processed/chatbot_train_fr_weekly_bundle.csv")
    parser.add_argument("--out-report", default="reports/feeding_pipeline_report.json")
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--min-quality-score", type=float, default=0.65)
    parser.add_argument("--max-feedback-rows", type=int, default=30000)
    parser.add_argument("--max-signature-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _heuristic_quality(event: dict[str, Any], min_response_chars: int) -> float:
    score = 0.0
    assistant_text = (event.get("assistant_text") or "").strip()

    if event.get("in_domain", False):
        score += 0.30
    if not event.get("is_low_quality", False):
        score += 0.25
    if not event.get("corrected_by_verifier", False):
        score += 0.15

    issues = event.get("verifier_issues") or []
    if not issues:
        score += 0.15

    if len(assistant_text) >= min_response_chars:
        score += 0.15

    return max(0.0, min(1.0, score))


def _event_quality(event: dict[str, Any], min_response_chars: int) -> float:
    internal = event.get("internal_scores") or {}
    if isinstance(internal, dict) and "quality" in internal:
        return max(0.0, min(1.0, _safe_float(internal.get("quality"), 0.0)))
    return _heuristic_quality(event, min_response_chars=min_response_chars)


def _extract_feedback_rows(
    events: list[dict[str, Any]],
    min_response_chars: int,
    min_quality_score: float,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    seen = set()

    stats = {
        "seen_events": 0,
        "kept_events": 0,
        "filtered_empty": 0,
        "filtered_short": 0,
        "filtered_domain": 0,
        "filtered_low_quality": 0,
        "filtered_score": 0,
        "filtered_duplicate": 0,
    }

    for event in events:
        stats["seen_events"] += 1
        user_text = (event.get("user_text") or "").strip()
        assistant_text = (event.get("assistant_text") or "").strip()

        if not user_text or not assistant_text:
            stats["filtered_empty"] += 1
            continue

        if len(assistant_text) < min_response_chars:
            stats["filtered_short"] += 1
            continue

        if not event.get("in_domain", False):
            stats["filtered_domain"] += 1
            continue

        if event.get("is_low_quality", False):
            stats["filtered_low_quality"] += 1
            continue

        quality = _event_quality(event, min_response_chars=min_response_chars)
        if quality < min_quality_score:
            stats["filtered_score"] += 1
            continue

        key = (user_text.lower(), assistant_text.lower())
        if key in seen:
            stats["filtered_duplicate"] += 1
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": user_text,
                "response": assistant_text,
                "history": (event.get("summary") or "")[:1200],
                "source": f"feedback_q{quality:.2f}",
            }
        )
        stats["kept_events"] += 1

    return rows, stats


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(rows: list[dict[str, str]], max_rows: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= max_rows:
        return list(rows)
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    picked = [rows[i] for i in idx[:max_rows]]
    return picked


def main() -> None:
    args = parse_args()

    chat_log_path = Path(args.chat_log)
    audit_log_path = Path(args.audit_log)
    signature_path = Path(args.signature_dataset)
    out_feedback_path = Path(args.out_feedback)
    out_bundle_path = Path(args.out_bundle)
    out_report_path = Path(args.out_report)

    events = _read_jsonl(chat_log_path)
    audits = _read_jsonl(audit_log_path)

    feedback_rows, feedback_stats = _extract_feedback_rows(
        events=events,
        min_response_chars=max(1, args.min_response_chars),
        min_quality_score=max(0.0, min(1.0, args.min_quality_score)),
    )

    feedback_rows = _sample_rows(feedback_rows, max_rows=max(1, args.max_feedback_rows), seed=args.seed)
    _write_csv(out_feedback_path, feedback_rows)

    signature_rows = _read_csv_rows(signature_path)
    signature_rows = [
        {
            "instruction": (r.get("instruction") or "").strip(),
            "response": (r.get("response") or "").strip(),
            "history": (r.get("history") or "").strip(),
            "source": (r.get("source") or "signature").strip() or "signature",
        }
        for r in signature_rows
        if (r.get("instruction") or "").strip() and (r.get("response") or "").strip()
    ]
    signature_rows = _sample_rows(signature_rows, max_rows=max(1, args.max_signature_rows), seed=args.seed)

    bundle_rows = signature_rows + feedback_rows

    # Keep deterministic ordering with signature first then feedback.
    _write_csv(out_bundle_path, bundle_rows)

    if not args.allow_empty and not bundle_rows:
        raise RuntimeError("No rows generated for training bundle. Use --allow-empty to bypass.")

    report = {
        "inputs": {
            "chat_log": str(chat_log_path),
            "audit_log": str(audit_log_path),
            "signature_dataset": str(signature_path),
        },
        "outputs": {
            "feedback_dataset": str(out_feedback_path),
            "weekly_bundle": str(out_bundle_path),
        },
        "counts": {
            "chat_events": len(events),
            "audit_events": len(audits),
            "feedback_rows": len(feedback_rows),
            "signature_rows": len(signature_rows),
            "bundle_rows": len(bundle_rows),
        },
        "feedback_filter_stats": feedback_stats,
        "policy": {
            "min_response_chars": args.min_response_chars,
            "min_quality_score": args.min_quality_score,
            "max_feedback_rows": args.max_feedback_rows,
            "max_signature_rows": args.max_signature_rows,
        },
        "training_hint": {
            "weekly": "Fine-tune with weekly_bundle (LoRA/light).",
            "monthly": "Regenerate signature dataset and refresh bundle policy.",
            "quarterly": "Add new modules and evaluation scenarios.",
        },
    }

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
