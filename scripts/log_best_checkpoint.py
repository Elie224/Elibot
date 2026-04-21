import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EPS = 1e-9


@dataclass
class EvalMetrics:
    token_f1_mean: float
    exact_match_rate: float
    score_mean: float
    pass_rate: float
    keyword_coverage_mean: float
    memory_recall_rate: float
    question_recognition_rate: float

    @classmethod
    def from_reports(
        cls,
        eval_json_path: Path,
        business_json_path: Path,
        conversation_json_path: Path,
        question_json_path: Path | None = None,
    ) -> "EvalMetrics":
        eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
        business_data = json.loads(business_json_path.read_text(encoding="utf-8"))
        conversation_data = json.loads(conversation_json_path.read_text(encoding="utf-8"))
        question_data: dict[str, Any] = {}
        if question_json_path is not None and question_json_path.exists():
            question_data = json.loads(question_json_path.read_text(encoding="utf-8"))

        return cls(
            token_f1_mean=float(eval_data.get("token_f1_mean", 0.0) or 0.0),
            exact_match_rate=float(eval_data.get("exact_match_rate", 0.0) or 0.0),
            score_mean=float(business_data.get("score_mean", 0.0) or 0.0),
            pass_rate=float(business_data.get("pass_rate", 0.0) or 0.0),
            keyword_coverage_mean=float(business_data.get("keyword_coverage_mean", 0.0) or 0.0),
            memory_recall_rate=float(conversation_data.get("memory_recall_rate", 0.0) or 0.0),
            question_recognition_rate=float(question_data.get("recognition_rate", 0.0) or 0.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "token_f1_mean": self.token_f1_mean,
            "exact_match_rate": self.exact_match_rate,
            "score_mean": self.score_mean,
            "pass_rate": self.pass_rate,
            "keyword_coverage_mean": self.keyword_coverage_mean,
            "memory_recall_rate": self.memory_recall_rate,
            "question_recognition_rate": self.question_recognition_rate,
        }

    def composite(self) -> float:
        # Prioritize business quality while keeping language quality and exactness in the loop.
        return (
            0.40 * self.score_mean
            + 0.20 * self.pass_rate
            + 0.25 * self.token_f1_mean
            + 0.10 * self.keyword_coverage_mean
            + 0.05 * self.exact_match_rate
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log weekly checkpoint when a new best score is detected")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--eval-json", default="reports/eval_weekly_latest.json")
    parser.add_argument("--business-json", default="reports/eval_business_weekly.json")
    parser.add_argument("--conversation-json", default="reports/eval_conversation_weekly.json")
    parser.add_argument(
        "--question-json",
        default="",
        help="Optional question-recognition evaluation report used for additional promotion constraints.",
    )
    parser.add_argument("--run-report", default="reports/weekly_train_runner_report.json")
    parser.add_argument("--feeding-report", default="reports/feeding_pipeline_report.json")
    parser.add_argument("--model-dir", default="models/chatbot-fr-flan-t5-small-weekly")
    parser.add_argument("--tag", default="weekly_scheduled")
    parser.add_argument("--latest-best-json", default="reports/best_checkpoint_latest.json")
    parser.add_argument("--history-jsonl", default="reports/best_checkpoint_history.jsonl")
    parser.add_argument("--snapshots-dir", default="reports/best_snapshots")
    parser.add_argument("--min-composite-improvement", type=float, default=0.002)
    parser.add_argument(
        "--lane",
        choices=["global", "question"],
        default="global",
        help="Promotion lane policy. 'global' prioritizes overall quality; 'question' prioritizes question recognition.",
    )
    parser.add_argument(
        "--primary-metric",
        choices=["composite", "question_recognition_rate"],
        default="composite",
        help="Primary metric used for best-checkpoint improvement decision.",
    )
    parser.add_argument(
        "--min-primary-improvement",
        type=float,
        default=None,
        help="Minimum required improvement on --primary-metric versus previous best.",
    )
    parser.add_argument(
        "--allow-memory-regression",
        action="store_true",
        help="If set, do not block promotion when memory_recall_rate decreases.",
    )
    parser.add_argument("--max-score-regression", type=float, default=0.005)
    parser.add_argument("--max-pass-rate-regression", type=float, default=0.02)
    parser.add_argument(
        "--min-question-recognition-rate",
        type=float,
        default=0.0,
        help="Minimum required recognition_rate from --question-json when provided.",
    )
    parser.add_argument(
        "--max-question-regression",
        type=float,
        default=0.05,
        help="Maximum allowed drop vs previous best question_recognition_rate when both are available.",
    )
    return parser.parse_args()


def _resolve(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return root / p


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_new_best(
    current: EvalMetrics,
    previous: EvalMetrics | None,
    lane: str,
    primary_metric: str,
    min_primary_improvement: float,
    allow_memory_regression: bool,
    max_score_regression: float,
    max_pass_rate_regression: float,
    min_question_recognition_rate: float,
    max_question_regression: float,
) -> tuple[bool, str]:
    if current.question_recognition_rate + EPS < min_question_recognition_rate:
        return False, "question_recognition_below_min_threshold"

    if previous is None:
        return True, "first_best"

    cur_comp = current.composite()
    prev_comp = previous.composite()

    if (not allow_memory_regression) and current.memory_recall_rate + EPS < previous.memory_recall_rate:
        return False, "memory_regression"

    if lane == "global" and current.score_mean + max_score_regression + EPS < previous.score_mean:
        return False, "score_regression_too_high"

    if lane == "global" and current.pass_rate + max_pass_rate_regression + EPS < previous.pass_rate:
        return False, "pass_rate_regression_too_high"

    if current.question_recognition_rate + max_question_regression + EPS < previous.question_recognition_rate:
        return False, "question_recognition_regression_too_high"

    if primary_metric == "question_recognition_rate":
        cur_primary = current.question_recognition_rate
        prev_primary = previous.question_recognition_rate
    else:
        cur_primary = cur_comp
        prev_primary = prev_comp

    if cur_primary > prev_primary + min_primary_improvement:
        return True, f"{primary_metric}_improved"

    return False, "no_significant_improvement"


def main() -> None:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()

    eval_json = _resolve(project_root, args.eval_json)
    business_json = _resolve(project_root, args.business_json)
    conversation_json = _resolve(project_root, args.conversation_json)
    question_json = _resolve(project_root, args.question_json) if args.question_json else None
    run_report_path = _resolve(project_root, args.run_report)
    feeding_report_path = _resolve(project_root, args.feeding_report)
    latest_best_path = _resolve(project_root, args.latest_best_json)
    history_jsonl_path = _resolve(project_root, args.history_jsonl)
    snapshots_dir = _resolve(project_root, args.snapshots_dir)

    current = EvalMetrics.from_reports(eval_json, business_json, conversation_json, question_json)
    previous_payload = _load_json_if_exists(latest_best_path)
    previous_metrics = None
    if previous_payload and isinstance(previous_payload.get("metrics"), dict):
        m = previous_payload["metrics"]
        previous_metrics = EvalMetrics(
            token_f1_mean=float(m.get("token_f1_mean", 0.0) or 0.0),
            exact_match_rate=float(m.get("exact_match_rate", 0.0) or 0.0),
            score_mean=float(m.get("score_mean", 0.0) or 0.0),
            pass_rate=float(m.get("pass_rate", 0.0) or 0.0),
            keyword_coverage_mean=float(m.get("keyword_coverage_mean", 0.0) or 0.0),
            memory_recall_rate=float(m.get("memory_recall_rate", 0.0) or 0.0),
            question_recognition_rate=float(m.get("question_recognition_rate", 0.0) or 0.0),
        )

    min_primary_improvement = (
        args.min_primary_improvement
        if args.min_primary_improvement is not None
        else args.min_composite_improvement
    )

    is_best, reason = _is_new_best(
        current=current,
        previous=previous_metrics,
        lane=args.lane,
        primary_metric=args.primary_metric,
        min_primary_improvement=min_primary_improvement,
        allow_memory_regression=bool(args.allow_memory_regression),
        max_score_regression=args.max_score_regression,
        max_pass_rate_regression=args.max_pass_rate_regression,
        min_question_recognition_rate=args.min_question_recognition_rate,
        max_question_regression=args.max_question_regression,
    )

    run_report = _load_json_if_exists(run_report_path) or {}
    feeding_report = _load_json_if_exists(feeding_report_path) or {}

    payload: dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "tag": args.tag,
        "model_dir": args.model_dir,
        "metrics": current.as_dict(),
        "composite_score": round(current.composite(), 6),
        "decision": {
            "is_new_best": bool(is_best),
            "reason": reason,
            "lane": args.lane,
            "primary_metric": args.primary_metric,
            "min_primary_improvement": min_primary_improvement,
            "allow_memory_regression": bool(args.allow_memory_regression),
            "min_composite_improvement": args.min_composite_improvement,
            "max_score_regression": args.max_score_regression,
            "max_pass_rate_regression": args.max_pass_rate_regression,
            "min_question_recognition_rate": args.min_question_recognition_rate,
            "max_question_regression": args.max_question_regression,
        },
        "run_report": {
            "status": run_report.get("status"),
            "steps": run_report.get("steps"),
        },
        "bundle_counts": feeding_report.get("counts", {}),
        "bundle_policy": feeding_report.get("policy", {}),
    }

    latest_best_path.parent.mkdir(parents=True, exist_ok=True)
    history_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    history_jsonl_path.write_text("", encoding="utf-8") if not history_jsonl_path.exists() else None
    with history_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    snapshot_path = None
    if is_best:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshots_dir / f"best_{stamp}.json"
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "is_new_best": is_best,
                "reason": reason,
                "composite_score": round(current.composite(), 6),
                "latest_best_path": str(latest_best_path),
                "history_jsonl_path": str(history_jsonl_path),
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
