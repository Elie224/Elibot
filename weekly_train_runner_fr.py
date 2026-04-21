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
    parser.add_argument("--core-datasets", nargs="*", default=["data/processed/chatbot_train_fr_core_intelligence.csv"])
    parser.add_argument("--ml-deep-datasets", nargs="*", default=[])
    parser.add_argument("--signature-datasets", nargs="*", default=["data/processed/chatbot_train_fr_signature_v2_domain.csv"])
    parser.add_argument("--direct-dataset", default="data/processed/chatbot_train_fr_direct_answers.csv")
    parser.add_argument("--conversation-base-dataset", default="data/processed/chatbot_train_fr_conversation_base.csv")
    parser.add_argument("--simple-detailed-dataset", default="data/processed/chatbot_train_fr_simple_detailed.csv")
    parser.add_argument("--multiturn-dataset", default="data/processed/chatbot_train_fr_multiturn_contextual.csv")
    parser.add_argument("--goal-following-dataset", default="data/processed/chatbot_train_fr_goal_following.csv")
    parser.add_argument("--question-recognition-dataset", default="data/processed/chatbot_train_fr_question_recognition.csv")
    parser.add_argument("--style-signature-dataset", default="data/processed/chatbot_train_fr_style_signature.csv")
    parser.add_argument("--signature-long-expert-dataset", default="data/processed/chatbot_train_fr_signature_long_expert.csv")
    parser.add_argument("--ml-concepts-dataset", default="data/processed/chatbot_train_fr_ml_concepts_deep.csv")
    parser.add_argument("--ml-algorithms-dataset", default="data/processed/chatbot_train_fr_ml_algorithms_deep.csv")
    parser.add_argument("--ml-learning-types-dataset", default="data/processed/chatbot_train_fr_ml_learning_types.csv")
    parser.add_argument("--ml-implementation-dataset", default="data/processed/chatbot_train_fr_ml_implementation.csv")
    parser.add_argument("--ml-visualizations-dataset", default="data/processed/chatbot_train_fr_ml_visualizations.csv")
    parser.add_argument("--ml-business-applied-dataset", default="data/processed/chatbot_train_fr_ml_business_applied.csv")
    parser.add_argument("--ml-applications-real-dataset", default="data/processed/chatbot_train_fr_ml_applications_real.csv")
    parser.add_argument("--ml-culture-dataset", default="data/processed/chatbot_train_fr_ml_culture_generale.csv")
    parser.add_argument("--ml-essentials-dataset", default="data/processed/chatbot_train_fr_ml_200_essentials.csv")
    parser.add_argument("--ml-advanced-dataset", default="data/processed/chatbot_train_fr_ml_150_advanced.csv")
    parser.add_argument("--ml-classical-dataset", default="data/processed/chatbot_train_fr_ml_120_classical_advanced.csv")
    parser.add_argument("--ml-classical-series4-dataset", default="data/processed/chatbot_train_fr_ml_120_classical_series4.csv")
    parser.add_argument("--agent-datasets", nargs="*", default=["data/processed/chatbot_train_fr_agent_actions_tools.csv"])
    parser.add_argument("--explanations-dataset", default="data/processed/chatbot_train_fr_explanations_detailed.csv")
    parser.add_argument("--gold-dataset", default="data/processed/chatbot_train_fr_gold_concepts.csv")
    parser.add_argument(
        "--memory-datasets",
        nargs="*",
        default=[
            "data/processed/chatbot_train_fr_memory_context_summary.csv",
            "data/processed/chatbot_train_fr_memory_synth.csv",
            "data/processed/chatbot_train_fr_memory_city_focus.csv",
        ],
    )
    parser.add_argument("--bundle-target-rows", type=int, default=82500)
    parser.add_argument("--vision-profile", choices=["balanced", "strict"], default="balanced")
    parser.add_argument("--ratio-core", type=float, default=None)
    parser.add_argument("--ratio-ml-deep", type=float, default=None)
    parser.add_argument("--ratio-signature", type=float, default=None)
    parser.add_argument("--ratio-agent", type=float, default=None)
    parser.add_argument("--ratio-memory", type=float, default=None)
    parser.add_argument("--strict-domain", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-core-rows", type=int, default=198000)
    parser.add_argument("--max-ml-deep-rows", type=int, default=132000)
    parser.add_argument("--max-signature-rows", type=int, default=99000)
    parser.add_argument("--max-agent-rows", type=int, default=99000)
    parser.add_argument("--max-memory-rows", type=int, default=66000)
    parser.add_argument("--dedupe-bundle", action="store_true")

    # Step 2: training
    parser.add_argument("--base-model", default="models/chatbot-fr-flan-t5-small-v2-signature")
    parser.add_argument("--output-model", default="models/chatbot-fr-flan-t5-small-weekly")
    parser.add_argument(
        "--resume-from-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled and output model exists, continue training from output model instead of base-model.",
    )
    parser.add_argument("--train-max-samples", type=int, default=115500)
    parser.add_argument("--train-max-eval-samples", type=int, default=8250)
    parser.add_argument("--train-epochs", type=float, default=0.8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-grad-accum", type=int, default=2)
    parser.add_argument("--train-learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-log-steps", type=int, default=50)
    parser.add_argument("--train-heartbeat-file", default="reports/training_heartbeat.json")
    parser.add_argument("--train-heartbeat-interval-seconds", type=int, default=30)
    parser.add_argument("--train-heartbeat-stale-seconds", type=int, default=300)
    parser.add_argument("--train-interrupted-report", default="reports/training_interrupted.json")
    parser.add_argument("--train-interrupted-checkpoint-dir", default="")

    # Step 3: evaluation
    parser.add_argument("--eval-samples", type=int, default=150)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--eval-data-file",
        default="data/eval/weekly_holdout_fr.csv",
        help="Held-out CSV for evaluation. Must be different from training bundle unless --allow-eval-on-train is set.",
    )
    parser.add_argument(
        "--allow-eval-on-train",
        action="store_true",
        help="Allow fallback evaluation on training bundle when held-out file is missing.",
    )
    parser.add_argument("--eval-out-json", default="reports/eval_weekly_latest.json")
    parser.add_argument("--eval-out-csv", default="reports/eval_weekly_latest_samples.csv")

    # Step 4: business benchmark
    parser.add_argument(
        "--run-business-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run fixed business benchmark after evaluation.",
    )
    parser.add_argument("--benchmark-cases-file", default="data/eval/business_benchmark_fr.json")
    parser.add_argument("--benchmark-max-cases", type=int, default=0)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=128)
    parser.add_argument("--benchmark-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--benchmark-no-repeat-ngram", type=int, default=3)
    parser.add_argument("--benchmark-out-json", default="reports/eval_business_weekly.json")
    parser.add_argument("--benchmark-out-csv", default="reports/eval_business_weekly_samples.csv")

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
    print(f"[runner] START: {cmd_display}", flush=True)
    try:
        # Stream child output directly to console so long steps remain visible.
        proc = subprocess.run(cmd, text=True)
    except KeyboardInterrupt:
        duration = time.time() - started
        return {
            "command": cmd,
            "command_display": cmd_display,
            "ok": False,
            "returncode": 130,
            "stdout": "",
            "stderr": "Interrupted by user (KeyboardInterrupt)",
            "duration_s": round(duration, 3),
        }
    duration = time.time() - started
    print(f"[runner] END: rc={proc.returncode} in {round(duration, 3)}s", flush=True)

    stderr_hint = ""
    if proc.returncode == 4294967295:
        stderr_hint = (
            "Process was likely terminated externally (Windows rc=4294967295). "
            "Check for concurrent runs or manual process termination."
        )

    return {
        "command": cmd,
        "command_display": cmd_display,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": "[streamed to console]",
        "stderr": stderr_hint,
        "duration_s": round(duration, 3),
    }


def main() -> None:
    args = parse_args()

    ratio_core = 0.40 if args.ratio_core is None else args.ratio_core
    ratio_ml_deep = 0.20 if args.ratio_ml_deep is None else args.ratio_ml_deep
    ratio_signature = 0.20 if args.ratio_signature is None else args.ratio_signature
    ratio_agent = 0.10 if args.ratio_agent is None else args.ratio_agent
    ratio_memory = 0.10 if args.ratio_memory is None else args.ratio_memory
    strict_domain = False if args.strict_domain is None else bool(args.strict_domain)

    if args.vision_profile == "strict":
        ratio_core = 0.40 if args.ratio_core is None else args.ratio_core
        ratio_ml_deep = 0.12 if args.ratio_ml_deep is None else args.ratio_ml_deep
        ratio_signature = 0.28 if args.ratio_signature is None else args.ratio_signature
        ratio_agent = 0.10 if args.ratio_agent is None else args.ratio_agent
        ratio_memory = 0.10 if args.ratio_memory is None else args.ratio_memory
        strict_domain = True if args.strict_domain is None else bool(args.strict_domain)

    run_started = _now_iso()
    report: dict[str, Any] = {
        "started_at": run_started,
        "dry_run": bool(args.dry_run),
        "steps": [],
        "status": "running",
        "training_observability": {
            "heartbeat_file": args.train_heartbeat_file,
            "heartbeat_stale_seconds": args.train_heartbeat_stale_seconds,
            "interrupted_report": args.train_interrupted_report,
        },
    }

    core_datasets = list(args.core_datasets)
    ml_deep_datasets = list(args.ml_deep_datasets)
    signature_datasets = list(args.signature_datasets)
    if args.explanations_dataset and Path(args.explanations_dataset).exists():
        core_datasets.append(args.explanations_dataset)
    if args.gold_dataset and Path(args.gold_dataset).exists():
        core_datasets.append(args.gold_dataset)
    if args.simple_detailed_dataset and Path(args.simple_detailed_dataset).exists():
        core_datasets.append(args.simple_detailed_dataset)
    if args.multiturn_dataset and Path(args.multiturn_dataset).exists():
        core_datasets.append(args.multiturn_dataset)
    if args.goal_following_dataset and Path(args.goal_following_dataset).exists():
        core_datasets.append(args.goal_following_dataset)
    if args.question_recognition_dataset and Path(args.question_recognition_dataset).exists():
        core_datasets.append(args.question_recognition_dataset)
    if args.ml_concepts_dataset and Path(args.ml_concepts_dataset).exists():
        ml_deep_datasets.append(args.ml_concepts_dataset)
    if args.ml_algorithms_dataset and Path(args.ml_algorithms_dataset).exists():
        ml_deep_datasets.append(args.ml_algorithms_dataset)
    if args.ml_learning_types_dataset and Path(args.ml_learning_types_dataset).exists():
        ml_deep_datasets.append(args.ml_learning_types_dataset)
    if args.ml_implementation_dataset and Path(args.ml_implementation_dataset).exists():
        ml_deep_datasets.append(args.ml_implementation_dataset)
    if args.ml_visualizations_dataset and Path(args.ml_visualizations_dataset).exists():
        ml_deep_datasets.append(args.ml_visualizations_dataset)
    if args.ml_business_applied_dataset and Path(args.ml_business_applied_dataset).exists():
        ml_deep_datasets.append(args.ml_business_applied_dataset)
    if args.ml_applications_real_dataset and Path(args.ml_applications_real_dataset).exists():
        ml_deep_datasets.append(args.ml_applications_real_dataset)
    if args.ml_culture_dataset and Path(args.ml_culture_dataset).exists():
        ml_deep_datasets.append(args.ml_culture_dataset)
    if args.ml_essentials_dataset and Path(args.ml_essentials_dataset).exists():
        ml_deep_datasets.append(args.ml_essentials_dataset)
    if args.ml_advanced_dataset and Path(args.ml_advanced_dataset).exists():
        ml_deep_datasets.append(args.ml_advanced_dataset)
    if args.ml_classical_dataset and Path(args.ml_classical_dataset).exists():
        ml_deep_datasets.append(args.ml_classical_dataset)
    if args.ml_classical_series4_dataset and Path(args.ml_classical_series4_dataset).exists():
        ml_deep_datasets.append(args.ml_classical_series4_dataset)
    if args.direct_dataset and Path(args.direct_dataset).exists():
        signature_datasets.append(args.direct_dataset)
    if args.conversation_base_dataset and Path(args.conversation_base_dataset).exists():
        signature_datasets.append(args.conversation_base_dataset)
    if args.style_signature_dataset and Path(args.style_signature_dataset).exists():
        signature_datasets.append(args.style_signature_dataset)
    if args.signature_long_expert_dataset and Path(args.signature_long_expert_dataset).exists():
        signature_datasets.append(args.signature_long_expert_dataset)

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
        "--core-datasets",
        *core_datasets,
        "--ml-deep-datasets",
        *ml_deep_datasets,
        "--signature-datasets",
        *signature_datasets,
        "--agent-datasets",
        *args.agent_datasets,
        "--memory-datasets",
        *args.memory_datasets,
        "--bundle-target-rows",
        str(args.bundle_target_rows),
        "--ratio-core",
        str(ratio_core),
        "--ratio-ml-deep",
        str(ratio_ml_deep),
        "--ratio-signature",
        str(ratio_signature),
        "--ratio-agent",
        str(ratio_agent),
        "--ratio-memory",
        str(ratio_memory),
        "--max-core-rows",
        str(args.max_core_rows),
        "--max-ml-deep-rows",
        str(args.max_ml_deep_rows),
        "--max-signature-rows",
        str(args.max_signature_rows),
        "--max-agent-rows",
        str(args.max_agent_rows),
        "--max-memory-rows",
        str(args.max_memory_rows),
        "--seed",
        str(args.pipeline_seed),
    ]
    step1.append("--dedupe-bundle")
    if strict_domain:
        step1.append("--strict-domain")
    if args.allow_empty:
        step1.append("--allow-empty")

    model_for_training = args.base_model
    if args.resume_from_output and Path(args.output_model).exists():
        model_for_training = args.output_model
        print(
            f"[runner] INFO: resume enabled, using existing output model as base: {model_for_training}",
            flush=True,
        )

    step2 = [
        args.python_exec,
        "train_chatbot_fr.py",
        "--data-file",
        args.bundle_out,
        "--model-name",
        model_for_training,
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
        "--heartbeat-file",
        args.train_heartbeat_file,
        "--heartbeat-interval-seconds",
        str(args.train_heartbeat_interval_seconds),
        "--heartbeat-stale-seconds",
        str(args.train_heartbeat_stale_seconds),
        "--interrupted-report",
        args.train_interrupted_report,
    ]
    if args.train_interrupted_checkpoint_dir:
        step2.extend(["--interrupted-checkpoint-dir", args.train_interrupted_checkpoint_dir])

    eval_data_file = args.eval_data_file
    eval_data_exists = Path(eval_data_file).exists()
    if not eval_data_exists:
        if args.allow_eval_on_train:
            eval_data_file = args.bundle_out
            report["evaluation_data_warning"] = {
                "warning": "eval_on_train_bundle",
                "requested_eval_data_file": args.eval_data_file,
                "fallback_eval_data_file": eval_data_file,
            }
        else:
            report["status"] = "failed"
            report["failed_step"] = "evaluation"
            report["failure_reason"] = f"eval_data_file_not_found:{args.eval_data_file}"
            out = Path(args.run_report)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "report": str(out),
                        "failure_reason": report["failure_reason"],
                    },
                    ensure_ascii=False,
                )
            )
            raise SystemExit(1)

    step3 = [
        args.python_exec,
        "evaluate_model_fr.py",
        "--model-dir",
        args.output_model,
        "--data-file",
        eval_data_file,
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

    step4 = [
        args.python_exec,
        "evaluate_business_benchmark_fr.py",
        "--model-dir",
        args.output_model,
        "--cases-file",
        args.benchmark_cases_file,
        "--max-cases",
        str(args.benchmark_max_cases),
        "--max-new-tokens",
        str(args.benchmark_max_new_tokens),
        "--repetition-penalty",
        str(args.benchmark_repetition_penalty),
        "--no-repeat-ngram",
        str(args.benchmark_no_repeat_ngram),
        "--out-json",
        args.benchmark_out_json,
        "--out-csv",
        args.benchmark_out_csv,
    ]

    steps: list[tuple[str, list[str]]] = [
        ("feeding_pipeline", step1),
        ("training", step2),
        ("evaluation", step3),
    ]
    if args.run_business_benchmark:
        steps.append(("business_benchmark", step4))

    for name, cmd in steps:
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
