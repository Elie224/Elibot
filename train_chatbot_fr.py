import argparse
import json
import math
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)


FORBIDDEN_DATASET_PATH_HINTS = [
    "chatbot_train_fr.csv",
    "chatbot_pairs_fr.csv",
    "chatbot_pairs_en.csv",
]

FORBIDDEN_SOURCE_HINTS = [
    "cornell_movie_dialogs",
    "dialog_csv",
    "movie",
]


class TrainingInterrupted(Exception):
    pass


class HeartbeatWriter:
    def __init__(self, heartbeat_file: str, interval_s: int = 30) -> None:
        self.heartbeat_file = heartbeat_file
        self.interval_s = max(5, int(interval_s))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._lock = threading.Lock()
        self._state = {
            "status": "starting",
            "phase": "init",
            "update_step": 0,
            "total_updates": 0,
            "pid": os.getpid(),
        }

    def start(self) -> None:
        Path(self.heartbeat_file).parent.mkdir(parents=True, exist_ok=True)
        self.write_once()
        self._thread.start()

    def stop(self, status: str, phase: str) -> None:
        self.update(status=status, phase=phase)
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.write_once()

    def update(self, **kwargs: object) -> None:
        with self._lock:
            self._state.update(kwargs)

    def write_once(self) -> None:
        with self._lock:
            payload = dict(self._state)
        payload["ts_utc"] = datetime.now(timezone.utc).isoformat()
        Path(self.heartbeat_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.write_once()


def _save_with_fallback(model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if attempt > 1:
                print({"save_retry_succeeded": attempt})
            return
        except Exception as exc:  # pragma: no cover - depends on host FS behavior
            last_err = exc
            print({"save_retry_failed": attempt, "error": str(exc)})
            time.sleep(1.5)

    # Fallback for Windows mapped-file lock issues with safetensors.
    try:
        print({"save_fallback": "safe_serialization=False"})
        model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        return
    except Exception as exc:  # pragma: no cover - depends on host FS behavior
        if last_err is not None:
            print({"save_initial_error": str(last_err)})
        raise exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a French chatbot model from CSV pairs")
    parser.add_argument("--data-file", default="data/processed/chatbot_train_fr.csv")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default="models/chatbot-fr-flan-t5-small")
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    parser.add_argument("--history-column", default="history")
    parser.add_argument(
        "--prompt-style",
        choices=["single-turn", "dialogue"],
        default="single-turn",
    )
    parser.add_argument(
        "--system-prompt",
        default="Tu es un assistant francophone utile, coherent et concis.",
    )
    parser.add_argument("--heartbeat-file", default="reports/training_heartbeat.json")
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=30)
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=300)
    parser.add_argument("--interrupted-report", default="reports/training_interrupted.json")
    parser.add_argument("--interrupted-checkpoint-dir", default="")
    return parser.parse_args()


def tokenize_batch(
    batch: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: int,
    history_column: str,
    prompt_style: str,
    system_prompt: str,
) -> Dict[str, List[List[int]]]:
    prompts = []
    history_values = batch.get(history_column, [""] * len(batch["instruction"]))

    for instruction, history_raw in zip(batch["instruction"], history_values):
        if prompt_style == "dialogue":
            lines = [f"Systeme: {system_prompt}"]

            history_text = (history_raw or "").strip()
            if history_text:
                for chunk in history_text.split("|||"):
                    clean = chunk.strip()
                    if clean:
                        lines.append(clean)

            lines.append(f"Utilisateur: {instruction}")
            lines.append("Assistant:")
            prompts.append("\n".join(lines))
        else:
            prompts.append(f"Utilisateur: {instruction}\nAssistant:")

    responses = [x for x in batch["response"]]

    model_inputs = tokenizer(
        prompts,
        text_target=responses,
        max_length=max_input_length,
        max_target_length=max_target_length,
        truncation=True,
    )

    return model_inputs


def main() -> None:
    args = parse_args()
    hb = HeartbeatWriter(args.heartbeat_file, args.heartbeat_interval_seconds)

    stop_requested = {"value": False, "reason": ""}

    def _handle_stop(signum: int, _frame: object) -> None:
        stop_requested["value"] = True
        stop_requested["reason"] = f"signal_{signum}"
        hb.update(status="stop_requested", phase="training", stop_reason=stop_requested["reason"])
        print({"training_stop_requested": stop_requested["reason"]})

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    hb.start()

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Missing dataset file: {args.data_file}")

    normalized_path = args.data_file.replace("\\", "/").lower()
    if any(hint in normalized_path for hint in FORBIDDEN_DATASET_PATH_HINTS):
        raise ValueError(
            "Forbidden dataset for training: movie/dialogue legacy corpus is disabled. "
            f"Use a specialized bundle instead of '{args.data_file}'."
        )

    try:
        dataset = load_dataset("csv", data_files={"train": args.data_file})["train"]
        dataset = dataset.filter(
            lambda x: bool((x.get("instruction") or "").strip()) and bool((x.get("response") or "").strip())
        )

        if "source" in dataset.column_names:
            sample_size = min(2000, len(dataset))
            source_values = dataset.select(range(sample_size))["source"]
            for source in source_values:
                src = str(source or "").lower()
                if any(h in src for h in FORBIDDEN_SOURCE_HINTS):
                    raise ValueError(
                        "Forbidden source detected in dataset (movie/dialogue legacy). "
                        f"Found source='{source}'."
                    )

        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

        if args.max_train_samples is not None:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

        if args.max_eval_samples is not None:
            eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        hb.update(status="running", phase="tokenize")
        train_tokenized = train_ds.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "history_column": args.history_column,
            "prompt_style": args.prompt_style,
            "system_prompt": args.system_prompt,
        },
        remove_columns=train_ds.column_names,
        desc="Tokenizing train split",
    )

        eval_tokenized = eval_ds.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "history_column": args.history_column,
            "prompt_style": args.prompt_style,
            "system_prompt": args.system_prompt,
        },
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval split",
    )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        train_loader = DataLoader(
        train_tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
        eval_loader = DataLoader(
        eval_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
        total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
        warmup_steps = int(total_updates * args.warmup_ratio)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )

        hb.update(status="running", phase="training", update_step=0, total_updates=total_updates)
        model.train()
        pbar = tqdm(total=total_updates, desc="Training updates")
        update_step = 0
        accum_counter = 0
        running_loss = 0.0

        while update_step < total_updates:
            for batch_idx, batch in enumerate(train_loader):
                if stop_requested["value"]:
                    raise TrainingInterrupted(stop_requested["reason"] or "signal")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / max(1, args.grad_accum)
                loss.backward()
                running_loss += float(loss.item())
                accum_counter += 1

                is_last_batch = (batch_idx + 1) == len(train_loader)

                if accum_counter >= args.grad_accum or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                    update_step += 1
                    pbar.update(1)
                    hb.update(status="running", phase="training", update_step=update_step)

                    if update_step % max(1, args.log_steps) == 0:
                        avg = running_loss / max(1, args.log_steps)
                        current_lr = lr_scheduler.get_last_lr()[0]
                        print({"update_step": update_step, "avg_loss": round(avg, 4), "lr": round(current_lr, 8)})
                        running_loss = 0.0

                    if update_step >= total_updates:
                        break
            else:
                continue
            break

        pbar.close()

        hb.update(status="running", phase="evaluation")
        model.eval()
        eval_losses: List[float] = []
        with torch.no_grad():
            for batch in eval_loader:
                if stop_requested["value"]:
                    raise TrainingInterrupted(stop_requested["reason"] or "signal")
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                eval_losses.append(float(loss.item()))

        hb.update(status="running", phase="saving")
        _save_with_fallback(model, tokenizer, args.output_dir)

        eval_loss = sum(eval_losses) / max(1, len(eval_losses))
        print(
            {
                "device": device,
                "train_samples": len(train_tokenized),
                "eval_samples": len(eval_tokenized),
                "total_updates": total_updates,
                "eval_loss": round(eval_loss, 4),
                "saved_model": args.output_dir,
            }
        )
        hb.stop(status="ok", phase="done")
    except TrainingInterrupted as interrupted:
        interrupted_checkpoint_dir = args.interrupted_checkpoint_dir.strip() if args.interrupted_checkpoint_dir else ""
        if not interrupted_checkpoint_dir:
            interrupted_checkpoint_dir = f"{args.output_dir}_interrupted"

        interruption_payload = {
            "status": "interrupted",
            "reason": str(interrupted),
            "pid": os.getpid(),
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "heartbeat_file": args.heartbeat_file,
            "checkpoint_dir": interrupted_checkpoint_dir,
        }

        try:
            hb.update(status="interrupted", phase="saving_interrupt_checkpoint", stop_reason=str(interrupted))
            _save_with_fallback(model, tokenizer, interrupted_checkpoint_dir)
            interruption_payload["checkpoint_saved"] = True
        except Exception as exc:  # pragma: no cover - host specific FS/GPU behavior
            interruption_payload["checkpoint_saved"] = False
            interruption_payload["checkpoint_error"] = str(exc)

        interrupted_report_path = Path(args.interrupted_report)
        interrupted_report_path.parent.mkdir(parents=True, exist_ok=True)
        interrupted_report_path.write_text(
            json.dumps(interruption_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        hb.stop(status="interrupted", phase="done")
        print(interruption_payload)
        raise SystemExit(143)
    except Exception:
        hb.stop(status="failed", phase="error")
        raise


if __name__ == "__main__":
    main()
