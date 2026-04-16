import argparse
import math
import os
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

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Missing dataset file: {args.data_file}")

    dataset = load_dataset("csv", data_files={"train": args.data_file})["train"]
    dataset = dataset.filter(
        lambda x: bool((x.get("instruction") or "").strip()) and bool((x.get("response") or "").strip())
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

    model.train()
    pbar = tqdm(total=total_updates, desc="Training updates")
    update_step = 0
    accum_counter = 0
    running_loss = 0.0

    while update_step < total_updates:
        for batch_idx, batch in enumerate(train_loader):
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

    model.eval()
    eval_losses: List[float] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            eval_losses.append(float(loss.item()))

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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


if __name__ == "__main__":
    main()
