import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FR chatbot model on held-out samples")
    parser.add_argument("--model-dir", default="models/chatbot-fr-flan-t5-small-v1")
    parser.add_argument("--data-file", default="data/processed/chatbot_train_fr.csv")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out-json", default="reports/eval_chatbot_fr_v1.json")
    parser.add_argument("--out-csv", default="reports/eval_chatbot_fr_v1_samples.csv")
    return parser.parse_args()


def token_f1(pred: str, ref: str) -> float:
    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0

    p_counts = {}
    r_counts = {}
    for t in p_tokens:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in r_tokens:
        r_counts[t] = r_counts.get(t, 0) + 1

    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, r_counts.get(t, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def load_samples(path: str, k: int, seed: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = (row.get("instruction") or "").strip()
            r = (row.get("response") or "").strip()
            if i and r:
                rows.append({"instruction": i, "response": r})

    random.seed(seed)
    random.shuffle(rows)
    return rows[: min(k, len(rows))]


def main() -> None:
    args = parse_args()
    samples = load_samples(args.data_file, args.samples, args.seed)
    if not samples:
        raise RuntimeError("No evaluation samples found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    per_sample = []
    f1_scores = []
    em_scores = []

    for idx, row in enumerate(samples, start=1):
        prompt = f"Utilisateur: {row['instruction']}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        ref = row["response"]

        f1 = token_f1(pred, ref)
        em = exact_match(pred, ref)
        f1_scores.append(f1)
        em_scores.append(em)

        per_sample.append(
            {
                "id": idx,
                "instruction": row["instruction"],
                "reference": ref,
                "prediction": pred,
                "token_f1": round(f1, 4),
                "exact_match": int(em),
            }
        )

    report = {
        "model_dir": args.model_dir,
        "data_file": args.data_file,
        "samples": len(samples),
        "token_f1_mean": round(mean(f1_scores), 4),
        "exact_match_rate": round(mean(em_scores), 4),
        "device": device,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "instruction", "reference", "prediction", "token_f1", "exact_match"],
        )
        writer.writeheader()
        writer.writerows(per_sample)

    print(report)
    print(f"Saved sample predictions to {out_csv}")
    print(f"Saved summary report to {out_json}")


if __name__ == "__main__":
    main()
