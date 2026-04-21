import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROMPTS = [
    "c koi le machine learning",
    "tu px expliquer precision recall",
    "pk mon modele overfit",
    "comment je choisis ridge lasso",
    "j'ai un doute sur la p value",
    "on fait comment pour detecter drift",
    "je comprends pas auc",
    "quelle metrique si classes desiquilibrees",
    "aide moi pour arima",
    "test stationnarite adf kpss comment",
    "mon modele diverge quoi faire",
    "faut normaliser ou standardiser",
    "je veux une explication du recall",
    "xgboost ou random forest pour churn",
    "comment evaluer en prod",
    "peux tu me dire c'est quoi leakage",
    "jhesite sur le seuil de decision",
    "calibrer probabilites comment",
    "mon score offline bon mais prod nul pourquoi",
    "debug pipeline ml stp",
]

POSITIVE_HINTS = [
    "question",
    "explication",
    "demande",
    "pourquoi",
    "comment",
    "voici",
    "reponse",
    "etape",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted question-recognition benchmark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional text file (one prompt per line) to override built-in prompt list.",
    )
    parser.add_argument(
        "--train-dataset",
        default="",
        help="Optional CSV path to compute prompt overlap against training instructions.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _normalize_soft(text: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _load_prompts(path: str) -> list[str]:
    if not path:
        return list(PROMPTS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompts_file_not_found:{path}")
    items = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise ValueError(f"prompts_file_empty:{path}")
    return items


def _compute_overlap(prompts: list[str], train_dataset: str) -> dict[str, float | int] | None:
    if not train_dataset:
        return None

    ds_path = Path(train_dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"train_dataset_not_found:{train_dataset}")

    with ds_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        train_instructions = [row.get("instruction", "") for row in reader]

    train_norm = [_normalize_soft(x) for x in train_instructions if (x or "").strip()]
    train_set = set(train_norm)

    prompt_norm = [_normalize_soft(x) for x in prompts]
    exact_overlap = sum(1 for x in prompt_norm if x in train_set)

    def jaccard(a: str, b: str) -> float:
        sa = set(a.split())
        sb = set(b.split())
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    high_overlap = 0
    for p in prompt_norm:
        best = 0.0
        for t in train_norm:
            score = jaccard(p, t)
            if score > best:
                best = score
            if best >= 0.70:
                break
        if best >= 0.70:
            high_overlap += 1

    total = max(1, len(prompt_norm))
    return {
        "train_dataset": train_dataset,
        "prompts": len(prompt_norm),
        "exact_norm_overlap": exact_overlap,
        "high_jaccard_overlap": high_overlap,
        "exact_norm_overlap_rate": round(exact_overlap / total, 4),
        "high_jaccard_overlap_rate": round(high_overlap / total, 4),
    }


def _recognition_score(prediction: str) -> tuple[float, int, list[str]]:
    p = _normalize(prediction)
    hits = [h for h in POSITIVE_HINTS if h in p]
    hint_score = min(1.0, len(hits) / 3.0)

    length_score = 1.0 if len(p) >= 35 else 0.0
    blank_penalty = 0.0 if p else 1.0

    # weighted score in [0, 1]
    score = max(0.0, (0.75 * hint_score) + (0.25 * length_score) - (0.5 * blank_penalty))
    recognized = int(score >= 0.60)
    return round(score, 4), recognized, hits


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args.prompts_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    scores = []
    recognized_flags = []

    for idx, prompt in enumerate(prompts, start=1):
        text = f"Utilisateur: {prompt}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        prediction = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        score, recognized, hits = _recognition_score(prediction)

        rows.append(
            {
                "id": idx,
                "prompt": prompt,
                "prediction": prediction,
                "recognition_score": score,
                "recognized": recognized,
                "hint_hits": ", ".join(hits),
            }
        )
        scores.append(score)
        recognized_flags.append(recognized)

    summary = {
        "model_dir": args.model_dir,
        "device": device,
        "cases": len(rows),
        "prompts_file": args.prompts_file or "<builtin>",
        "recognition_score_mean": round(mean(scores), 4),
        "recognition_rate": round(mean(recognized_flags), 4),
        "policy": {
            "recognized_if": "score >= 0.60",
            "score_formula": "0.75*hint_score + 0.25*length_score - 0.5*blank_penalty",
            "positive_hints": POSITIVE_HINTS,
        },
    }

    overlap = _compute_overlap(prompts, args.train_dataset)
    if overlap is not None:
        summary["train_overlap"] = overlap

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "prompt", "prediction", "recognition_score", "recognized", "hint_hits"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
