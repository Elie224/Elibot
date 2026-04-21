import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_BAD_MARKERS = [
    "navigation privee",
    "extension",
    "mot de passe",
    "incognito",
    "compte bloque",
    "sur le sport",
    "journal de recherche",
    "je ne sais pas",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on fixed business benchmark prompts")
    parser.add_argument("--model-dir", default="models/chatbot-fr-flan-t5-small-weekly")
    parser.add_argument("--cases-file", default="data/eval/business_benchmark_fr.json")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--out-json", default="reports/eval_business_weekly.json")
    parser.add_argument("--out-csv", default="reports/eval_business_weekly_samples.csv")
    return parser.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _load_cases(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[dict] = []
    for row in raw:
        q = (row.get("question") or "").strip()
        kws = [str(x).strip().lower() for x in row.get("expected_keywords", []) if str(x).strip()]
        if q and kws:
            cases.append({"question": q, "expected_keywords": kws})
    if not cases:
        raise RuntimeError("No valid benchmark cases found")
    return cases


def _keyword_coverage(answer_l: str, keywords: list[str]) -> tuple[int, float, list[str]]:
    matched = [kw for kw in keywords if kw in answer_l]
    hits = len(matched)
    coverage = hits / len(keywords) if keywords else 0.0
    return hits, coverage, matched


def _bad_marker_hit(answer_l: str) -> tuple[bool, str]:
    for marker in DEFAULT_BAD_MARKERS:
        if marker in answer_l:
            return True, marker
    return False, ""


def _score_case(answer: str, keywords: list[str]) -> dict:
    answer_l = _normalize(answer)
    hits, coverage, matched = _keyword_coverage(answer_l, keywords)
    bad_hit, bad_marker = _bad_marker_hit(answer_l)

    length_ok = 1.0 if 80 <= len(answer.strip()) <= 1200 else 0.0
    safety_score = 0.0 if bad_hit else 1.0

    final_score = (0.70 * coverage) + (0.20 * safety_score) + (0.10 * length_ok)
    passed = final_score >= 0.60

    return {
        "keyword_hits": hits,
        "keyword_total": len(keywords),
        "keyword_coverage": round(coverage, 4),
        "matched_keywords": matched,
        "bad_marker_hit": int(bad_hit),
        "bad_marker": bad_marker,
        "length_chars": len(answer),
        "length_ok": int(length_ok),
        "score": round(final_score, 4),
        "passed": int(passed),
    }


def main() -> None:
    args = parse_args()
    cases = _load_cases(args.cases_file)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows: list[dict] = []
    scores: list[float] = []
    pass_flags: list[int] = []
    coverage_values: list[float] = []

    for idx, case in enumerate(cases, start=1):
        question = case["question"]
        keywords = case["expected_keywords"]

        prompt = f"Utilisateur: {question}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=max(0, args.no_repeat_ngram),
            )

        prediction = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        score_data = _score_case(prediction, keywords)

        row = {
            "id": idx,
            "question": question,
            "prediction": prediction,
            "expected_keywords": ", ".join(keywords),
            "matched_keywords": ", ".join(score_data["matched_keywords"]),
            "keyword_hits": score_data["keyword_hits"],
            "keyword_total": score_data["keyword_total"],
            "keyword_coverage": score_data["keyword_coverage"],
            "bad_marker_hit": score_data["bad_marker_hit"],
            "bad_marker": score_data["bad_marker"],
            "length_chars": score_data["length_chars"],
            "length_ok": score_data["length_ok"],
            "score": score_data["score"],
            "passed": score_data["passed"],
        }
        rows.append(row)
        scores.append(score_data["score"])
        pass_flags.append(score_data["passed"])
        coverage_values.append(score_data["keyword_coverage"])

    report = {
        "model_dir": args.model_dir,
        "cases_file": args.cases_file,
        "cases": len(rows),
        "device": device,
        "score_mean": round(mean(scores), 4),
        "keyword_coverage_mean": round(mean(coverage_values), 4),
        "pass_rate": round(mean(pass_flags), 4),
        "bad_marker_count": int(sum(r["bad_marker_hit"] for r in rows)),
        "policy": {
            "score_formula": "0.70*keyword_coverage + 0.20*safety + 0.10*length_ok",
            "pass_threshold": 0.60,
            "bad_markers": DEFAULT_BAD_MARKERS,
        },
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "prediction",
                "expected_keywords",
                "matched_keywords",
                "keyword_hits",
                "keyword_total",
                "keyword_coverage",
                "bad_marker_hit",
                "bad_marker",
                "length_chars",
                "length_ok",
                "score",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(report)
    print(f"Saved business benchmark samples to {out_csv}")
    print(f"Saved business benchmark summary to {out_json}")


if __name__ == "__main__":
    main()
