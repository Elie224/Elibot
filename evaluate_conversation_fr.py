import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn conversation stability and memory")
    parser.add_argument("--model-dir", default="models/chatbot-fr-flan-t5-small-v2-signature")
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument(
        "--history-mode",
        choices=["full", "user-only"],
        default="user-only",
    )
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--use-slot-memory", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default="Tu es un assistant francophone utile, coherent et concis.")
    parser.add_argument("--out-json", default="reports/eval_conversation_fr.json")
    parser.add_argument("--out-csv", default="reports/eval_conversation_fr_details.csv")
    return parser.parse_args()


def update_profile_from_user_text(user_text: str, profile: dict[str, str]) -> None:
    text = user_text.strip()

    name_match = re.search(
        r"(?:je m'appelle|mon prenom est|mon prénom est|je suis)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        profile["prenom"] = name_match.group(1)

    city_match = re.search(
        r"(?:je viens de|j'habite a|j'habite à|je vis(?: actuellement)? a|je vis(?: actuellement)? à|ma ville est)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if city_match:
        profile["ville"] = city_match.group(1)

    asks_for_sport = bool(re.search(r"\bquel(?:le)?\s+sport\b", text, flags=re.IGNORECASE))
    if not asks_for_sport:
        sport_match = re.search(
            r"(?:j'aime le|j'aime la|j'adore le|j'adore la|mon sport prefere est le|mon sport prefere est la|mon sport prefere c'est le|mon sport prefere c'est la|mon sport préféré est le|mon sport préféré est la|mon sport préféré c'est le|mon sport préféré c'est la)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
            text,
            flags=re.IGNORECASE,
        )
        if not sport_match:
            sport_match = re.search(
                r"(?:mon sport prefere|mon sport préféré)\s*[,;:\-]?\s*(?:est|c'est)?\s*(?:le|la)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
                text,
                flags=re.IGNORECASE,
            )
        if sport_match:
            profile["sport"] = sport_match.group(1)

    goal_match = re.search(r"(?:objectif|courir|course|entrainement|entraînement|seance|séance)?.{0,20}?(\d+)\s*minutes", text, flags=re.IGNORECASE)
    if goal_match:
        profile["objectif_minutes"] = goal_match.group(1)


def build_memory_lines(profile: dict[str, str]) -> list[str]:
    lines = []
    if "prenom" in profile:
        lines.append(f"Memoire: Le prenom utilisateur est {profile['prenom']}.")
    if "ville" in profile:
        lines.append(f"Memoire: La ville utilisateur est {profile['ville']}.")
    if "sport" in profile:
        lines.append(f"Memoire: Le sport prefere utilisateur est {profile['sport']}.")
    if "objectif_minutes" in profile:
        lines.append(f"Memoire: L'objectif sport utilisateur est {profile['objectif_minutes']} minutes.")
    return lines


def maybe_rule_reply(user_text: str, profile: dict[str, str]) -> str | None:
    q = user_text.lower()

    asks_name = ("prenom" in q) or ("prénom" in q) or ("comment je m'appelle" in q) or ("qui suis-je" in q)
    asks_city = ("ville" in q) or ("j'habite" in q) or ("je vis" in q)
    asks_sport = ("sport" in q) or ("j'aime" in q)
    asks_goal = ("objectif" in q) or ("course" in q) or ("entrain" in q)
    asks_advice = ("conseil" in q) or ("demain" in q)

    if asks_name and asks_city:
        if "prenom" in profile and "ville" in profile:
            return f"Tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return None

    if asks_name and "prenom" in profile:
        return f"Ton prenom est {profile['prenom']}."

    if asks_city and "ville" in profile:
        return f"Ta ville est {profile['ville']}."

    if asks_advice and "sport" in profile:
        return (
            f"Conseil pour demain: fais 30 minutes de {profile['sport']}, "
            "commence par 5 minutes d'echauffement, et termine par des etirements."
        )

    if asks_sport and "sport" in profile:
        return f"Ton sport prefere est le {profile['sport']}."

    if asks_goal and "objectif_minutes" in profile:
        return f"Ton objectif est de courir {profile['objectif_minutes']} minutes."

    return None


def build_prompt(
    system_prompt: str,
    history: list[tuple[str, str]],
    user_text: str,
    history_turns: int,
    history_mode: str,
    profile: dict[str, str],
    use_slot_memory: bool,
) -> str:
    turns = max(0, history_turns)
    recent_messages = history[-(turns * 2) :] if turns > 0 else []

    lines = [f"Systeme: {system_prompt}"]
    if use_slot_memory:
        lines.extend(build_memory_lines(profile))
    for role, text in recent_messages:
        if history_mode == "user-only" and role != "Utilisateur":
            continue
        lines.append(f"{role}: {text}")
    lines.append(f"Utilisateur: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_scenarios() -> list[dict]:
    return [
        {
            "name": "memoire_identite_localisation",
            "turns": [
                {"user": "Salut, je m'appelle Karim Benali.", "expect": None},
                {"user": "Je vis actuellement a Lyon.", "expect": None},
                {"user": "Tu peux me dire comment je m'appelle ?", "expect": ["karim"]},
                {"user": "Et tu te souviens de ma ville ?", "expect": ["lyon"]},
            ],
        },
        {
            "name": "memoire_preferences_sport",
            "turns": [
                {"user": "Mon sport prefere, c'est le foot.", "expect": None},
                {"user": "Je m'entraine surtout le dimanche matin.", "expect": None},
                {"user": "Quel sport j'aime le plus ?", "expect": ["foot"]},
            ],
        },
        {
            "name": "suivi_objectif_entrainement",
            "turns": [
                {"user": "Demain, objectif: courir pendant 30 minutes.", "expect": None},
                {"user": "Je veux aussi penser a bien m'hydrater.", "expect": None},
                {"user": "Tu peux me rappeler mon objectif pour demain ?", "expect": ["30", "minutes"]},
            ],
        },
    ]


def contains_expected(answer: str, expected: list[str]) -> int:
    normalized = answer.lower()
    return int(all(token.lower() in normalized for token in expected))


def generate_answer(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    no_repeat_ngram_size: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "no_repeat_ngram_size": max(0, no_repeat_ngram_size),
    }

    if temperature > 0:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        })
    else:
        generate_kwargs.update({"do_sample": False})

    with torch.no_grad():
        out_ids = model.generate(**inputs, **generate_kwargs)

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    scenarios = build_scenarios()

    details = []
    scenario_scores = []

    for scenario in scenarios:
        history: list[tuple[str, str]] = []
        profile: dict[str, str] = {}
        checks = []

        for idx, turn in enumerate(scenario["turns"], start=1):
            if args.use_slot_memory:
                update_profile_from_user_text(turn["user"], profile)

            direct = maybe_rule_reply(turn["user"], profile) if args.use_slot_memory else None

            if direct is not None:
                answer = direct
            else:
                prompt = build_prompt(
                    args.system_prompt,
                    history,
                    turn["user"],
                    args.history_turns,
                    args.history_mode,
                    profile,
                    args.use_slot_memory,
                )
                answer = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    max_input_length=args.max_input_length,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )

            history.append(("Utilisateur", turn["user"]))
            history.append(("Assistant", answer))

            score = None
            if turn["expect"]:
                score = contains_expected(answer, turn["expect"])
                checks.append(score)

            details.append(
                {
                    "scenario": scenario["name"],
                    "turn": idx,
                    "user": turn["user"],
                    "assistant": answer,
                    "expected_tokens": " | ".join(turn["expect"]) if turn["expect"] else "",
                    "memory_ok": "" if score is None else str(score),
                }
            )

        scenario_score = mean(checks) if checks else 0.0
        scenario_scores.append(scenario_score)

    memory_recall = mean(scenario_scores) if scenario_scores else 0.0

    report = {
        "model_dir": args.model_dir,
        "device": device,
        "history_turns": args.history_turns,
        "use_slot_memory": args.use_slot_memory,
        "scenarios": len(scenarios),
        "memory_recall_rate": round(memory_recall, 4),
        "scenario_scores": {
            scenario["name"]: round(score, 4)
            for scenario, score in zip(scenarios, scenario_scores)
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
            fieldnames=["scenario", "turn", "user", "assistant", "expected_tokens", "memory_ok"],
        )
        writer.writeheader()
        writer.writerows(details)

    print(report)
    print(f"Saved conversation details to {out_csv}")
    print(f"Saved summary report to {out_json}")


if __name__ == "__main__":
    main()
