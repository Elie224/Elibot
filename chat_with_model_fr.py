import argparse
import json
import os
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DOMAIN_TOPICS = [
    "analyse de donnees",
    "machine learning",
    "ia appliquee",
    "automatisation",
    "pipelines",
    "api",
]

IN_DOMAIN_KEYWORDS = {
    "data", "donnee", "donnees", "dataset", "csv", "json", "excel", "table", "sql",
    "pandas", "numpy", "analyse", "nettoyage", "feature", "visualisation", "statistique",
    "ml", "ia", "ai", "machine learning", "modele", "model", "entrainement", "evaluation",
    "classification", "regression", "prediction", "prompt", "llm", "token", "embedding",
    "pipeline", "workflow", "automatisation", "script", "python", "fastapi", "api", "docker",
}

OUT_DOMAIN_KEYWORDS = {
    "medecine", "medical", "maladie", "diagnostic", "traitement", "politique", "election",
    "religion", "psychologie", "depression", "amour", "relation", "sexe", "voyance", "astrologie",
    "finance personnelle", "pari", "bet", "casino", "juridique", "avocat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the trained French chatbot model")
    parser.add_argument("--model-dir", default="models/chatbot-fr-flan-t5-small-v2-convfix")
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument(
        "--history-mode",
        choices=["full", "user-only"],
        default="user-only",
    )
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--use-slot-memory", action="store_true")
    parser.add_argument("--disable-rule-replies", action="store_true")
    parser.add_argument("--save-profile", action="store_true")
    parser.add_argument("--profile-path", default="data/memory/chat_profile_fr.json")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default=(
            "Tu es Elibot, un assistant specialise en analyse de donnees, IA appliquee et automatisation. "
            "Tu reponds de facon claire, concise et professionnelle. "
            "Tu refuses poliment les sujets hors domaine et rediriges vers une demande technique."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def load_profile(profile_path: str) -> dict[str, str]:
    if not os.path.exists(profile_path):
        return {}
    try:
        with open(profile_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def save_profile(profile_path: str, profile: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def clean_generated_text(text: str) -> str:
    value = " ".join(text.strip().split())
    if not value:
        return value

    # Remove obvious repeated sentence loops.
    parts = re.split(r"(?<=[.!?])\s+", value)
    dedup_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if dedup_parts and part.lower() == dedup_parts[-1].lower():
            continue
        dedup_parts.append(part)
    value = " ".join(dedup_parts)

    # Collapse runs of identical tokens beyond 2 repetitions.
    tokens = value.split()
    collapsed: list[str] = []
    run_token = ""
    run_count = 0
    for tok in tokens:
        key = tok.lower()
        if key == run_token:
            run_count += 1
        else:
            run_token = key
            run_count = 1
        if run_count <= 2:
            collapsed.append(tok)
    value = " ".join(collapsed)

    if value and value[-1] not in ".!?":
        value += "."
    return value


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", value.lower())).strip()


def is_in_domain_query(user_text: str) -> bool:
    text = _normalize_text(user_text)
    if not text:
        return True

    # Keep short social openers in-domain to avoid robotic UX.
    if text in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc", "merci"}:
        return True

    if any(k in text for k in OUT_DOMAIN_KEYWORDS):
        return False

    return any(k in text for k in IN_DOMAIN_KEYWORDS)


def out_of_domain_reply() -> str:
    topics = ", ".join(DOMAIN_TOPICS)
    return (
        "Je suis specialise en "
        f"{topics}. "
        "Je ne traite pas les sujets hors de ce cadre. "
        "Envoie une question technique (ex: pipeline ML, API FastAPI, nettoyage de dataset) et je t'aide."
    )


def is_low_quality_answer(answer: str) -> bool:
    text = " ".join(answer.strip().lower().split())
    if not text:
        return True

    # Catch unstable patterns frequently produced by the base model.
    bad_patterns = [
        "je ne peux pas te dire que tu veux dire",
        "je veux dire que je n'ai pas besoin",
        "je ne sais pas. je veux dire",
        "je n'ai pas l'air",
        "je veux dire que",
    ]
    if any(p in text for p in bad_patterns):
        return True

    if re.search(r"\b(de\s+te\s+dire\s+que\s+tu|vous\s+avez\s+une\s+idee\s+de)\b", text):
        return True

    # If a sentence is repeated verbatim, treat as low quality.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if len(parts) >= 2 and len(set(parts)) < len(parts):
        return True

    # Extremely short low-information outputs.
    if text in {"je ne sais pas.", "je sais pas.", "ok.", "d'accord."}:
        return True

    # Detect odd self-contradictory structures often seen in bad generations.
    if text.count("je ne") >= 2 and "je veux dire" in text:
        return True

    return False


def fallback_reply(user_text: str) -> str:
    q = user_text.strip().lower()
    if not is_in_domain_query(q):
        return out_of_domain_reply()
    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Dis-moi ce que tu veux faire et je t'aide."
    if "comment" in q and "ca va" in q:
        return "Je vais bien, merci. Et toi, comment tu vas ?"
    return "Je n'ai pas bien compris. Peux-tu reformuler en une phrase simple et precise ?"


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
        r"(?:je viens de|j'habite a|j'habite à|je vis a|je vis à|ma ville est)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if city_match:
        profile["ville"] = city_match.group(1)

    sport_match = re.search(
        r"(?:j'aime le|j'aime la|j'adore le|j'adore la|mon sport prefere est le|mon sport prefere est la|mon sport prefere c'est le|mon sport prefere c'est la|mon sport préféré est le|mon sport préféré est la|mon sport préféré c'est le|mon sport préféré c'est la)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if not sport_match:
        sport_match = re.search(
            r"(?:mon sport prefere|mon sport préféré)\s*(?:est|c'est)?\s*(?:le|la)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
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
    q = user_text.lower().strip()
    q_norm = re.sub(r"[^a-zà-öø-ÿ0-9\s]", "", q)

    if not is_in_domain_query(q):
        return out_of_domain_reply()

    if q_norm in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    if q in {"je comprends pas", "je ne comprends pas", "j'ai pas compris"}:
        return "Pas de souci. Dis-moi juste ce que tu veux faire en une phrase simple, et je t'aide."

    if "je m'appelle" in q or q.startswith("je suis "):
        if "prenom" in profile and "ville" in profile:
            return f"Enchanté {profile['prenom']}. J'ai noté que tu viens de {profile['ville']}."
        if "prenom" in profile:
            return f"Enchanté {profile['prenom']}. Ravi de faire ta connaissance."
        return "Ravi de te connaitre."

    if ("comment" in q and "ca va" in q) or ("comment" in q and "ça va" in q):
        return "Je vais bien, merci. Et toi, comment tu vas ?"

    if "merci" in q:
        return "Avec plaisir. Si tu veux, on continue."

    if "que fais tu" in q_norm or "tu fais quoi" in q_norm or "ton domaine" in q:
        return (
            "Je suis specialise en analyse de donnees, IA appliquee et automatisation. "
            "Je peux t'aider sur pipelines, modeles, API, debugging Python et workflows techniques."
        )

    asks_name = ("prenom" in q) or ("prénom" in q) or ("comment je m'appelle" in q) or ("qui suis-je" in q)
    asks_city = ("ville" in q) or ("j'habite" in q) or ("je vis" in q)
    asks_sport = ("sport" in q) or ("j'aime" in q)
    asks_goal = ("objectif" in q) or ("course" in q) or ("entrain" in q)
    asks_advice = ("conseil" in q) or ("demain" in q)

    if ("qui es-tu" in q) or ("qui es tu" in q) or ("ton nom" in q) or ("comment tu t'appelles" in q):
        return "Je m'appelle Elibot. Je suis la pour discuter avec toi de facon naturelle et utile."

    if asks_name and asks_city:
        if "prenom" in profile and "ville" in profile:
            return f"Si je me souviens bien, tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return None

    if asks_name and "prenom" in profile:
        return f"Bien sur: tu t'appelles {profile['prenom']}."

    if asks_city and "ville" in profile:
        return f"Tu m'avais dit que tu viens de {profile['ville']}."

    if asks_advice and "sport" in profile:
        return (
            f"Top, on fait simple pour demain: 30 minutes de {profile['sport']}, "
            "5 minutes d'echauffement au debut, puis etirements en fin de seance."
        )

    if asks_sport and "sport" in profile:
        return f"Ton sport prefere, c'est le {profile['sport']}."

    if asks_goal and "objectif_minutes" in profile:
        return f"Ton objectif, c'est de courir {profile['objectif_minutes']} minutes."

    if ("qui suis-je" in q or "qui je suis" in q) and "prenom" in profile:
        if "ville" in profile:
            return f"Tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return f"Tu t'appelles {profile['prenom']}."

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


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    history: list[tuple[str, str]] = []
    profile: dict[str, str] = load_profile(args.profile_path) if args.save_profile else {}

    print("Elibot est pret. Tape 'exit' pour quitter, '/reset' pour effacer la memoire, '/profile' pour l'afficher.")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        if user_text.lower() == "/profile":
            if profile:
                print(f"Bot: Memoire active: {profile}\n")
            else:
                print("Bot: Memoire vide.\n")
            continue
        if user_text.lower() == "/reset":
            history.clear()
            profile.clear()
            if args.save_profile and os.path.exists(args.profile_path):
                os.remove(args.profile_path)
            print("Bot: Conversation memory cleared.\n")
            continue

        if args.use_slot_memory:
            update_profile_from_user_text(user_text, profile)
            if args.save_profile:
                save_profile(args.profile_path, profile)

        if not args.disable_rule_replies:
            direct = maybe_rule_reply(user_text, profile)
            if direct:
                history.append(("Utilisateur", user_text))
                history.append(("Assistant", direct))
                print(f"Bot: {direct}\n")
                continue

        prompt = build_prompt(
            system_prompt=args.system_prompt,
            history=history,
            user_text=user_text,
            history_turns=args.history_turns,
            history_mode=args.history_mode,
            profile=profile,
            use_slot_memory=args.use_slot_memory,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)

        with torch.no_grad():
            use_sampling = args.temperature > 0
            generate_kwargs = {
                "do_sample": use_sampling,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": max(0, args.no_repeat_ngram_size),
                "max_new_tokens": args.max_new_tokens,
                "num_beams": 1,
            }
            if use_sampling:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p

            output_ids = model.generate(
                **inputs,
                **generate_kwargs,
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if args.clean_output:
            answer = clean_generated_text(answer)
        if is_low_quality_answer(answer):
            answer = fallback_reply(user_text)
        history.append(("Utilisateur", user_text))
        history.append(("Assistant", answer))
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
