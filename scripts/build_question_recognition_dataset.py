import csv
import random
from pathlib import Path

OUT = Path("data/processed/chatbot_train_fr_question_recognition.csv")

SEED = 42

BASE_QUESTIONS = [
    "c koi le machine learning",
    "tu px expliquer precision recall",
    "pk mon modele overfit",
    "comment je choisis ridge lasso",
    "j ai un doute sur la p value",
    "on fait comment pour detecter drift",
    "je comprends pas auc",
    "quelle metrique si classes desequilibrees",
    "aide moi pour arima",
    "test stationnarite adf kpss comment",
    "mon modele diverge quoi faire",
    "faut normaliser ou standardiser",
    "je veux une explication du recall",
    "xgboost ou random forest pour churn",
    "comment evaluer en prod",
    "peux tu me dire c est quoi leakage",
    "jhesite sur le seuil de decision",
    "calibrer probabilites comment",
    "mon score offline bon mais prod nul pourquoi",
    "debug pipeline ml stp",
    "comment choisir entre ridge et lasso",
    "pourquoi mon modele diverge",
    "quel seuil utiliser pour la fraude",
    "peux tu expliquer le recall",
    "quoi faire si perte validation monte",
    "ca veut dire quoi auc",
    "quelle difference entre precision et recall",
    "quand utiliser standardisation",
    "quand utiliser normalisation",
    "faut il recalibrer les probabilites",
    "on fait comment un backtesting temporel",
    "quel test pour stationnarite",
    "comment verifier la robustesse",
    "quelles features ajouter en priorite",
    "je dois faire du target encoding",
    "comment eviter fuite target encoding",
    "pourquoi exact match bas mais f1 ok",
    "quelle strategie pour classes rares",
]

PREFIXES = [
    "Utilisateur:",
    "Demande:",
    "Question:",
    "Besoin:",
    "Contexte client:",
    "Ticket:",
    "Prompt:",
    "Input:",
    "",
]

SUFFIXES = [
    "",
    " stp",
    " urgent",
    " en prod",
    " pour demain",
    " besoin rapide",
    " pls",
    " asap",
    " ???",
    " ??",
    " ?",
]

CHAT_NOISE_PREFIXES = [
    "yo",
    "salut",
    "hello",
    "svp",
    "stp",
    "bro",
    "coach",
    "assistant",
    "",
]

QUESTION_OPENERS = [
    "Question detectee:",
    "Type: question utilisateur",
    "Je reconnais une question ML.",
    "Demande comprise: question.",
]

ACTION_TEMPLATES = [
    "Je reponds clairement en 4 blocs: definition, causes, methode, checklist.",
    "Je donne une reponse structuree: idee cle, diagnostic, action immediate, verification.",
    "Je fournis une explication actionnable avec etapes et exemple concret.",
]

HISTORY_OPTIONS = [
    "Utilisateur: formulation courte avec fautes et sans ponctuation. ||| Assistant: je dois reconnaitre une question puis repondre de facon utile.",
    "Utilisateur: message bruité, abreviations, style chat. ||| Assistant: detecter la question et proposer une reponse structuree.",
    "Utilisateur: besoin ML en contexte prod. ||| Assistant: confirmer la question et donner des etapes pragmatiques.",
]


def _compact_spaces(text: str) -> str:
    return " ".join(text.split())


def _noisy_variant(rng: random.Random, q: str) -> str:
    q2 = q
    replacements = [
        ("pourquoi", "pq"),
        ("comment", "comt"),
        ("probabilites", "probas"),
        ("modele", "model"),
        ("evaluation", "eval"),
    ]
    for old, new in replacements:
        if rng.random() < 0.25:
            q2 = q2.replace(old, new)
    if rng.random() < 0.30:
        q2 = q2.replace(" ", "") if len(q2) < 28 else q2
    if rng.random() < 0.40:
        q2 = q2.lower()
    return q2


def make_instruction(rng: random.Random, q: str) -> str:
    prefix = rng.choice(PREFIXES)
    noise = rng.choice(CHAT_NOISE_PREFIXES)
    suffix = rng.choice(SUFFIXES)
    q_form = _noisy_variant(rng, q)

    chunks = []
    if prefix:
        chunks.append(prefix)
    if noise:
        chunks.append(noise)
    chunks.append(q_form)
    if suffix:
        chunks.append(suffix)

    text = _compact_spaces(" ".join(chunks))
    if rng.random() < 0.35 and "?" not in text:
        text += " ?"
    return _compact_spaces(text)


def make_response(rng: random.Random, q: str) -> str:
    opener = rng.choice(QUESTION_OPENERS)
    action = rng.choice(ACTION_TEMPLATES)
    return (
        f"{opener}\n"
        f"Question comprise: {q}\n"
        "Reponse: je traite cette demande comme une question d information ML.\n"
        f"{action}\n"
        "Etapes: 1) clarifier le besoin 2) expliquer 3) proposer une action 4) verifier en pratique."
    )


def main() -> None:
    rng = random.Random(SEED)

    rows = []
    for _ in range(120):
        for q in BASE_QUESTIONS:
            rows.append(
                {
                    "instruction": make_instruction(rng, q),
                    "response": make_response(rng, q),
                    "history": rng.choice(HISTORY_OPTIONS),
                    "source": "question_recognition_booster_v2",
                }
            )

    # Stable dedupe by normalized instruction text.
    dedup = {}
    for row in rows:
        key = _compact_spaces(row["instruction"].lower())
        dedup.setdefault(key, row)

    final_rows = list(dedup.values())
    final_rows.sort(key=lambda r: r["instruction"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"saved={OUT}")
    print(f"rows={len(final_rows)}")
    print("source=question_recognition_booster_v2")


if __name__ == "__main__":
    main()
