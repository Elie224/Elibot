import argparse
import csv
import random
from pathlib import Path


NAMES = [
    "Karim",
    "Lea",
    "Nadia",
    "Yanis",
    "Sofia",
    "Rachid",
    "Ines",
    "Amine",
]

CITIES = [
    "Lyon",
    "Paris",
    "Marseille",
    "Lille",
    "Toulouse",
    "Nantes",
]

SPORTS = [
    "foot",
    "tennis",
    "natation",
    "basket",
    "running",
    "velo",
]

DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic FR memory dataset for multi-turn fine-tuning")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_memory_synth.csv")
    return parser.parse_args()


def make_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n):
        name = random.choice(NAMES)
        city = random.choice(CITIES)
        sport = random.choice(SPORTS)
        minutes = random.choice([20, 30, 40, 45, 60])
        day = random.choice(DAYS)

        history = [
            f"Utilisateur: Bonjour, je m'appelle {name}.",
            "Assistant: Enchante, je retiens ton prenom.",
            f"Utilisateur: J'habite a {city}.",
            "Assistant: D'accord, je retiens ta ville.",
            f"Utilisateur: Mon sport prefere est le {sport}.",
            "Assistant: Parfait, je retiens ton sport prefere.",
            f"Utilisateur: Demain je veux m'entrainer {minutes} minutes le {day}.",
            "Assistant: Tres bien, je retiens ton objectif sportif.",
        ]

        sample_type = random.randint(1, 5)
        if sample_type == 1:
            instruction = "Quel est mon prenom ?"
            response = f"Ton prenom est {name}."
        elif sample_type == 2:
            instruction = "Dans quelle ville j'habite ?"
            response = f"Tu habites a {city}."
        elif sample_type == 3:
            instruction = "Quel est mon sport prefere ?"
            response = f"Ton sport prefere est le {sport}."
        elif sample_type == 4:
            instruction = "Rappelle mon objectif d'entrainement."
            response = f"Ton objectif est de t'entrainer {minutes} minutes le {day}."
        else:
            instruction = "Fais un recapitulatif de ce que tu sais sur moi."
            response = (
                f"Tu t'appelles {name}, tu habites a {city}, ton sport prefere est le {sport}, "
                f"et tu veux t'entrainer {minutes} minutes le {day}."
            )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": " ||| ".join(history),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = make_rows(args.rows, args.seed)

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out_file": str(out)})


if __name__ == "__main__":
    main()
