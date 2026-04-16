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
    "Mehdi",
    "Sarah",
]

CITIES = [
    "Lyon",
    "Paris",
    "Marseille",
    "Lille",
    "Toulouse",
    "Nantes",
    "Bordeaux",
    "Nice",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build focused FR memory dataset (name/city)")
    parser.add_argument("--rows", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_memory_city_focus.csv")
    return parser.parse_args()


def make_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []

    for _ in range(n):
        name = random.choice(NAMES)
        city = random.choice(CITIES)

        history = [
            f"Utilisateur: Salut, je m'appelle {name}.",
            "Assistant: Enchante, je retiens ton prenom.",
            f"Utilisateur: Je viens de {city}.",
            "Assistant: Parfait, je retiens ta ville.",
        ]

        t = random.randint(1, 6)
        if t in {1, 2, 3}:
            instruction = "Tu peux me rappeler mon prenom ?"
            response = f"Ton prenom est {name}."
        elif t in {4, 5}:
            instruction = "Et ma ville ?"
            response = f"Ta ville est {city}."
        else:
            instruction = "Rappelle mon prenom et ma ville."
            response = f"Tu t'appelles {name} et ta ville est {city}."

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
