import argparse
import csv
import random
from pathlib import Path


NAMES = ["Karim", "Lea", "Nadia", "Yanis", "Sofia", "Rachid", "Ines", "Amine"]
CITIES = ["Lyon", "Paris", "Marseille", "Lille", "Toulouse", "Nantes"]
SPORTS = ["foot", "tennis", "natation", "basket", "running", "velo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a French style-polish dataset for chatbot")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_style_polish.csv")
    return parser.parse_args()


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n):
        name = random.choice(NAMES)
        city = random.choice(CITIES)
        sport = random.choice(SPORTS)

        history = [
            f"Utilisateur: Salut, je m'appelle {name}.",
            f"Utilisateur: Je viens de {city}.",
            f"Utilisateur: J'aime le {sport}.",
        ]

        task = random.randint(1, 6)
        if task == 1:
            instruction = "Tu peux me rappeler mon prenom ?"
            response = f"Oui, bien sur. Tu t'appelles {name}."
        elif task == 2:
            instruction = "Et ma ville ?"
            response = f"Tu viens de {city}."
        elif task == 3:
            instruction = "Rappelle mon prenom et ma ville."
            response = f"Tu t'appelles {name} et tu viens de {city}."
        elif task == 4:
            instruction = "Donne-moi un conseil sport pour demain."
            response = (
                f"Demain, fais une seance courte de {sport} (30 minutes), "
                "echauffe-toi 5 minutes et termine par des etirements."
            )
        elif task == 5:
            instruction = "Reponds de maniere courte et polie."
            response = "Avec plaisir. Je peux t'aider en quelques phrases claires."
        else:
            instruction = "Tu peux reformuler ma demande clairement ?"
            response = "Bien sur. Je reformule ta demande de facon simple et concise."

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
    rows = build_rows(args.rows, args.seed)

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out_file": str(out)})


if __name__ == "__main__":
    main()
