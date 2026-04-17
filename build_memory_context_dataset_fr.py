import argparse
import csv
import random
from pathlib import Path

PROFILES = [
    ("Karim", "Lyon", "MLOps"),
    ("Lea", "Paris", "NLP"),
    ("Nadia", "Lille", "Data Engineering"),
    ("Yanis", "Toulouse", "Computer Vision"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory/context/summary dataset for Elibot")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_memory_context_summary.csv")
    return parser.parse_args()


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n):
        name, city, domain = random.choice(PROFILES)
        task = random.choice(["resume", "compress", "retrieve", "handoff"])

        history = " ||| ".join([
            f"Utilisateur: Je m'appelle {name} et je travaille en {domain}.",
            f"Utilisateur: Je suis base a {city}.",
            "Utilisateur: Mon objectif est d'automatiser mon pipeline.",
            "Assistant: Je conserve ces informations pour les prochaines etapes.",
        ])

        if task == "resume":
            instruction = "Fais un resume operationnel de la conversation."
            response = (
                f"Resume: utilisateur {name}, base a {city}, domaine {domain}, "
                "objectif principal: automatiser un pipeline. "
                "Actions suivantes: definir etapes, contraintes et indicateurs."
            )
        elif task == "compress":
            instruction = "Compresse le contexte en 3 points cles."
            response = (
                f"1) Profil: {name}, {domain}. "
                f"2) Contexte: localise a {city}. "
                "3) But: automatisation pipeline avec suivi metriques."
            )
        elif task == "retrieve":
            instruction = "Recupere les informations utiles pour la prochaine action."
            response = (
                f"Memoire utile: nom={name}; ville={city}; domaine={domain}; "
                "objectif=automatiser pipeline; prochaine_action=proposer workflow JSON."
            )
        else:
            instruction = "Prepare un handoff clair pour un autre agent."
            response = (
                "Handoff: contexte utilisateur etabli, objectifs confirms, "
                "contraintes implicites techniques. "
                "Prochaine etape: plan + action outillee + validation resultats."
            )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": "memory_context_summary",
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args.rows, args.seed)

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out_file": str(out)})


if __name__ == "__main__":
    main()
