import argparse
import csv
import random
import re
from pathlib import Path

PROFILES = [
    ("Karim", "Lyon", "MLOps"),
    ("Lea", "Paris", "NLP"),
    ("Nadia", "Lille", "Data Engineering"),
    ("Yanis", "Toulouse", "Computer Vision"),
    ("Sofia", "Marseille", "Analytics Engineering"),
    ("Amine", "Nantes", "LLM Ops"),
    ("Ines", "Bordeaux", "Platform Engineering"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory/context/summary dataset for Elibot")
    parser.add_argument("--rows", type=int, default=99000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_memory_context_summary.csv")
    return parser.parse_args()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    projects = ["chatbot enterprise", "pipeline ml", "assistant ops", "reporting data"]

    for case_id in range(1, n + 1):
        name, city, domain = random.choice(PROFILES)
        task = random.choice(["resume", "compress", "retrieve", "handoff"])
        project = random.choice(projects)

        history = " ||| ".join([
            f"Utilisateur: Je m'appelle {name} et je travaille en {domain}.",
            f"Utilisateur: Je suis base a {city}.",
            f"Utilisateur: Mon objectif est d'automatiser {project}.",
            "Assistant: Je conserve ces informations pour les prochaines etapes.",
        ])

        if task == "resume":
            instruction = f"Cas {case_id}: Fais un resume operationnel de la conversation."
            response = (
                f"Resume: utilisateur {name}, base a {city}, domaine {domain}, "
                f"objectif principal: automatiser {project}. "
                "Actions suivantes: definir etapes, contraintes et indicateurs."
            )
        elif task == "compress":
            instruction = f"Cas {case_id}: Compresse le contexte en 3 points cles."
            response = (
                f"1) Profil: {name}, {domain}. "
                f"2) Contexte: localise a {city}. "
                "3) But: automatisation pipeline avec suivi metriques."
            )
        elif task == "retrieve":
            instruction = f"Cas {case_id}: Recupere les informations utiles pour la prochaine action."
            response = (
                f"Memoire utile: nom={name}; ville={city}; domaine={domain}; "
                f"objectif=automatiser {project}; prochaine_action=proposer workflow JSON."
            )
        else:
            instruction = f"Cas {case_id}: Prepare un handoff clair pour un autre agent."
            response = (
                "Handoff: contexte utilisateur etabli, objectifs confirms, "
                "contraintes implicites techniques. "
                "Prochaine etape: plan + action outillee + validation resultats."
            )

        if len(instruction) < 15 or len(response) < 30:
            continue

        key = (_norm(instruction), _norm(response))
        if key in seen:
            continue
        seen.add(key)

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
