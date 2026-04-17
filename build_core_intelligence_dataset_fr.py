import argparse
import csv
import random
from pathlib import Path

TOPICS = [
    "pipeline machine learning",
    "validation de donnees",
    "optimisation pandas",
    "debug Python",
    "requetes SQL analytiques",
    "evaluation de modele",
    "classification d'intentions",
    "planification de taches techniques",
    "automatisation ETL",
    "monitoring ML",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core intelligence dataset for Elibot")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_core_intelligence.csv")
    return parser.parse_args()


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n):
        topic = random.choice(TOPICS)
        mode = random.choice(["plan", "debug", "code", "intent"])

        if mode == "plan":
            instruction = f"Explique une methode professionnelle pour {topic}."
            response = (
                "1) Clarifier objectif et contraintes. "
                "2) Definir les entrees/sorties et criteres de qualite. "
                "3) Prototyper rapidement avec tests minimaux. "
                "4) Instrumenter logs/metriques et traiter les erreurs. "
                "5) Evaluer, documenter, puis iterer."
            )
        elif mode == "debug":
            instruction = f"J'ai une erreur sur {topic}. Quelle procedure de debug suivre ?"
            response = (
                "1) Reproduire le bug avec un cas minimal. "
                "2) Verifier schema, types et versions. "
                "3) Ajouter logs autour de l'etape fautive. "
                "4) Ecrire un test qui echoue puis corriger. "
                "5) Revalider sur plusieurs echantillons."
            )
        elif mode == "code":
            instruction = f"Donne un exemple Python court pour {topic}."
            response = (
                "```python\n"
                "def run_pipeline(rows):\n"
                "    cleaned = [r for r in rows if r is not None]\n"
                "    return {\"count\": len(cleaned), \"ok\": len(cleaned) > 0}\n"
                "```"
            )
        else:
            instruction = f"Classe cette demande dans une intention pour {topic}."
            response = (
                "Intent: technical_execution\n"
                "Reasoning: la demande cible une action technique explicite, "
                "avec besoin de procedure reproductible."
            )

        history = " ||| ".join([
            "Utilisateur: Je veux une solution robuste.",
            "Assistant: Je fournis une reponse technique structuree.",
        ])

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": "core_intelligence",
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
