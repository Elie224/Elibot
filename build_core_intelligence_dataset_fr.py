import argparse
import csv
import random
import re
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
    "feature engineering tabulaire",
    "gestion de drift de donnees",
    "versioning de modeles",
    "tests unitaires Python",
    "conception d'API FastAPI",
    "orchestration Airflow",
    "vector search et retrieval",
    "optimisation de prompt systeme",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core intelligence dataset for Elibot")
    parser.add_argument("--rows", type=int, default=297000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_core_intelligence.csv")
    return parser.parse_args()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for case_id in range(1, n + 1):
        topic = random.choice(TOPICS)
        mode = random.choice(["plan", "debug", "code", "intent", "root_cause", "tradeoff"])
        persona = random.choice(["mlops", "data", "backend", "analytics"])
        urgency = random.choice(["normal", "urgent", "postmortem"])

        if mode == "plan":
            instruction = f"Cas {case_id} [{persona}|{urgency}] Explique une methode professionnelle pour {topic}."
            response = (
                "1) Clarifier objectif et contraintes. "
                "2) Definir les entrees/sorties et criteres de qualite. "
                "3) Prototyper rapidement avec tests minimaux. "
                "4) Instrumenter logs/metriques et traiter les erreurs. "
                "5) Evaluer, documenter, puis iterer."
            )
        elif mode == "debug":
            instruction = f"Cas {case_id} [{persona}|{urgency}] J'ai une erreur sur {topic}. Quelle procedure de debug suivre ?"
            response = (
                "1) Reproduire le bug avec un cas minimal. "
                "2) Verifier schema, types et versions. "
                "3) Ajouter logs autour de l'etape fautive. "
                "4) Ecrire un test qui echoue puis corriger. "
                "5) Revalider sur plusieurs echantillons."
            )
        elif mode == "code":
            instruction = f"Cas {case_id} [{persona}|{urgency}] Donne un exemple Python court pour {topic}."
            response = (
                "```python\n"
                "def run_pipeline(rows):\n"
                "    cleaned = [r for r in rows if r is not None]\n"
                "    return {\"count\": len(cleaned), \"ok\": len(cleaned) > 0}\n"
                "```"
            )
        elif mode == "root_cause":
            instruction = f"Cas {case_id} [{persona}|{urgency}] Fais une analyse de cause racine pour un incident sur {topic}."
            response = (
                "Hypothese 1: donnees d'entree incoherentes. "
                "Hypothese 2: rupture de schema entre environnements. "
                "Hypothese 3: regression applicative recente. "
                "Plan: verifier logs, comparer versions, executer tests de non-regression, puis corriger avec rollback securise."
            )
        elif mode == "tradeoff":
            instruction = f"Cas {case_id} [{persona}|{urgency}] Compare deux approches pour {topic} avec avantages et risques."
            response = (
                "Approche A: implementation rapide, faible cout initial, mais risque technique plus eleve. "
                "Approche B: architecture robuste, meilleure maintenabilite, mais delai plus long. "
                "Decision: choisir selon criticite, budget et exigence de fiabilite."
            )
        else:
            instruction = f"Cas {case_id} [{persona}|{urgency}] Classe cette demande dans une intention pour {topic}."
            response = (
                "Intent: technical_execution\n"
                "Reasoning: la demande cible une action technique explicite, "
                "avec besoin de procedure reproductible."
            )

        if len(instruction) < 20 or len(response) < 30:
            continue

        key = (_norm(instruction), _norm(response))
        if key in seen:
            continue
        seen.add(key)

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
