import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-form expert signature booster dataset")
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_signature_long_expert.csv")
    return parser.parse_args()


def build_rows(n: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)

    topics = [
        "pipeline ML de bout en bout",
        "optimisation pandas sur gros volumes",
        "architecture FastAPI pour inference",
        "monitoring drift et qualite des predictions",
        "strategie de feature engineering tabulaire",
        "gestion des erreurs et rollback en production",
        "calibration et choix de seuil metier",
        "debugging data leakage",
    ]
    sectors = ["retail", "banque", "telecom", "industrie", "SaaS"]
    constraints = ["latence", "cout", "fiabilite", "explicabilite", "conformite"]

    rows: list[dict[str, str]] = []
    for i in range(max(1, n)):
        topic = topics[i % len(topics)]
        sector = sectors[i % len(sectors)]
        constraint = constraints[i % len(constraints)]

        instruction = (
            f"Donne une reponse expert, longue et structuree sur {topic}. "
            f"Contexte: {sector}. Contrainte prioritaire: {constraint}."
        )
        response = (
            f"Voici un cadre expert pour {topic} en contexte {sector}.\n"
            "1) Cadrage metier\n"
            "- Definir la decision a ameliorer, les risques et le ROI attendu.\n"
            "- Etablir une metrique principale et des garde-fous (qualite, securite, derive).\n"
            "2) Donnees et pipeline\n"
            "- Contrat de donnees: schema, fraicheur, nulls, deduplication, versioning.\n"
            "- Pipeline reproductible: extraction, validation, features, split temporel, tracking.\n"
            "3) Modelisation\n"
            "- Baseline simple puis modele plus performant sous contrainte reelle.\n"
            "- Calibration, seuils par segment et analyse d'erreurs par classe critique.\n"
            "4) Production\n"
            "- Packaging, tests, CI/CD, canary/champion-challenger, rollback instantane.\n"
            "- Monitoring online: performance, drift, latence, cout, incidents.\n"
            "5) Gouvernance\n"
            "- Journal d'experiences, audit des changements, revue hebdo des KPIs.\n"
            "- Plan d'amelioration continue: corriger les biais, enrichir features, retuner si necessaire.\n"
            f"Remarque pratique: prioriser {constraint} des la conception pour eviter les regressions tardives."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux une explication detaillee et exploitable. ||| Assistant: reponse expert structuree.",
                "source": "signature_long_expert",
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

    print({"rows": len(rows), "out_file": str(out).replace('\\\\', '/')})


if __name__ == "__main__":
    main()
