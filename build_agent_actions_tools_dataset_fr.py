import argparse
import csv
import random
from pathlib import Path

ACTIONS = [
    "send_email",
    "run_python",
    "query_sql",
    "call_api",
    "create_task",
    "update_task",
    "schedule_job",
    "notify_slack",
]


SCENARIOS = [
    ("Envoie un email de suivi au client", "send_email", {"to": "client@example.com", "subject": "Suivi projet", "body": "Bonjour, voici la mise a jour du projet."}),
    ("Analyse ce dataset CSV et donne les colonnes manquantes", "run_python", {"code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.isna().sum())"}),
    ("Donne le top 10 des ventes par ville", "query_sql", {"query": "SELECT city, SUM(amount) AS total FROM sales GROUP BY city ORDER BY total DESC LIMIT 10;"}),
    ("Appelle l'API GitHub pour lister mes repos", "call_api", {"method": "GET", "url": "https://api.github.com/user/repos"}),
    ("Cree une tache de deploiement", "create_task", {"title": "Deploiement API", "priority": "high", "steps": ["build", "tests", "deploy"]}),
    ("Passe la tache en termine", "update_task", {"task_id": "task-001", "status": "done"}),
    ("Planifie un job hebdomadaire dimanche 03:00", "schedule_job", {"cron": "0 3 * * SUN", "job": "weekly_train_runner"}),
    ("Alerte Slack quand le train est termine", "notify_slack", {"channel": "#ml-ops", "message": "Entrainement termine avec succes."}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build agent actions/tools dataset for Elibot")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_agent_actions_tools.csv")
    return parser.parse_args()


def _format_json(action: str, payload: dict) -> str:
    parts = [f'  "action": "{action}"']
    for key, value in payload.items():
        if isinstance(value, str):
            parts.append(f'  "{key}": "{value}"')
        elif isinstance(value, list):
            list_items = ", ".join(f'"{x}"' for x in value)
            parts.append(f'  "{key}": [{list_items}]')
        else:
            parts.append(f'  "{key}": {value}')
    return "{\n" + ",\n".join(parts) + "\n}"


def build_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []

    for _ in range(n):
        user_task, action, payload = random.choice(SCENARIOS)
        intent = random.choice([
            "choisir l'outil adapte",
            "generer un JSON d'action",
            "planifier un workflow",
            "executer une action de maniere sure",
        ])

        instruction = f"Utilisateur: {user_task}. Donne une reponse orientee agent et actionnable."
        response = _format_json(action, payload)
        history = " ||| ".join([
            "Utilisateur: Tu es un agent IA technique.",
            "Assistant: Je dois repondre avec une action structuree et sure.",
            f"Utilisateur: Objectif: {intent}.",
        ])

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": "agent_actions_tools",
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
