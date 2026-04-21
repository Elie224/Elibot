import argparse
import csv
import random
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build broad ML general knowledge dataset")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_ml_culture_generale.csv")
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (_norm(row.get("instruction", "")), _norm(row.get("response", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_rows(target_rows: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)

    catalog = [
        {
            "topic": "difference IA, ML, deep learning",
            "area": "fondamentaux",
            "core": "IA est le champ global, ML est l'apprentissage a partir de donnees, deep learning est un sous-ensemble ML base sur reseaux profonds.",
            "formula": "DL subset ML subset IA",
            "code": "from sklearn.linear_model import LogisticRegression\nfrom tensorflow import keras",
            "pitfalls": "melanger capacite de modele et qualite des donnees",
        },
        {
            "topic": "bias-variance tradeoff",
            "area": "fondamentaux",
            "core": "Biais eleve = modele trop simple, variance elevee = modele instable selon echantillons.",
            "formula": "Erreur totale = biais^2 + variance + bruit irreductible",
            "code": "from sklearn.model_selection import learning_curve",
            "pitfalls": "optimiser uniquement le score train",
        },
        {
            "topic": "data leakage",
            "area": "qualite donnees",
            "core": "Une information de la cible ou du futur fuit dans les features et gonfle artificiellement les performances.",
            "formula": "score_offline >> score_online => suspicion fuite",
            "code": "from sklearn.pipeline import Pipeline  # eviter fit global avant split",
            "pitfalls": "fit du scaler/imputer avant split",
        },
        {
            "topic": "metriques classification",
            "area": "evaluation",
            "core": "Precision, recall, F1, ROC-AUC et PR-AUC servent des objectifs differents selon cout metier.",
            "formula": "F1 = 2*(P*R)/(P+R)",
            "code": "from sklearn.metrics import classification_report, roc_auc_score",
            "pitfalls": "utiliser accuracy seule en classes desequilibrees",
        },
        {
            "topic": "calibration des probabilites",
            "area": "evaluation",
            "core": "Une proba de 0.8 doit correspondre a environ 80% de vrais positifs; sinon il faut calibrer.",
            "formula": "Brier score = moyenne((p-y)^2)",
            "code": "from sklearn.calibration import CalibratedClassifierCV",
            "pitfalls": "appliquer un seuil fixe sans calibration",
        },
        {
            "topic": "xgboost vs random forest",
            "area": "algorithmes",
            "core": "RF est robuste et simple, XGBoost performe souvent mieux mais exige un tuning prudent.",
            "formula": "Boosting: F_t(x)=F_{t-1}(x)+eta*h_t(x)",
            "code": "from xgboost import XGBClassifier\nfrom sklearn.ensemble import RandomForestClassifier",
            "pitfalls": "sur-tuning et overfitting local",
        },
        {
            "topic": "feature engineering tabulaire",
            "area": "donnees",
            "core": "Creer des variables stables, informatives et compatibles avec l'inference online.",
            "formula": "signal utile > bruit ajoute",
            "code": "df['ratio_util'] = df['usage']/df['quota']",
            "pitfalls": "features indisponibles en production",
        },
        {
            "topic": "validation temporelle",
            "area": "evaluation",
            "core": "Pour series temporelles, le split doit respecter l'ordre temporel pour simuler la production.",
            "formula": "train < validation < test (timeline)",
            "code": "from sklearn.model_selection import TimeSeriesSplit",
            "pitfalls": "shuffle aleatoire sur donnees temporelles",
        },
        {
            "topic": "drift de donnees et concept drift",
            "area": "mlops",
            "core": "Data drift: distribution des features change; concept drift: relation feature->cible change.",
            "formula": "PSI/KL pour features + chute KPI pour concept",
            "code": "# monitor PSI + performance par segment",
            "pitfalls": "ne monitorer que la metrique globale",
        },
        {
            "topic": "MLOps cycle de vie",
            "area": "mlops",
            "core": "Versionner donnees/modeles, CI/CD, deployment progressif, monitoring, rollback.",
            "formula": "data+code+model+config => artefact reproductible",
            "code": "# train -> eval -> register -> deploy canary -> monitor",
            "pitfalls": "pas de rollback ni seuil d'alerte",
        },
        {
            "topic": "NLP embeddings et retrieval",
            "area": "nlp",
            "core": "Transformer le texte en vecteurs semantiques, puis retrouver les passages pertinents.",
            "formula": "similarite cosine(q, doc)",
            "code": "from sentence_transformers import SentenceTransformer",
            "pitfalls": "index non mis a jour et chunks mal decoupes",
        },
        {
            "topic": "computer vision detection objet",
            "area": "vision",
            "core": "Detection = localisation + classification; metrics cles: mAP, IoU, latence.",
            "formula": "IoU = intersection/union",
            "code": "# entrainer YOLO puis evaluer mAP@0.5:0.95",
            "pitfalls": "dataset annote de maniere incoherente",
        },
        {
            "topic": "reinforcement learning",
            "area": "rl",
            "core": "Un agent apprend une politique qui maximise la recompense cumulative.",
            "formula": "Q(s,a)=Q(s,a)+alpha*(r+gamma*max Q(s',a')-Q(s,a))",
            "code": "# boucle interaction environnement + update",
            "pitfalls": "reward mal defini et exploration insuffisante",
        },
        {
            "topic": "ethique, biais et fairness",
            "area": "gouvernance",
            "core": "Verifier disparites de performance selon groupes sensibles et documenter les limites.",
            "formula": "gap_tpr = TPR_g1 - TPR_g2",
            "code": "# evaluer metriques par sous-population",
            "pitfalls": "monitoring absent apres deploiement",
        },
        {
            "topic": "debugging de modeles en production",
            "area": "operations",
            "core": "Diagnostiquer via tranche temporelle, segment utilisateur, version modele, version features.",
            "formula": "incident = divergence KPI x segment",
            "code": "# comparez distribution features train vs prod",
            "pitfalls": "corriger sans reproduction minimale",
        },
    ]

    question_styles = [
        "Explique {topic} comme un expert pedago.",
        "Donne une fiche complete sur {topic} avec formule et exemple.",
        "Comment bien comprendre {topic} pour un projet reel ?",
        "Je veux une reponse solide sur {topic} en contexte entreprise.",
        "Quels sont les pieges et bonnes pratiques pour {topic} ?",
    ]
    personas = [
        "debutant motivé",
        "data analyst",
        "data scientist",
        "ml engineer",
        "chef de produit data",
    ]
    contexts = [
        "fraude bancaire",
        "churn telecom",
        "scoring credit",
        "prevision de ventes retail",
        "maintenance predictive",
        "support client automatise",
    ]
    answer_modes = [
        "cours structure",
        "checklist actionnable",
        "focus erreurs frequentes",
        "focus production",
    ]
    constraints = [
        "latence stricte",
        "budget cloud limite",
        "classes tres desequilibrees",
        "forte saisonnalite",
        "labels imparfaits",
        "explicabilite obligatoire",
        "conformite RGPD",
        "donnees manquantes",
        "derive rapide",
        "fort volume temps reel",
    ]
    diagnostics = [
        "analyser matrice de confusion par segment",
        "controler la calibration et le Brier score",
        "comparer offline vs online sur meme fenetre",
        "verifier la stabilite des features importantes",
        "auditer le pipeline train/inference",
        "isoler les faux positifs couteux",
        "isoler les faux negatifs critiques",
        "mesurer derive de distribution",
        "suivre latence p95 et cout par requete",
        "tester rollback et reprise",
    ]
    tools = [
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "pytorch",
        "tensorflow",
        "mlflow",
        "prefect",
        "airflow",
        "fastapi",
        "evidently",
    ]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < max(1, target_rows):
        card = random.choice(catalog)
        style = random.choice(question_styles)
        persona = random.choice(personas)
        context = random.choice(contexts)
        mode = random.choice(answer_modes)
        constraint = random.choice(constraints)
        diagnostic = random.choice(diagnostics)
        tool = random.choice(tools)
        instruction = (
            f"{style.format(topic=card['topic'])} "
            f"Public: {persona}. Contexte: {context}. Contrainte: {constraint}."
        )

        response = (
            f"Theme: {card['topic']} ({card['area']})\n"
            f"Definition: {card['core']}\n"
            f"Repere mathematique: {card['formula']}\n"
            "Exemple concret:\n"
            f"- Cas: {context}.\n"
            "- Decision metier: choisir metrique/seuil selon cout des erreurs.\n"
            "Code ou pseudo-code:\n"
            "```python\n"
            f"{card['code']}\n"
            "```\n"
            f"Erreur frequente: {card['pitfalls']}\n"
            f"Diagnostic recommande: {diagnostic}.\n"
            f"Stack suggeree: {tool}.\n"
            "Bonnes pratiques immediates:\n"
            "1) Verifier qualite des donnees et split.\n"
            "2) Mesurer metriques globales + par segment.\n"
            "3) Documenter hypothese, limite et plan de monitoring.\n"
            f"Mode de reponse: {mode}."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux maximiser ma culture generale ML. ||| Assistant: je reponds avec explication complete, formule, exemple et pratique.",
                "source": "ml_culture_generale",
            }
        )
        i += 1

    return _dedupe(rows)[:target_rows]


def main() -> None:
    args = parse_args()
    rows = build_rows(args.rows, args.seed)
    out_file = Path(args.out_file)
    _write_rows(out_file, rows)
    print({"rows": len(rows), "out_file": str(out_file).replace('\\\\', '/')})


if __name__ == "__main__":
    main()
