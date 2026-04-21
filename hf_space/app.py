import gradio as gr
import json
import os
import re
import tempfile
import torch
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DOMAIN_TOPICS = [
    "analyse de donnees",
    "machine learning",
    "ia appliquee",
    "automatisation",
    "pipelines",
    "api",
]

IN_DOMAIN_KEYWORDS = {
    "data", "donnee", "donnees", "dataset", "csv", "json", "excel", "table", "sql",
    "pandas", "numpy", "analyse", "nettoyage", "feature", "visualisation", "statistique",
    "ml", "ia", "ai", "machine learning", "modele", "model", "entrainement", "evaluation",
    "classification", "regression", "prediction", "prompt", "llm", "token", "embedding",
    "pipeline", "workflow", "automatisation", "script", "python", "fastapi", "api", "docker",
    "overfitting", "surapprentissage", "underfitting", "sous apprentissage", "biais variance",
    "precision", "recall", "f1", "matrice de confusion", "cross validation", "validation croisee",
    "k fold", "feature engineering", "data leakage", "fuite de donnees", "regularisation",
    "learning rate", "gradient descent", "desequilibre", "class imbalance", "one hot",
    "normalisation", "standardisation", "feature selection", "hyperparametre", "tuning",
    "baseline", "mae", "rmse", "r2", "roc", "auc", "pr auc",
    "data drift", "concept drift", "drift", "derive des donnees", "derive de concept",
    "seuil de decision", "threshold", "xgboost", "random forest",
}

OUT_DOMAIN_KEYWORDS = {
    "medecine", "medical", "maladie", "diagnostic", "traitement", "politique", "election",
    "religion", "psychologie", "depression", "amour", "relation", "sexe", "voyance", "astrologie",
    "finance personnelle", "pari", "bet", "casino", "juridique", "avocat",
}

MODEL_ID = "Elie224/Elibot"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 192
TEMPERATURE = 0.0
TOP_P = 0.9
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 3
HISTORY_TURNS = 6
HISTORY_MODE = "full"
SYSTEM_PROMPT = (
    "Tu es Elibot, un assistant specialise en analyse de donnees, IA appliquee et automatisation. "
    "Tu tiens une conversation technique fluide, avec un style humain, naturel et professionnel. "
    "Tu peux donner des definitions, des remarques d'expert, des details concrets et des exemples pratiques. "
    "Tu evites les listes seches sans contexte: tu expliques le pourquoi, le comment, et les compromis. "
    "Tu privilegies des reponses actionnables et bien structurees (objectif, execution, risques, prochaines actions). "
    "Tu refuses poliment les sujets hors domaine et rediriges vers une demande technique."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHAT_LOG_PATH = Path(os.getenv("CHAT_LOG_PATH", "/tmp/elibot_chat_events.jsonl"))
AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "/tmp/elibot_audit.jsonl"))
PLOT_TMP_DIR = Path(tempfile.gettempdir()) / "elibot_plots"

RESPONSE_MODES = ["Court", "Expert"]
QUESTION_MARKERS = {
    "cest quoi", "ca veut dire quoi", "definition", "definir", "explique", "expliquer",
    "difference", "pourquoi", "comment", "quand", "a quoi sert", "role",
}

FOLLOWUP_MARKERS = {
    "plus de details", "plus de detail", "detaille", "detailler", "approfondis", "approfondir",
    "des remarques", "remarques", "ton avis", "en pratique", "donne un exemple", "explique mieux",
    "ok et", "et du coup", "et ensuite", "pour aller plus loin",
}

AFFIRMATION_MARKERS = {
    "oui", "ouais", "yes", "ok", "okay", "daccord", "d accord", "vas y", "go", "parfait"
}

AFFIRMATION_TOKENS = {
    "oui", "ouais", "yes", "ok", "okay", "daccord", "d", "accord", "vas", "y", "go",
    "parfait", "fait", "fais", "le", "continuer", "continue",
}

AFFIRMATION_CONNECTORS = {"stp", "svp", "please", "alors", "du", "coup", "maintenant"}

CONCEPT_CARDS = [
    {
        "keys": ["overfitting", "surapprentissage"],
        "title": "Overfitting",
        "definition": "Le modele memorise le train et generalise mal sur de nouvelles donnees.",
        "example": "Accuracy train tres haute mais performance validation/test qui chute.",
        "best": "Cross-validation, regularisation, plus de donnees, simplifier le modele, early stopping.",
        "pitfalls": "Tuner longtemps sur un seul jeu de validation.",
    },
    {
        "keys": ["underfitting", "sous apprentissage", "sousapprentissage"],
        "title": "Underfitting",
        "definition": "Le modele est trop simple et n'apprend pas assez les patterns utiles.",
        "example": "Scores faibles a la fois sur train et test.",
        "best": "Modele plus expressif, meilleures features, plus d'epoque ou meilleur tuning.",
        "pitfalls": "Conclure trop vite que les donnees sont mauvaises sans tester une baseline correcte.",
    },
    {
        "keys": ["bias variance", "biais variance"],
        "title": "Compromis Biais-Variance",
        "definition": "Biais eleve: modele trop rigide. Variance elevee: modele trop sensible au bruit.",
        "example": "Regression lineaire simple vs arbre tres profond.",
        "best": "Ajuster complexite, regularisation et volume de donnees.",
        "pitfalls": "Optimiser uniquement le train sans verifier la variance.",
    },
    {
        "keys": ["precision recall", "precision rappel"],
        "title": "Precision vs Recall",
        "definition": "Precision mesure la fiabilite des positifs predits; recall mesure les positifs retrouves.",
        "example": "Fraude: recall haut pour rater moins de fraudes.",
        "best": "Choisir selon le cout metier des faux positifs/faux negatifs.",
        "pitfalls": "Comparer des modeles sans fixer le seuil de decision.",
    },
    {
        "keys": ["f1", "f1 score"],
        "title": "F1-score",
        "definition": "Moyenne harmonique precision/recall, utile sur classes desequilibrees.",
        "example": "Un modele avec precision 0.9 et recall 0.5 aura un F1 moyen.",
        "best": "Utiliser F1 quand precision et recall sont toutes deux importantes.",
        "pitfalls": "Utiliser uniquement l'accuracy sur un dataset tres desequilibre.",
    },
    {
        "keys": ["matrice de confusion", "confusion matrix"],
        "title": "Matrice de confusion",
        "definition": "Tableau des vrais/faux positifs/negatifs pour analyser les erreurs de classification.",
        "example": "Permet d'identifier quelle classe est le plus confondue.",
        "best": "Analyser par classe puis ajuster seuil/features/donnees.",
        "pitfalls": "Ne pas regarder les distributions de classes en meme temps.",
    },
    {
        "keys": ["cross validation", "validation croisee", "k fold", "kfold"],
        "title": "Cross-validation",
        "definition": "Evaluation robuste en entrainant/testant sur plusieurs decoupages du dataset.",
        "example": "K-fold CV pour stabiliser l'estimation de performance.",
        "best": "Utiliser stratification en classification desequilibree.",
        "pitfalls": "Fuite de donnees si preprocessing fait avant split.",
    },
    {
        "keys": ["train val test", "train validation test", "split"],
        "title": "Split Train/Validation/Test",
        "definition": "Train pour apprendre, validation pour tuner, test pour eval finale.",
        "example": "70/15/15 ou 80/10/10 selon taille et contexte.",
        "best": "Fixer le test une fois pour toutes et ne pas le reutiliser pour tuner.",
        "pitfalls": "Utiliser le test pendant l'iteratif de tuning.",
    },
    {
        "keys": ["feature engineering", "features"],
        "title": "Feature Engineering",
        "definition": "Creation/transformation de variables pour rendre le signal plus apprenable.",
        "example": "Date -> jour, mois, saison; texte -> TF-IDF/embeddings.",
        "best": "Features interpretable + validation metier + tests de fuite.",
        "pitfalls": "Encoder des informations indisponibles au moment de prediction.",
    },
    {
        "keys": ["data leakage", "fuite de donnees", "leakage"],
        "title": "Data Leakage",
        "definition": "Information future ou cible injectee dans les features durant l'entrainement.",
        "example": "Normaliser sur tout le dataset avant split.",
        "best": "Construire pipeline fit uniquement sur train.",
        "pitfalls": "Resultats excellents offline puis echec en production.",
    },
    {
        "keys": ["regularisation", "regularization", "l1", "l2", "dropout"],
        "title": "Regularisation",
        "definition": "Techniques pour reduire l'overfitting en contraignant le modele.",
        "example": "L2 en regression, dropout en reseaux neuronaux.",
        "best": "Tuner la force de regularisation avec validation.",
        "pitfalls": "Sur-regulariser et perdre le signal utile.",
    },
    {
        "keys": ["learning rate", "taux dapprentissage", "lr"],
        "title": "Learning Rate",
        "definition": "Pas de mise a jour des poids pendant l'optimisation.",
        "example": "Trop haut: divergence; trop bas: entrainement lent.",
        "best": "Warmup + scheduler et suivi de loss.",
        "pitfalls": "Changer plusieurs hyperparametres sans isoler l'effet du LR.",
    },
    {
        "keys": ["gradient descent", "descente de gradient"],
        "title": "Descente de gradient",
        "definition": "Algorithme d'optimisation qui ajuste les poids pour minimiser la perte.",
        "example": "SGD/Adam en mini-batch sur chaque iteration.",
        "best": "Surveiller gradients, loss et stabilite numerique.",
        "pitfalls": "Ignorer le gradient exploding/vanishing.",
    },
    {
        "keys": ["desequilibre", "imbalance", "class imbalance"],
        "title": "Desequilibre de classes",
        "definition": "Une classe est tres majoritaire, ce qui biaise la prediction.",
        "example": "Fraude: 1% positifs, 99% negatifs.",
        "best": "Class weights, re-sampling, metriques adaptees (F1, recall, PR-AUC).",
        "pitfalls": "Se fier a une accuracy elevee mais trompeuse.",
    },
    {
        "keys": ["one hot", "onehot", "encodage categoriel", "categorical encoding", "categorical encoding options"],
        "title": "One-hot Encoding",
        "definition": "Transformation des categories en colonnes binaires.",
        "example": "Ville={Paris,Lyon} -> deux colonnes 0/1.",
        "best": "Options: one-hot, target encoding, hashing. Gerer la cardinalite et prevenir la fuite.",
        "pitfalls": "Explosion dimensionnelle sur forte cardinalite ou target encoding mal valide (fuite).",
    },
    {
        "keys": ["normalisation", "normalization", "standardisation", "standardization", "scaling", "normalisation vs standardisation"],
        "title": "Scaling des features",
        "definition": "Mise a l'echelle des variables numeriques pour stabiliser l'apprentissage.",
        "example": "StandardScaler (moyenne 0, ecart-type 1).",
        "best": "Normalisation (min-max) ou standardisation (z-score) selon le modele, avec fit sur train uniquement.",
        "pitfalls": "Scalage global avant split (fuite de donnees).",
    },
    {
        "keys": ["feature selection", "selection de variables"],
        "title": "Feature Selection",
        "definition": "Choisir les variables les plus utiles pour performance et robustesse.",
        "example": "Filtrage par importance modele ou tests statistiques.",
        "best": "Combiner approche metier + validation empirique.",
        "pitfalls": "Retirer des variables utiles a la generalisation hors echantillon.",
    },
    {
        "keys": ["hyperparameter", "hyperparametre", "tuning"],
        "title": "Hyperparameter Tuning",
        "definition": "Recherche des meilleurs reglages d'un modele (profondeur, LR, regularisation...).",
        "example": "Grid Search, Random Search, Bayesian Optimization.",
        "best": "Limiter l'espace de recherche et utiliser CV pour robustesse.",
        "pitfalls": "Overfit sur validation par essais excessifs.",
    },
    {
        "keys": ["baseline"],
        "title": "Modele Baseline",
        "definition": "Reference simple pour mesurer si un modele complexe apporte un vrai gain.",
        "example": "Regle majoritaire ou regression lineaire comme point de depart.",
        "best": "Toujours comparer les nouvelles versions a la baseline.",
        "pitfalls": "Complexifier sans demonstrer d'amelioration mesurable.",
    },
    {
        "keys": ["mae", "rmse", "r2", "metrique regression", "metrics regression"],
        "title": "Metriques de regression",
        "definition": "MAE: erreur moyenne absolue, RMSE: penalise plus les grosses erreurs, R2: variance expliquee.",
        "example": "RMSE utile si les grosses erreurs sont critiques.",
        "best": "Choisir la metrique selon impact metier et unite interpretable.",
        "pitfalls": "Comparer des metriques non alignes avec l'objectif produit.",
    },
    {
        "keys": ["roc", "auc", "pr auc"],
        "title": "ROC-AUC / PR-AUC",
        "definition": "Scores de qualite de classement sur plusieurs seuils de decision.",
        "example": "PR-AUC souvent plus informative en fort desequilibre de classes.",
        "best": "Analyser AUC + courbes + seuil operationnel.",
        "pitfalls": "Utiliser seulement AUC sans fixer un seuil exploitable.",
    },
    {
        "keys": ["data drift", "derive des donnees", "drift des donnees"],
        "title": "Data Drift",
        "definition": "La distribution des features (variables d'entree) change entre train et production.",
        "example": "Le profil des clients en production n'a plus la meme repartition qu'au moment de l'entrainement.",
        "best": "Faire de la surveillance continue (PSI/KS), fixer des alertes, puis lancer un reentrainement si necessaire.",
        "pitfalls": "Attendre une baisse forte de performance avant d'agir.",
    },
    {
        "keys": ["concept drift", "derive de concept"],
        "title": "Concept Drift",
        "definition": "La relation entre features et cible evolue, meme si les entrees semblent stables.",
        "example": "Une baisse de performance sur une fenetre temporelle recente indique un changement du concept appris.",
        "best": "Surveiller la performance par fenetre temporelle avec alerte automatique puis retraining/versioning.",
        "pitfalls": "Confondre concept drift et simple bruit statistique court terme.",
    },
    {
        "keys": ["seuil de decision", "choisir un seuil", "threshold"],
        "title": "Choix du seuil de decision",
        "definition": "Le seuil transforme un score en decision finale, avec compromis precision/recall.",
        "example": "Monter le seuil reduit les faux positifs mais augmente les faux negatifs.",
        "best": "Optimiser le seuil selon cout metier, courbes PR/ROC et contraintes operationnelles.",
        "pitfalls": "Garder le seuil par defaut (0.5) sans validation metier.",
    },
    {
        "keys": ["xgboost vs random forest", "xgboost", "random forest", "quand utiliser random forest"],
        "title": "XGBoost vs Random Forest",
        "definition": "Random Forest bagge des arbres independants; XGBoost booste des arbres sequentiels qui corrigent les erreurs.",
        "example": "XGBoost performe souvent mieux apres tuning, Random Forest est plus robuste en baseline rapide.",
        "best": "Comparer bagging vs boosting sur donnees tabulaire non-lineaire, puis arbitrer performance/latence/interpretabilite.",
        "pitfalls": "Comparer sans meme split, sans calibration du seuil, ou sans controle du temps d'inference.",
    },
    {
        "keys": ["classification logistique", "classification logistique intuition"],
        "title": "Classification logistique",
        "definition": "Modele lineaire qui estime une probabilite via une sigmoide pour faire de la classification.",
        "example": "On transforme le score en classe via un seuil (ex: 0.5).",
        "best": "Verifier calibration des probabilites et regler le seuil selon le cout metier.",
        "pitfalls": "Supposer une frontiere non-lineaire sans feature engineering.",
    },
    {
        "keys": ["regression lineaire", "regression lineaire limites"],
        "title": "Limites de la regression lineaire",
        "definition": "La regression lineaire suppose une relation lineaire et des hypotheses statistiques.",
        "example": "Des outliers forts peuvent tirer les coefficients et degrader la generalisation.",
        "best": "Verifier hypotheses, traiter les outliers et tester regularisation (Ridge/Lasso).",
        "pitfalls": "Appliquer ce modele a une relation clairement non-lineaire sans transformation.",
    },
    {
        "keys": ["versionner un modele", "versionner un modele en production", "model registry", "registry"],
        "title": "Versionner un modele en production",
        "definition": "Utiliser un registry pour tracer version, artefacts et metadonnees d'evaluation.",
        "example": "Chaque version est deployee avec rollback et/ou canary pour limiter le risque.",
        "best": "Associer version du modele, version des features et signature d'API.",
        "pitfalls": "Deployer sans trace complete, ce qui bloque audit et rollback.",
    },
    {
        "keys": ["monitoring d un modele en prod", "monitoring dun modele en prod", "monitoring modele", "mlops monitoring"],
        "title": "Monitoring d'un modele en production",
        "definition": "Suivre en continu drift, latence, taux d'erreur et metriques metier.",
        "example": "Alertes sur drift de donnees, hausse de latence, hausse d'erreur et derive des metriques metier.",
        "best": "Mettre des SLO, alertes et dashboards de metriques par segment.",
        "pitfalls": "Observer uniquement la latence sans surveiller la qualite predictive.",
    },
    {
        "keys": ["quand relancer un retraining", "retraining"],
        "title": "Quand relancer un retraining",
        "definition": "Declencher quand drift ou degradation depasse un seuil defini.",
        "example": "Chute durable de F1 ou hausse d'erreur sur une fenetre temporelle recente.",
        "best": "Automatiser un pipeline de retraining avec validation avant promotion.",
        "pitfalls": "Re-entrainer trop souvent sans verifier la qualite des donnees entrantes.",
    },
    {
        "keys": ["mlops", "mlops c'est quoi", "mlops c'est quoi en pratique"],
        "title": "MLOps en pratique",
        "definition": "Ensemble de pratiques pour industrialiser le cycle de vie ML.",
        "example": "CI/CD, versionning des donnees/modeles, deploiement controle et gouvernance.",
        "best": "Standardiser les pipelines et la reproductibilite des experiments.",
        "pitfalls": "Avoir du ML performant en notebook mais non deployable en production.",
    },
    {
        "keys": ["valeurs manquantes", "comment gerer les valeurs manquantes"],
        "title": "Gestion des valeurs manquantes",
        "definition": "Traiter les NA selon leur mecanisme: suppression, imputation ou modele natif.",
        "example": "Imputation median pour numerique, mode pour categoriel.",
        "best": "Comparer strategies d'imputation et mesurer l'impact metrique.",
        "pitfalls": "Imputer avant split train/validation/test (fuite).",
    },
    {
        "keys": ["health", "ready", "readiness", "api /health et /ready pourquoi"],
        "title": "Endpoints /health et /ready",
        "definition": "/health verifie que le service vit; /ready verifie qu'il est pret a servir.",
        "example": "Un pod peut etre vivant mais non pret tant que le modele n'est pas charge.",
        "best": "Utiliser readiness pour l'orchestration (Kubernetes) et eviter du trafic trop tot.",
        "pitfalls": "Confondre liveness/health et readiness dans le routage.",
    },
]


def load_tokenizer(model_id):
    # Some Spaces/transformers versions fail if extra_special_tokens is a list in tokenizer config.
    try:
        return AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})
    except Exception:
        return AutoTokenizer.from_pretrained(model_id)


TOKENIZER = load_tokenizer(MODEL_ID)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
MODEL.eval()


def clean_generated_text(text):
    value = " ".join(text.strip().split())
    if not value:
        return value

    parts = re.split(r"(?<=[.!?])\s+", value)
    dedup_parts = []
    for part in parts:
        if not part:
            continue
        if dedup_parts and part.lower() == dedup_parts[-1].lower():
            continue
        dedup_parts.append(part)
    value = " ".join(dedup_parts)

    tokens = value.split()
    collapsed = []
    run_token = ""
    run_count = 0
    for tok in tokens:
        key = tok.lower()
        if key == run_token:
            run_count += 1
        else:
            run_token = key
            run_count = 1
        if run_count <= 2:
            collapsed.append(tok)
    value = " ".join(collapsed)

    if value and value[-1] not in ".!?":
        value += "."
    return value


def _normalize_text(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", value.lower())).strip()


def is_short_affirmation(user_text: str) -> bool:
    q_norm = _normalize_text(user_text or "")
    if not q_norm:
        return False
    if q_norm in AFFIRMATION_MARKERS:
        return True
    tokens = q_norm.split()
    if not tokens or len(tokens) > 6:
        return False
    has_affirmation = any(t in AFFIRMATION_TOKENS for t in tokens)
    all_allowed = all((t in AFFIRMATION_TOKENS) or (t in AFFIRMATION_CONNECTORS) for t in tokens)
    return has_affirmation and all_allowed


def is_in_domain_query(user_text):
    text = _normalize_text(user_text or "")
    if not text:
        return True
    if is_short_affirmation(text):
        return True
    if any(marker in text for marker in FOLLOWUP_MARKERS):
        return True
    if text in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc", "merci"}:
        return True
    if any(k in text for k in OUT_DOMAIN_KEYWORDS):
        return False
    return any(k in text for k in IN_DOMAIN_KEYWORDS)


def out_of_domain_reply():
    topics = ", ".join(DOMAIN_TOPICS)
    return (
        "Je suis specialise en "
        f"{topics}. "
        "Je ne traite pas les sujets hors de ce cadre. "
        "Pose une question technique (ex: pipeline ML, API FastAPI, nettoyage de dataset) et je t'aide."
    )


def is_low_quality_answer(answer):
    text = " ".join((answer or "").strip().lower().split())
    if not text:
        return True

    bad_patterns = [
        "je ne peux pas te dire que tu veux dire",
        "je veux dire que je n'ai pas besoin",
        "reheatreheatreheat",
        "synchronoussynchronous",
        "municipal municipal",
    ]
    if any(p in text for p in bad_patterns):
        return True

    # Many duplicated words strongly indicate degeneration.
    tokens = text.split()
    if len(tokens) >= 12:
        uniq_ratio = len(set(tokens)) / max(1, len(tokens))
        if uniq_ratio < 0.55:
            return True

    if text in {"je ne sais pas.", "je sais pas.", "ok.", "d'accord."}:
        return True

    return False


def pick_mode_text(response_mode, court_text, expert_text):
    return expert_text if response_mode == "Expert" else court_text


def format_structured_expert(title, definition, example, best, pitfalls):
    return (
        f"{title}:\n"
        f"- Definition: {definition}\n"
        f"- Exemple: {example}\n"
        f"- Bonnes pratiques: {best}\n"
        f"- Pieges a eviter: {pitfalls}"
    )


def find_concept_card(q_norm):
    for card in CONCEPT_CARDS:
        if any(term in q_norm for term in card["keys"]):
            return card
    return None


def concept_reply(q_norm, response_mode):
    card = find_concept_card(q_norm)
    if not card:
        return None
    court = f"{card['title']}: {card['definition']}"
    expert = format_structured_expert(
        card["title"],
        card["definition"],
        card["example"],
        card["best"],
        card["pitfalls"],
    )
    return pick_mode_text(response_mode, court, expert)


def estimate_answer_confidence(user_text, answer, direct_hit=False):
    q_norm = _normalize_text(user_text or "")
    a_norm = _normalize_text(answer or "")
    score = 0.0
    if direct_hit:
        score += 0.55

    key_hits = 0
    for k in IN_DOMAIN_KEYWORDS:
        if k in q_norm:
            key_hits += 1
    score += min(0.30, key_hits * 0.04)

    if any(m in q_norm for m in QUESTION_MARKERS):
        score += 0.10
    if len(a_norm.split()) >= 18:
        score += 0.10

    return max(0.0, min(1.0, score))


def clarification_reply(response_mode):
    if response_mode == "Expert":
        return (
            "Pour te donner une reponse vraiment utile, precise l'un de ces axes:\n"
            "1) Definition (concept),\n"
            "2) Implementation (code/outils),\n"
            "3) Production (architecture, monitoring, cout).\n"
            "Tu peux aussi copier un extrait de code ou de donnees."
        )
    return "Je peux mieux t'aider si tu precises: definition, implementation, ou production."


def discussion_reply(user_text, response_mode):
    q_norm = _normalize_text(user_text or "")
    asks_depth = any(m in q_norm for m in {"detail", "details", "detaille", "approfond", "plus"})
    asks_remarks = any(m in q_norm for m in {"remarque", "remarques", "avis", "critique"})

    if not (asks_depth or asks_remarks):
        return None

    if "pipeline" in q_norm or "ml" in q_norm or "machine learning" in q_norm:
        court = (
            "Remarques pro: clarifier la metrique metier, verrouiller anti-data leakage, "
            "et monitorer drift/performance en production."
        )
        expert = (
            "Remarques d'expert sur un projet ML (version detaillee):\n"
            "- Definition utile: un bon modele ne suffit pas, il faut un systeme fiable bout en bout.\n"
            "- Point critique 1: eviter la fuite de donnees (split + preprocessing correctement ordonnes).\n"
            "- Point critique 2: aligner la metrique avec le cout metier (precision/recall/seuil).\n"
            "- Point critique 3: deploiement observable (latence, erreurs, drift, qualite predictive).\n"
            "- Point critique 4: boucle d'amelioration continue (analyse d'erreurs -> retraining cible).\n"
            "Si tu veux, je peux te faire une checklist operationnelle en 10 points pour ton cas."
        )
        return pick_mode_text(response_mode, court, expert)

    if "fastapi" in q_norm or "api" in q_norm:
        court = "Remarques API: contrats stricts, timeouts/retries, observabilite, versioning et rollback."
        expert = (
            "Remarques d'expert pour une API ML robuste:\n"
            "- Definition: une API production doit etre fiable, mesurable et reversible.\n"
            "- Details techniques: schema strict, validation metier, gestion d'erreurs homogène.\n"
            "- Remarque ops: monitorer P95/P99, taux d'erreur et saturation ressources.\n"
            "- Gouvernance: version du modele + version des features + plan de rollback.\n"
            "Si tu veux, je peux te proposer une structure de dossier FastAPI prete a coder."
        )
        return pick_mode_text(response_mode, court, expert)

    court = "Je peux te donner des remarques et details utiles; precise si tu veux angle data, modelisation ou production."
    expert = (
        "Je peux approfondir en mode expert avec des remarques concretes. Choisis un angle:\n"
        "1) Data (qualite, fuite, features),\n"
        "2) Modelisation (metriques, seuil, erreurs),\n"
        "3) Production (API, monitoring, retraining).\n"
        "Donne ton contexte et je te fais une analyse detaillee et actionnable."
    )
    return pick_mode_text(response_mode, court, expert)


def fallback_reply(user_text, profile, response_mode="Court"):
    q = (user_text or "").lower().strip()
    q_norm = re.sub(r"[^a-zà-öø-ÿ0-9\s]", "", q)

    discuss = discussion_reply(user_text, response_mode)
    if discuss:
        return discuss

    if not is_in_domain_query(q):
        return out_of_domain_reply()

    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    if ("comment" in q and "t'appelle" in q) or ("ton nom" in q) or ("qui es tu" in q) or ("qui es-tu" in q):
        return "Je m'appelle Elibot."

    if "je comprends pas" in q or "je ne comprends pas" in q:
        return "Pas de souci. Dis-moi juste ce que tu veux faire, en une phrase simple."

    if "merci" in q:
        return "Avec plaisir."

    concept = concept_reply(q_norm, response_mode)
    if concept:
        return concept

    # Provide a useful technical fallback for common in-domain asks.
    if "pipeline" in q_norm and ("ml" in q_norm or "machine learning" in q_norm):
        court = (
            "Version pro d'un pipeline ML de bout en bout: "
            "1) Cadrage, 2) data prep, 3) features + split propre, "
            "4) baseline + tuning, 5) evaluation, 6) deploiement API, 7) monitoring."
        )
        expert = (
            "Version pro d'un pipeline ML de bout en bout:\n"
            "1) Cadrage: objectif business, variable cible, metrique de succes.\n"
            "2) Data: collecte, controle qualite, nettoyage, gestion des valeurs manquantes.\n"
            "3) Features: encodage, normalisation, split train/validation/test sans fuite de donnees.\n"
            "4) Modelisation: baseline, tuning, comparaison des modeles.\n"
            "5) Evaluation: metriques adaptees + analyse des erreurs.\n"
            "6) Deploiement: API versionnee + monitoring.\n"
            "7) Exploitation: suivi de la derive des donnees et re-entrainement planifie."
        )
        return pick_mode_text(response_mode, court, expert)
    if "fastapi" in q_norm or ("api" in q_norm and "modele" in q_norm):
        court = (
            "Architecture FastAPI pro: /health + /predict + /version, "
            "schemas Pydantic, gestion d'erreurs centralisee, logs et metriques."
        )
        expert = (
            "Architecture FastAPI professionnelle recommandee:\n"
            "- Endpoints: /health, /predict, /version.\n"
            "- Contrats: schemas Pydantic stricts (input/output).\n"
            "- Fiabilite: gestion d'erreurs centralisee + timeouts + retries cotes clients.\n"
            "- Exploitation: logs structures, traces, metriques de latence et taux d'erreur.\n"
            "- Cycle de vie: versionnage du modele et strategie de rollback."
        )
        return pick_mode_text(response_mode, court, expert)
    if "pandas" in q_norm or "csv" in q_norm:
        court = (
            "Optimisation pandas (pro): typer les colonnes, vectoriser les operations, "
            "eviter apply ligne a ligne, optimiser les jointures, traiter les gros CSV par chunks."
        )
        expert = (
            "Optimisation pandas (niveau pro):\n"
            "- Typage explicite des colonnes des le chargement (dtype, parse_dates).\n"
            "- Operations vectorisees (eviter apply ligne par ligne).\n"
            "- Jointures performantes (index sur cles, colonnes reduites au strict necessaire).\n"
            "- Lecture de gros fichiers en chunks + ecriture incrementalisee.\n"
            "- Profiling systematique (temps, memoire) avant/apres optimisation."
        )
        return pick_mode_text(response_mode, court, expert)

    if "prenom" in profile:
        return f"D'accord {profile['prenom']}, peux-tu reformuler en une phrase simple ?"

    return clarification_reply(response_mode)


def update_profile_from_user_text(user_text, profile):
    text = user_text.strip()

    name_match = re.search(
        r"(?:je m'appelle|mon prenom est|mon prénom est|je suis)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        profile["prenom"] = name_match.group(1)

    city_match = re.search(
        r"(?:je viens de|j'habite a|j'habite à|je vis(?: actuellement)? a|je vis(?: actuellement)? à|ma ville est)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if city_match:
        profile["ville"] = city_match.group(1)

    asks_for_sport = bool(re.search(r"\bquel(?:le)?\s+sport\b", text, flags=re.IGNORECASE))
    if not asks_for_sport:
        sport_match = re.search(
            r"(?:j'aime le|j'aime la|j'adore le|j'adore la|mon sport prefere est le|mon sport prefere est la|mon sport prefere c'est le|mon sport prefere c'est la|mon sport préféré est le|mon sport préféré est la|mon sport préféré c'est le|mon sport préféré c'est la)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
            text,
            flags=re.IGNORECASE,
        )
        if not sport_match:
            sport_match = re.search(
                r"(?:mon sport prefere|mon sport préféré)\s*[,;:\-]?\s*(?:est|c'est)?\s*(?:le|la)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
                text,
                flags=re.IGNORECASE,
            )
        if sport_match:
            profile["sport"] = sport_match.group(1)

    goal_match = re.search(
        r"(?:objectif|courir|course|entrainement|entraînement|seance|séance)?.{0,20}?(\d+)\s*minutes",
        text,
        flags=re.IGNORECASE,
    )
    if goal_match:
        profile["objectif_minutes"] = goal_match.group(1)


def build_memory_lines(profile):
    lines = []
    if "prenom" in profile:
        lines.append(f"Memoire: Le prenom utilisateur est {profile['prenom']}.")
    if "ville" in profile:
        lines.append(f"Memoire: La ville utilisateur est {profile['ville']}.")
    if "sport" in profile:
        lines.append(f"Memoire: Le sport prefere utilisateur est {profile['sport']}.")
    if "objectif_minutes" in profile:
        lines.append(f"Memoire: L'objectif sport utilisateur est {profile['objectif_minutes']} minutes.")
    return lines


def maybe_rule_reply(user_text, profile, response_mode="Court"):
    q = user_text.lower().strip()
    q_norm = re.sub(r"[^a-zà-öø-ÿ0-9\s]", "", q)

    if "pipeline ml de bout en bout" in q_norm:
        court = "Pipeline ML: preprocessing, entrainement, evaluation, deploiement, monitoring."
        expert = (
            "Bonne question. Si tu veux un pipeline ML vraiment professionnel, il faut le penser comme un produit et pas seulement comme un modele. "
            "On commence par cadrer le probleme metier (decisions a prendre, KPI cibles, contraintes de latence/cout), puis on securise les donnees "
            "(qualite, nettoyage, schema stable, anti-data-leakage).\n\n"
            "Ensuite, on construit une baseline simple mais solide pour avoir un point de comparaison clair, avant de passer au tuning avance "
            "(cross-validation, choix des hyperparametres, calibration des seuils). La partie evaluation doit aller au-dela d'une metrique globale: "
            "il faut analyser les erreurs par segment (type client, geographie, plage temporelle) pour verifier que la performance est robuste.\n\n"
            "Cote production, je recommande une API versionnee avec observabilite complete (logs structures, latence P95, taux d'erreur, drift de donnees et de concept), "
            "plus un plan de retraining declenche par seuils. Par exemple, si la precision baisse durablement sur un segment critique, on lance une boucle de re-entrainement "
            "avec validation avant promotion.\n\n"
            "Si tu veux, je peux te donner la version operationnelle en 30-60-90 jours (MVP, hardening, industrialisation) selon ton contexte."
        )
        return pick_mode_text(response_mode, court, expert)

    if (
        ("pipeline" in q_norm)
        and ("ml" in q_norm or "machine learning" in q_norm)
        and ("fraude" in q_norm or "fraud" in q_norm)
        and ("temps reel" in q_norm or "temps reel" in q or "reel" in q_norm or "real time" in q_norm)
    ):
        court = (
            "Pour la fraude en temps reel, il faut raisonner en 4 blocs: signal (donnees), score (modele), "
            "decision (regles + seuils), et action (blocage, challenge, revue manuelle)."
        )
        expert = (
            "Excellente question. Pour detecter la fraude en temps reel, la logique pro n'est pas "
            "\"entrainer un modele\" puis deployer. Il faut construire une chaine de decision complete, avec un compromis clair entre fraude evitee et friction client.\n\n"
            "1) Raisonnement metier (avant la technique)\n"
            "- Objectif: reduire les pertes fraude sans casser l'experience des bons clients.\n"
            "- KPI a suivre ensemble: taux de fraude capturee, faux positifs, taux de challenge, latence de decision.\n"
            "- Regle cle: en fraude, un faux positif coute du churn; un faux negatif coute de l'argent. Le seuil se choisit avec le metier, pas uniquement avec F1.\n\n"
            "2) Architecture cible temps reel\n"
            "- Ingestion evenementielle (paiement, device, geoloc, historique court).\n"
            "- Feature store online pour recuperer des variables fraiches en quelques millisecondes.\n"
            "- Service de scoring (modele + regles expertes) expose en API faible latence.\n"
            "- Moteur de decision: accepter, challenger (OTP/3DS), ou bloquer.\n\n"
            "3) Strategie modelisation\n"
            "- Baseline interpretable d'abord (logistic/XGBoost simple), puis enrichissement progressif.\n"
            "- Dataset desequilibre: class weights, seuil optimise cout-risque, monitoring precision/recall par segment.\n"
            "- Evaluation temporelle obligatoire: split par temps pour simuler la vraie prod et eviter les illusions offline.\n\n"
            "4) Exploitation production\n"
            "- Observabilite en continu: latence P95, drift data/concept, taux de rejet par segment, alertes.\n"
            "- Boucle d'apprentissage: feedback analystes fraude, labellisation retardee, retraining pilote par seuils.\n"
            "- Plan de securite: fallback regles si modele indisponible, versioning strict, rollback instantane.\n\n"
            "Si tu veux, je peux te donner une architecture de reference concrete (Kafka + Feature Store + FastAPI + Redis + monitoring) avec budget latence cible < 150 ms."
        )
        return pick_mode_text(response_mode, court, expert)

    if (
        ("blueprint" in q_norm or "go live" in q_norm or "golive" in q_norm or "30 60 90" in q_norm)
        and ("pipeline" in q_norm or "ml" in q_norm or "machine learning" in q_norm or "fraude" in q_norm)
    ):
        court = (
            "Blueprint rapide: 30 jours pour stabiliser les donnees et la baseline, "
            "60 jours pour industrialiser le scoring temps reel, 90 jours pour monitoring/retraining pilote par KPI."
        )
        expert = (
            "Parfait, voici un blueprint concret en 30-60-90 jours pour un pipeline ML fraude temps reel.\n\n"
            "0-30 jours (fondations)\n"
            "- Cadrage metier: KPI cibles (fraude captee, faux positifs, latence de decision).\n"
            "- Donnees: schema d'evenements stable, qualite, anti-data leakage, labels fiables.\n"
            "- Baseline: modele simple + regles metier minimales pour avoir une reference solide.\n\n"
            "31-60 jours (industrialisation)\n"
            "- Scoring temps reel: service API faible latence + feature store online.\n"
            "- Moteur de decision: accepter / challenger / bloquer selon seuils et criticite.\n"
            "- Evaluation business: calibration du seuil avec metier (cout faux positifs vs faux negatifs).\n\n"
            "61-90 jours (fiabilisation production)\n"
            "- Observabilite complete: latence P95, drift data/concept, performance par segment.\n"
            "- Gouvernance: versionning modele/features, rollback, audit trail des decisions.\n"
            "- Retraining pilote: declenchement par seuils + validation avant promotion.\n\n"
            "Si tu veux, je peux te fournir la version architecture technique (components, contrats API, dashboards) directement exploitable par ton equipe."
        )
        return pick_mode_text(response_mode, court, expert)

    discuss = discussion_reply(user_text, response_mode)
    if discuss:
        return discuss

    # Always prioritize concept cards before domain gating.
    concept = concept_reply(q_norm, response_mode)
    if concept:
        return concept

    if not is_in_domain_query(q):
        return out_of_domain_reply()

    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    if "que fais tu" in q_norm or "tu fais quoi" in q_norm or "ton domaine" in q:
        return (
            "Je suis specialise en analyse de donnees, IA appliquee et automatisation. "
            "Je peux t'aider sur pipelines, modeles, API, debugging Python et workflows techniques."
        )

    # Concept intents: answer directly with useful, structured detail.
    if (
        ("machine learning" in q_norm)
        and ("cest quoi" in q_norm or "ca veut dire quoi" in q_norm or "definir" in q_norm or "definition" in q_norm or "explique" in q_norm)
    ):
        court = (
            "Le machine learning est une methode ou un modele apprend des patterns a partir de donnees "
            "pour predire ou classer sans regler chaque cas a la main."
        )
        expert = (
            "Le machine learning (ML), c'est l'apprentissage automatique: au lieu d'ecrire toutes les regles, "
            "on entraine un modele sur des exemples pour qu'il generalise sur de nouvelles donnees.\n"
            "- Entree: des donnees (features).\n"
            "- Sortie: une prediction (classe, valeur, score).\n"
            "- Processus: preparation des donnees, entrainement, evaluation, deploiement, monitoring.\n"
            "Exemples: detection de fraude, prevision de ventes, classification d'emails, recommandation."
        )
        return pick_mode_text(response_mode, court, expert)

    if (
        ("difference" in q_norm or "diff" in q_norm)
        and ("ia" in q_norm or "intelligence artificielle" in q_norm)
        and ("machine learning" in q_norm or "deep learning" in q_norm)
    ):
        court = (
            "IA est le domaine global; ML est une sous-partie de l'IA; Deep Learning est une sous-partie du ML "
            "basee sur des reseaux de neurones profonds."
        )
        expert = (
            "Difference IA / ML / Deep Learning:\n"
            "- IA: champ global pour creer des systemes intelligents.\n"
            "- ML: technique de l'IA qui apprend a partir de donnees.\n"
            "- Deep Learning: ML base sur des reseaux de neurones multi-couches.\n"
            "En pratique: IA est l'objectif, ML est la methode la plus utilisee, DL est tres performant sur image/texte/audio."
        )
        return pick_mode_text(response_mode, court, expert)

    # High-priority in-domain intents: answer directly instead of relying on generation.
    if "pipeline" in q_norm and ("ml" in q_norm or "machine learning" in q_norm):
        court = (
            "Version pro d'un pipeline ML de bout en bout:\n"
            "1) Cadrage (objectif/metrique), 2) collecte et preprocessing des donnees, "
            "3) feature engineering + split train/validation/test, 4) entrainement baseline + tuning, "
            "5) evaluation et analyse d'erreurs, 6) deploiement API, 7) monitoring et retraining."
        )
        expert = (
            "Tu as raison de viser une approche plus pro. Un bon pipeline ML ne doit pas juste aligner des etapes techniques: "
            "il doit relier chaque etape a une decision metier concrete.\n\n"
            "1) Cadrage: on clarifie la question metier, la metrique de succes et le cout de l'erreur. "
            "Exemple: faux negatif plus grave que faux positif en detection de fraude.\n"
            "2) Donnees: on industrialise la qualite (schema, valeurs manquantes, dedoublonnage, anti-fuite). "
            "Sans cette base, un modele performant en test peut echouer en production.\n"
            "3) Modelisation: baseline d'abord, puis tuning cible. L'objectif est d'ameliorer de facon mesurable, pas de complexifier pour rien.\n"
            "4) Evaluation: au-dela du score global, on lit les erreurs par segment et on ajuste le seuil de decision selon le risque metier.\n"
            "5) Mise en production: API versionnee + traçabilite (version modele, features, dataset) + rollback rapide.\n"
            "6) Monitoring continu: drift des donnees, derive du concept, latence, taux d'erreur, metriques metier; puis retraining pilote par seuils.\n\n"
            "Si tu veux, je te fais maintenant un blueprint concret (stack, endpoints, dashboards, et checklist de go-live) adapte a ton projet."
        )
        return pick_mode_text(response_mode, court, expert)
    if "pandas" in q_norm or "csv" in q_norm:
        court = (
            "Optimisation pandas (niveau pro): typer les colonnes, vectoriser les traitements, "
            "eviter apply ligne par ligne, indexer les cles de jointure, "
            "et traiter les gros volumes en chunks avec mesure de performance."
        )
        expert = (
            "Approche expert pour un pipeline pandas lent:\n"
            "- Profiling initial (cProfile/line_profiler + memoire).\n"
            "- Lecture optimisee (dtype, usecols, parse_dates, chunksize).\n"
            "- Transformations vectorisees, suppression des boucles Python.\n"
            "- Jointures: indexation des cles + reduction de cardinalite en amont.\n"
            "- Industrialisation: tests de non-regression perf + seuils de latence."
        )
        return pick_mode_text(response_mode, court, expert)
    if "fastapi" in q_norm or ("api" in q_norm and "modele" in q_norm):
        court = (
            "Architecture FastAPI professionnelle: /health + /predict + /version, "
            "schemas Pydantic stricts, gestion d'erreurs centralisee, logs structures, "
            "metriques de prod et versionnage du modele."
        )
        expert = (
            "Blueprint expert FastAPI pour servir un modele ML:\n"
            "- Couche API: /health, /ready, /predict, /version.\n"
            "- Contrats stricts: Pydantic + validation metier + gestion des valeurs aberrantes.\n"
            "- Robustesse: timeouts, retries, circuit breaker cote appelant.\n"
            "- Observabilite: logs structures, traces distribuees, metriques SLO (P95, taux d'erreur).\n"
            "- Gouvernance modele: registry, canary/rollback, version de features et de modele."
        )
        return pick_mode_text(response_mode, court, expert)

    asks_name = (
        ("prenom" in q)
        or ("prénom" in q)
        or ("comment je m'appelle" in q)
        or ("qui suis-je" in q)
        or ("comment tu t'appelle" in q)
        or ("comment tu t'appelles" in q)
        or ("ton nom" in q)
    )
    asks_city = ("ville" in q) or ("j'habite" in q) or ("je vis" in q)
    asks_sport = ("sport" in q) or ("j'aime" in q)
    asks_goal = ("objectif" in q) or ("course" in q) or ("entrain" in q)
    asks_advice = ("conseil" in q) or ("demain" in q)

    if (
        ("qui es-tu" in q)
        or ("qui es tu" in q)
        or ("ton nom" in q)
        or ("comment tu t'appelle" in q)
        or ("comment tu t'appelles" in q)
    ):
        return "Je m'appelle Elibot. Je suis la pour discuter avec toi de facon naturelle et utile."

    if asks_name and asks_city:
        if "prenom" in profile and "ville" in profile:
            return f"Si je me souviens bien, tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return None

    if asks_name and "prenom" in profile:
        return f"Bien sur: tu t'appelles {profile['prenom']}."

    if asks_city and "ville" in profile:
        return f"Tu m'avais dit que tu viens de {profile['ville']}."

    if asks_advice and "sport" in profile:
        return (
            f"Top, on fait simple pour demain: 30 minutes de {profile['sport']}, "
            "5 minutes d'echauffement au debut, puis etirements en fin de seance."
        )

    if asks_sport and "sport" in profile:
        return f"Ton sport prefere, c'est le {profile['sport']}."

    if asks_goal and "objectif_minutes" in profile:
        return f"Ton objectif, c'est de courir {profile['objectif_minutes']} minutes."

    if ("qui suis-je" in q or "qui je suis" in q) and "prenom" in profile:
        if "ville" in profile:
            return f"Tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return f"Tu t'appelles {profile['prenom']}."

    return None


def build_prompt(history, user_text, profile):
    lines = [f"Systeme: {SYSTEM_PROMPT}"]
    lines.extend(build_memory_lines(profile))

    if history:
        recent = history[-HISTORY_TURNS:]
        for user_msg, bot_msg in recent:
            lines.append(f"Utilisateur: {user_msg}")
            if HISTORY_MODE == "full":
                lines.append(f"Assistant: {bot_msg}")

    lines.append(f"Utilisateur: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _append_jsonl_safe(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break chat responses.
        return


def _log_chat_event(user_text: str, answer: str, response_mode: str, direct_hit: bool, confidence: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": now,
        "user": user_text,
        "assistant": answer,
        "response_mode": response_mode,
        "direct_hit": bool(direct_hit),
        "confidence": round(float(confidence), 4),
        "source": "hf_space",
    }
    _append_jsonl_safe(CHAT_LOG_PATH, payload)


def _log_audit_event(kind: str, user_text: str, answer: str, confidence: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": now,
        "kind": kind,
        "user": user_text,
        "assistant_len": len(answer or ""),
        "confidence": round(float(confidence), 4),
        "source": "hf_space",
    }
    _append_jsonl_safe(AUDIT_LOG_PATH, payload)


def _extract_python_blocks(text: str) -> list[str]:
    if not text:
        return []
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def _is_safe_plot_code(code: str) -> tuple[bool, str]:
    if not code or len(code) > 7000:
        return False, "Code vide ou trop long."

    lowered = code.lower()
    blocked_tokens = [
        "import os",
        "from os",
        "import sys",
        "from sys",
        "import subprocess",
        "from subprocess",
        "import socket",
        "from socket",
        "import requests",
        "from requests",
        "import pathlib",
        "from pathlib",
        "import shutil",
        "from shutil",
        "open(",
        "exec(",
        "eval(",
        "__",
    ]
    if any(tok in lowered for tok in blocked_tokens):
        return False, "Code refuse: operation non autorisee."

    allowed_import_roots = {
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sklearn",
    }
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("import "):
            mod = stripped.replace("import", "", 1).strip().split(" as ")[0].split(",")[0].strip()
            root = mod.split(".")[0]
            if root not in allowed_import_roots:
                return False, f"Import non autorise: {root}"
        if stripped.startswith("from "):
            root = stripped.replace("from", "", 1).strip().split("import")[0].strip().split(".")[0]
            if root not in allowed_import_roots:
                return False, f"Import non autorise: {root}"

    return True, "ok"


def _execute_plot_code(code: str) -> tuple[str | None, str]:
    safe, reason = _is_safe_plot_code(code)
    if not safe:
        return None, reason

    PLOT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    env = {
        "__builtins__": safe_builtins,
        "np": np,
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "feature_names": feature_names,
    }

    try:
        exec(compile(code, "<assistant_plot_code>", "exec"), env, env)
    except Exception as exc:
        return None, f"Execution echec: {type(exc).__name__}: {exc}"

    fig = plt.gcf()
    if not fig.get_axes():
        return None, "Aucun graphe detecte."

    out_path = PLOT_TMP_DIR / f"plot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path), "Graphe genere avec succes."


def render_last_python_plot(history):
    history = history or []
    if not history:
        return None, "Aucune reponse a analyser."

    last_answer = history[-1][1] if history[-1] else ""
    blocks = _extract_python_blocks(last_answer)
    if not blocks:
        return None, "Aucun bloc ```python``` trouve dans la derniere reponse."

    # Execute the last block first; if it fails, try previous ones.
    for block in reversed(blocks):
        img_path, status = _execute_plot_code(block)
        if img_path:
            return img_path, status

    return None, status


def chat(message, history, response_mode="Court"):
    history = history or []
    state = {
        "history": list(history),
        "profile": {},
    }
    for h_user, _ in state["history"]:
        update_profile_from_user_text(h_user, state["profile"])

    user_text = (message or "").strip()
    if not user_text:
        return state["history"], state["history"]

    update_profile_from_user_text(user_text, state["profile"])
    direct = None

    # Treat short confirmations ("oui", "ok", "vas y", etc.) as contextual follow-ups
    # before generic rule routing (which may otherwise fall back to out-of-domain).
    if state["history"] and is_short_affirmation(user_text):
        prev_user, prev_assistant = state["history"][-1]
        prev_assistant_norm = _normalize_text(prev_assistant or "")
        if any(marker in prev_assistant_norm for marker in ["si tu veux", "blueprint", "go live", "30 60 90", "je peux te", "je te fais"]):
            contextual_user_text = f"{prev_user} blueprint go live 30 60 90"
        else:
            contextual_user_text = f"{prev_user} approfondis avec plus de details"
        direct = maybe_rule_reply(contextual_user_text, state["profile"], response_mode=response_mode)

    if not direct:
        direct = maybe_rule_reply(user_text, state["profile"], response_mode=response_mode)

    # If the user writes an implicit follow-up ("approfondis", "plus de details", etc.),
    # reuse the previous user topic so the answer stays contextual instead of generic.
    if (not direct) and state["history"]:
        q_norm = _normalize_text(user_text)
        if any(marker in q_norm for marker in FOLLOWUP_MARKERS):
            prev_user = state["history"][-1][0]
            contextual_user_text = f"{prev_user} {user_text}"
            direct = maybe_rule_reply(contextual_user_text, state["profile"], response_mode=response_mode)

    if direct:
        answer = direct
        conf = estimate_answer_confidence(user_text, answer, direct_hit=True)
        if conf < 0.45:
            answer = clarification_reply(response_mode)
            _log_audit_event("low_confidence_direct", user_text, answer, conf)
        _log_chat_event(user_text, answer, response_mode=response_mode, direct_hit=True, confidence=conf)
        state["history"].append((user_text, answer))
        return state["history"], state["history"]

    prompt = build_prompt(state["history"], user_text, state["profile"])

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)

    use_sampling = TEMPERATURE > 0
    gen_kwargs = {
        "do_sample": use_sampling,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM,
        "num_beams": 1,
    }
    if use_sampling:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **gen_kwargs)

    answer = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    answer = clean_generated_text(answer)
    conf = estimate_answer_confidence(user_text, answer, direct_hit=False)
    if not answer or is_low_quality_answer(answer) or conf < 0.40:
        answer = fallback_reply(user_text, state["profile"], response_mode=response_mode)
        _log_audit_event("fallback_generation", user_text, answer, conf)

    _log_chat_event(user_text, answer, response_mode=response_mode, direct_hit=False, confidence=conf)
    state["history"].append((user_text, answer))
    return state["history"], state["history"]


def handle_submit(message, history, response_mode="Court"):
    new_chat, new_state = chat(message, history, response_mode=response_mode)
    return new_chat, new_state, ""


APP_CSS = """
.app-shell {
    max-width: 980px;
    margin: 0 auto;
    border-radius: 20px;
    border: 1px solid #d7e4da;
    background: linear-gradient(180deg, #f8fffb 0%, #f1f7f4 100%);
    box-shadow: 0 20px 40px rgba(18, 52, 38, 0.08);
    padding: 18px;
}
.hero-title h1 {
    margin: 0;
    color: #14532d;
    letter-spacing: 0.2px;
}
.hero-sub {
    color: #3f4f46;
    margin-top: 6px;
}
.quick-row .gr-button {
    border-radius: 999px !important;
    border: 1px solid #b6d8c4 !important;
    background: #e7f5ec !important;
    color: #0f5132 !important;
}
#send-btn {
    background: #166534 !important;
    color: white !important;
}
#reset-btn {
    background: #0f766e !important;
    color: white !important;
}
"""


with gr.Blocks(title="Elibot", css=APP_CSS, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.Markdown("## Elibot\nAssistant specialise en data, IA appliquee et automatisation.")

        chatbot = gr.Chatbot(
            label="Conversation",
            height=430,
            bubble_full_width=False,
            show_copy_button=True,
        )
        state = gr.State([])

        with gr.Row(elem_classes=["quick-row"]):
            quick_1 = gr.Button("Explique un pipeline ML", size="sm")
            quick_2 = gr.Button("Corrige ce code pandas", size="sm")
            quick_3 = gr.Button("Architecture API FastAPI", size="sm")
            quick_4 = gr.Button("Automatiser un workflow CSV", size="sm")

        msg = gr.Textbox(
            label="Message",
            placeholder="Ecris ton message ici...",
            lines=2,
            max_lines=4,
        )

        response_mode = gr.Radio(
            choices=RESPONSE_MODES,
            value="Expert",
            label="Style de reponse",
            info="Court: synthese rapide | Expert: version detaillee et operationnelle",
        )

        with gr.Row():
            send = gr.Button("Envoyer", elem_id="send-btn", variant="primary")
            clear = gr.Button("Reinitialiser", elem_id="reset-btn")

        with gr.Row():
            run_plot = gr.Button("Executer le code Python (graphe)")

        plot_image = gr.Image(label="Graphe genere", type="filepath")
        plot_status = gr.Markdown(value="Aucun graphe execute.")

    send.click(handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg])
    msg.submit(handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg])
    clear.click(lambda: ([], [], ""), inputs=None, outputs=[chatbot, state, msg])
    run_plot.click(render_last_python_plot, inputs=[state], outputs=[plot_image, plot_status])

    quick_1.click(lambda: "Peux-tu expliquer un pipeline machine learning de bout en bout ?", outputs=[msg]).then(
        handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg]
    )
    quick_2.click(lambda: "Voici un script pandas lent, comment l'optimiser ?", outputs=[msg]).then(
        handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg]
    )
    quick_3.click(lambda: "Propose une architecture FastAPI pour servir un modele ML.", outputs=[msg]).then(
        handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg]
    )
    quick_4.click(lambda: "Comment automatiser un workflow de nettoyage CSV en Python ?", outputs=[msg]).then(
        handle_submit, inputs=[msg, state, response_mode], outputs=[chatbot, state, msg]
    )


demo.launch()
