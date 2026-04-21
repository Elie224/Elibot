import csv
from pathlib import Path

OUT_PATH = Path("data/processed/chatbot_train_fr_ml_150_advanced.csv")

SECTIONS = {
    "erreurs_frequentes": [
        "Pourquoi un modele overfit meme avec beaucoup de donnees",
        "Pourquoi un modele underfit meme avec un modele complexe",
        "Pourquoi un modele converge lentement",
        "Pourquoi un modele diverge",
        "Pourquoi un modele donne toujours la meme prediction",
        "Pourquoi un modele est instable entre deux runs",
        "Pourquoi un modele est sensible aux outliers",
        "Pourquoi un modele ne generalise pas",
        "Pourquoi un modele a une variance elevee",
        "Pourquoi un modele a un biais eleve",
        "Pourquoi un modele ne s ameliore plus apres un certain point",
        "Pourquoi un modele performe mal sur les classes minoritaires",
        "Pourquoi un modele performe mal sur les donnees reelles",
        "Pourquoi un modele performe mal apres deploiement",
        "Pourquoi un modele performe mal sur un nouveau pays",
        "Pourquoi un modele performe mal sur un nouveau segment client",
        "Pourquoi un modele performe mal apres mise a jour des donnees",
        "Pourquoi un modele performe mal apres un changement de distribution",
        "Pourquoi un modele performe mal malgre un bon score offline",
        "Pourquoi un modele performe mal malgre un bon tuning",
    ],
    "arbitrages": [
        "Comment choisir entre un modele simple et un modele complexe",
        "Comment choisir entre SVM et Random Forest",
        "Comment choisir entre XGBoost et Neural Network",
        "Comment choisir entre PCA et autoencoder",
        "Comment choisir entre K-means et DBSCAN",
        "Comment choisir entre regression lineaire et Ridge",
        "Comment choisir entre Ridge et Lasso",
        "Comment choisir entre GridSearch et RandomSearch",
        "Comment choisir entre accuracy et F1",
        "Comment choisir entre ROC-AUC et PR-AUC",
        "Comment choisir entre normalisation et standardisation",
        "Comment choisir entre one-hot et embeddings",
        "Comment choisir entre batchnorm et layernorm",
        "Comment choisir entre dropout et L2",
        "Comment choisir entre early stopping et plus d epoques",
        "Comment choisir entre CPU et GPU",
        "Comment choisir entre PyTorch et TensorFlow",
        "Comment choisir entre modele rapide et modele precis",
        "Comment choisir entre modele interpretable et modele performant",
        "Comment choisir entre modele unique et ensemble",
    ],
    "optimisation": [
        "Comment fonctionne gradient descent",
        "Comment fonctionne stochastic gradient descent",
        "Comment fonctionne Adam",
        "Comment fonctionne RMSProp",
        "Comment fonctionne momentum",
        "Comment choisir un learning rate",
        "Comment choisir un batch size",
        "Comment choisir un nombre d epoques",
        "Comment detecter un learning rate trop eleve",
        "Comment detecter un learning rate trop faible",
        "Comment detecter un batch size trop petit",
        "Comment detecter un batch size trop grand",
        "Comment optimiser un modele profond",
        "Comment optimiser un modele tabulaire",
        "Comment optimiser un modele NLP",
        "Comment optimiser un modele vision",
        "Comment optimiser un modele RL",
        "Comment optimiser un modele en production",
        "Comment optimiser un pipeline ML complet",
        "Comment optimiser un modele sous contrainte de latence",
    ],
    "donnees": [
        "Comment detecter un dataset corrompu",
        "Comment detecter un dataset desequilibre",
        "Comment detecter un dataset bruite",
        "Comment detecter un dataset biaise",
        "Comment detecter un dataset incomplet",
        "Comment detecter un dataset non representatif",
        "Comment detecter un dataset avec fuite de donnees",
        "Comment detecter un dataset avec duplicats",
        "Comment detecter un dataset avec anomalies",
        "Comment detecter un dataset avec derive",
        "Comment nettoyer un dataset",
        "Comment enrichir un dataset",
        "Comment augmenter un dataset",
        "Comment fusionner plusieurs datasets",
        "Comment gerer les donnees temporelles",
        "Comment gerer les donnees geographiques",
        "Comment gerer les donnees textuelles",
        "Comment gerer les donnees images",
        "Comment gerer les donnees tabulaires",
        "Comment gerer les donnees multi-modalites",
    ],
    "modeles_modernes": [
        "Comment fonctionne un transformer",
        "Comment fonctionne l attention",
        "Comment fonctionne self-attention",
        "Comment fonctionne multi-head attention",
        "Comment fonctionne positional encoding",
        "Comment fonctionne un encoder",
        "Comment fonctionne un decoder",
        "Comment fonctionne BERT",
        "Comment fonctionne GPT",
        "Comment fonctionne ViT",
        "Comment fonctionne CLIP",
        "Comment fonctionne diffusion model",
        "Comment fonctionne un LLM",
        "Comment fonctionne fine-tuning",
        "Comment fonctionne LoRA",
        "Comment fonctionne quantization",
        "Comment fonctionne distillation",
        "Comment fonctionne retrieval-augmented generation",
        "Comment fonctionne un modele hybride ML + regles",
        "Comment fonctionne un modele multi-modal",
    ],
    "theorie_statistique": [
        "Qu est-ce qu un estimateur",
        "Qu est-ce qu un biais d estimateur",
        "Qu est-ce qu une variance d estimateur",
        "Qu est-ce qu un intervalle de confiance",
        "Qu est-ce qu un test statistique",
        "Qu est-ce qu une p-value",
        "Qu est-ce qu un intervalle de prediction",
        "Qu est-ce qu une distribution normale",
        "Qu est-ce qu une distribution log-normale",
        "Qu est-ce qu une distribution exponentielle",
        "Qu est-ce qu un bootstrap",
        "Qu est-ce qu un jackknife",
        "Qu est-ce qu un estimateur robuste",
        "Qu est-ce qu un estimateur bayesien",
        "Qu est-ce qu un prior",
        "Qu est-ce qu un posterior",
        "Qu est-ce qu un MAP",
        "Qu est-ce qu un MLE",
        "Qu est-ce qu un intervalle bayesien",
        "Qu est-ce qu un modele generatif",
    ],
    "mlops_production": [
        "Comment deployer un modele",
        "Comment monitorer un modele",
        "Comment detecter le drift",
        "Comment detecter le concept drift",
        "Comment detecter le data drift",
        "Comment gerer le retraining",
        "Comment gerer le versioning",
        "Comment gerer le rollback",
        "Comment gerer la latence",
        "Comment gerer la scalabilite",
        "Comment gerer la memoire",
        "Comment gerer les logs",
        "Comment gerer les erreurs",
        "Comment gerer les timeouts",
        "Comment gerer les quotas",
        "Comment gerer les API",
        "Comment gerer les batch jobs",
        "Comment gerer les pipelines",
        "Comment gerer les tests ML",
        "Comment gerer les tests data",
        "Comment gerer les tests de performance",
        "Comment gerer les tests de robustesse",
        "Comment gerer les tests de securite",
        "Comment gerer les tests de non-regression",
        "Comment gerer les tests de fairness",
        "Comment gerer les tests de stabilite",
        "Comment gerer les tests de coherence",
        "Comment gerer les tests multi-modeles",
        "Comment gerer les tests multi-environnements",
        "Comment gerer un modele en production sur plusieurs pays",
    ],
}


def build_response(section: str, question: str, idx: int) -> str:
    prefix = f"Question: {question}\n"

    if section == "erreurs_frequentes":
        body = (
            "Diagnostic expert:\n"
            "- Symptome observe: verifier la metrique, le segment impacte et la fenetre temporelle.\n"
            "- Causes probables: split non representatif, preprocessing inconsistent, regularisation inadaptee, ou drift.\n"
            "- Verification rapide: comparer train/validation/production, analyser erreurs par sous-population, et controler les features dominantes.\n"
            "- Correctifs concrets: revoir split temporel, recalibrer seuil, ajuster capacite modele, renforcer nettoyage et monitoring.\n"
            "- Regle terrain: prioriser la reduction du risque metier avant de chercher un gain global de score."
        )
    elif section == "arbitrages":
        body = (
            "Cadre de decision:\n"
            "- Contrainte business: latence, cout inference, explicabilite, robustesse reglementaire.\n"
            "- Profil data: volume, bruit, dimension, sparsity, derive attendue.\n"
            "- Evaluation: comparer baseline simple vs candidat avance sur les memes splits et memes KPIs metier.\n"
            "- Arbitrage final: choisir le modele qui maximise valeur metier nette, pas seulement la meilleure metrique offline.\n"
            "- Bon reflexe: documenter clairement pourquoi une option est rejetee pour faciliter le rollback futur."
        )
    elif section == "optimisation":
        body = (
            "Procedure d optimisation:\n"
            "- Stabiliser la base: pipeline deterministic, seed fixe, metriques de reference, checks data.\n"
            "- Explorer hyperparametres clefs: learning rate, batch size, regularisation, scheduler, early stopping.\n"
            "- Lire les signaux: courbes loss, ecart train/val, gradient norms, saturation/instabilite.\n"
            "- Ajuster par priorite: d abord convergence, ensuite generalisation, enfin latence et cout de serving.\n"
            "- Sortie attendue: configuration reproductible avec compromis performance-latence documente."
        )
    elif section == "donnees":
        body = (
            "Workflow data engineering:\n"
            "- Profilage: schema, types, manquants, duplicats, outliers, valeurs impossibles.\n"
            "- Qualite: tests automatiques sur fraicheur, unicite, plage, coherence inter-colonnes.\n"
            "- Representativite: couverture par segment, saison, geographie, canal, device.\n"
            "- Remediation: nettoyage versionne, enrichissement cible, et trace des transformations.\n"
            "- Production: monitorer les ruptures de distribution et relier chaque derive a une action de retraining."
        )
    elif section == "modeles_modernes":
        body = (
            "Explication operationnelle:\n"
            "- Intuition: decrire le mecanisme central et ce qu il optimise effectivement.\n"
            "- Architecture: composants critiques, cout compute, sensibilite aux donnees.\n"
            "- Limites: hallucination, fuite memorielle, biais, instabilite hors distribution.\n"
            "- Levier production: fine-tuning cible, quantization, distillation, RAG et garde-fous.\n"
            "- Validation: evaluer qualite, robustesse et securite avant promotion en prod."
        )
    elif section == "theorie_statistique":
        body = (
            "Lecture statistique:\n"
            "- Definition formelle: preciser l objet, ses hypotheses et son domaine de validite.\n"
            "- Interpretation: ce que la quantite mesure reellement et ce qu elle ne garantit pas.\n"
            "- Pieges: confusion frequentiste/bayesienne, p-value mal interpretee, leakage dans l estimation.\n"
            "- Usage ML: impact sur calibration, selection de modele, intervalle d incertitude, decision seuil.\n"
            "- Bon usage: toujours contextualiser par la taille d effet et le cout d erreur metier."
        )
    else:
        body = (
            "Playbook MLOps:\n"
            "- Design: definir SLO/SLI, contrats d entree, budget latence et capacite.\n"
            "- Fiabilite: observabilite bout-en-bout, tracing, alertes actionnables, runbooks de crise.\n"
            "- Qualite continue: tests data/ML/non-regression avant et apres chaque release.\n"
            "- Gouvernance: versioning modele-data-code, approvals, rollback instantane, auditabilite.\n"
            "- Multi-pays: adaptation locale, seuils regionaux, surveillance drift par marche."
        )

    suffix = (
        "\nMini checklist:\n"
        "1) Verifier metrique principale + metrique de risque.\n"
        "2) Segmenter les erreurs par population critique.\n"
        "3) Valider la reproductibilite sur un rerun complet."
    )

    return prefix + body + suffix


def main() -> None:
    rows = []
    for section, questions in SECTIONS.items():
        for idx, question in enumerate(questions, start=1):
            rows.append(
                {
                    "instruction": question,
                    "response": build_response(section, question, idx),
                    "history": "Utilisateur: je veux une reponse experte, exploitable en production. ||| Assistant: je fournis un diagnostic structure, des arbitrages et une checklist actionnable.",
                    "source": "ml_150_advanced_essentials",
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={OUT_PATH}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
