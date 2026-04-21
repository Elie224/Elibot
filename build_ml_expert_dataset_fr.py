import argparse
import csv
import random
from pathlib import Path


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            " ".join((row.get("instruction") or "").lower().split()),
            " ".join((row.get("response") or "").lower().split()),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_ml_concepts_deep_rows(target_rows: int) -> list[dict[str, str]]:
    cards = [
        {
            "name": "machine learning",
            "definition": "Le machine learning apprend des patterns a partir de donnees pour predire/classer.",
            "example": "Predire le churn client a partir des usages.",
            "kpi": "F1, precision, recall et cout metier",
            "pitfalls": "data leakage, split invalide, seuil non aligne metier",
        },
        {
            "name": "overfitting",
            "definition": "Le modele memorise le train et generalise mal.",
            "example": "Train score haut, test score bas.",
            "kpi": "gap train/validation, variance des folds",
            "pitfalls": "modele trop complexe, tuning excessif",
        },
        {
            "name": "underfitting",
            "definition": "Le modele est trop simple pour capter le signal.",
            "example": "Scores bas sur train et test.",
            "kpi": "baseline trop faible et erreur elevee",
            "pitfalls": "features pauvres, regularisation trop forte",
        },
        {
            "name": "bias variance",
            "definition": "Compromis entre erreur de simplification et sensibilite au bruit.",
            "example": "Modele lineaire (biais), arbre profond (variance).",
            "kpi": "stabilite cross-validation",
            "pitfalls": "optimiser seulement le train",
        },
        {
            "name": "data leakage",
            "definition": "Fuite d'informations futures/cibles dans les features.",
            "example": "Scaler fit sur train+test.",
            "kpi": "ecart offline/online suspect",
            "pitfalls": "preprocessing global avant split",
        },
        {
            "name": "cross-validation",
            "definition": "Evaluation robuste sur plusieurs partitions.",
            "example": "Stratified k-fold en classification desequilibree.",
            "kpi": "moyenne et ecart-type des scores",
            "pitfalls": "fuite dans le pipeline",
        },
        {
            "name": "metriques classification",
            "definition": "Accuracy, precision, recall, F1, ROC-AUC mesurent des choses differentes.",
            "example": "Fraude: recall et PR-AUC prioritaires.",
            "kpi": "precision@recall cible, cout faux positifs",
            "pitfalls": "accuracy seule en classes desequilibrees",
        },
        {
            "name": "metriques regression",
            "definition": "MAE, RMSE, R2 selon nature des erreurs et usage metier.",
            "example": "RMSE utile si grosses erreurs critiques.",
            "kpi": "MAE par segment metier",
            "pitfalls": "metrique non alignee avec objectif produit",
        },
    ]

    frames = [
        "Explique {name} de facon complete et structuree.",
        "Donne une explication experte de {name} avec cas d'usage et limites.",
        "Je veux comprendre {name}: definition, exemple, KPI, pieges, actions.",
        "Fais une fiche de reference sur {name} pour un projet ML en production.",
    ]
    contexts = [
        "fraude bancaire",
        "scoring client",
        "maintenance predictive",
        "prevision de ventes",
        "recommandation de produits",
        "classification de tickets support",
        "detection d'anomalies IoT",
        "vision industrielle",
    ]
    audiences = [
        "debutant data",
        "data analyst",
        "data scientist",
        "ml engineer",
        "lead technique",
    ]
    objectives = [
        "reduire les faux positifs",
        "ameliorer la robustesse hors echantillon",
        "stabiliser la performance production",
        "accelerer la mise en production",
        "améliorer l'interpretabilite",
        "reduire le cout inference",
        "ameliorer le rappel sur cas rares",
        "améliorer la precision au seuil metier",
        "renforcer la qualite des donnees",
        "eviter les regressions de performance",
        "mieux calibrer les probabilites",
        "fiabiliser les diagnostics d'erreurs",
    ]
    controls = [
        "controle schema et distribution",
        "controle fuite de donnees",
        "controle metrique par segment",
        "controle de calibration",
        "controle robustesse temporelle",
        "controle latence et cout",
        "controle drift et alerting",
        "controle rollback",
        "controle seuil de decision",
        "controle qualite labels",
    ]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        card = cards[i % len(cards)]
        instruction = (
            f"{frames[i % len(frames)].format(name=card['name'])} "
            f"Contexte: {contexts[i % len(contexts)]}. Public: {audiences[i % len(audiences)]}."
        )
        response = (
            f"Definition: {card['definition']}\n"
            f"Exemple concret: {card['example']}\n"
            f"Indicateurs a suivre: {card['kpi']}\n"
            f"Erreurs frequentes: {card['pitfalls']}\n"
            f"Objectif prioritaire: {objectives[i % len(objectives)]}.\n"
            "Checklist actionnable:\n"
            "1) Verifier la qualite des donnees et le split.\n"
            "2) Choisir metriques et seuil selon cout metier.\n"
            f"3) Monitorer drift/performance en production pour {contexts[(i + 3) % len(contexts)]}.\n"
            f"Controle cle: {controls[i % len(controls)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux une reponse complete. ||| Assistant: Reponse structuree niveau expert.",
                "source": "ml_concepts_deep",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_algorithms_deep_rows(target_rows: int) -> list[dict[str, str]]:
    algos = [
        {
            "name": "regression lineaire",
            "intuition": "modeliser une relation lineaire entre features et cible",
            "formula": "y_hat = beta0 + beta1*x1 + ... + betan*xn",
            "params": "fit_intercept, alpha (Ridge/Lasso)",
            "code": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression().fit(X_train, y_train)",
            "use": "prevision continue interpretable",
            "pitfalls": "non-linearite ignoree, outliers non traites",
        },
        {
            "name": "regression logistique",
            "intuition": "estimer une probabilite de classe",
            "formula": "p(y=1|x)=1/(1+exp(-w.x))",
            "params": "C, penalty, class_weight",
            "code": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000).fit(X_train, y_train)",
            "use": "classification interpretable",
            "pitfalls": "features mal scalees, seuil non ajuste",
        },
        {
            "name": "svm",
            "intuition": "maximiser la marge entre classes",
            "formula": "min 1/2||w||^2 + C*sum(xi)",
            "params": "C, kernel, gamma",
            "code": "from sklearn.svm import SVC\nmodel = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)",
            "use": "classification robuste sur taille moyenne",
            "pitfalls": "scaling oublie, cout inference eleve",
        },
        {
            "name": "knn",
            "intuition": "predire selon voisins les plus proches",
            "formula": "classe majoritaire dans les k voisins",
            "params": "n_neighbors, metric, weights",
            "code": "from sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)",
            "use": "baseline rapide non parametrique",
            "pitfalls": "dimensionnalite elevee, inference lente",
        },
        {
            "name": "random forest",
            "intuition": "moyenner plusieurs arbres decorreles",
            "formula": "bagging d'arbres + vote/moyenne",
            "params": "n_estimators, max_depth, min_samples_leaf",
            "code": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)",
            "use": "tabulaire non-lineaire robuste",
            "pitfalls": "modele lourd, explication locale necessaire",
        },
        {
            "name": "xgboost",
            "intuition": "booster sequentiellement des arbres faibles",
            "formula": "F_t(x)=F_{t-1}(x)+eta*h_t(x)",
            "params": "n_estimators, max_depth, learning_rate, subsample",
            "code": "from xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05).fit(X_train, y_train)",
            "use": "forte performance tabulaire",
            "pitfalls": "surapprentissage si tuning excessif",
        },
        {
            "name": "k-means",
            "intuition": "regrouper en k clusters par proximite",
            "formula": "min somme ||x - mu_c||^2",
            "params": "n_clusters, init, n_init",
            "code": "from sklearn.cluster import KMeans\nmodel = KMeans(n_clusters=4, random_state=42, n_init='auto').fit(X)",
            "use": "segmentation client",
            "pitfalls": "choix de k arbitraire, sensibles aux outliers",
        },
        {
            "name": "pca",
            "intuition": "projeter sur composantes maximisant variance",
            "formula": "X_proj = X * W_k",
            "params": "n_components, whiten",
            "code": "from sklearn.decomposition import PCA\npca = PCA(n_components=2).fit(X_train)\nX_train_pca = pca.transform(X_train)",
            "use": "reduction dimensionnelle et visualisation",
            "pitfalls": "interpretation difficile des composantes",
        },
        {
            "name": "dbscan",
            "intuition": "clusters par densite + detection bruit",
            "formula": "voisinage epsilon avec min_samples",
            "params": "eps, min_samples, metric",
            "code": "from sklearn.cluster import DBSCAN\nlabels = DBSCAN(eps=0.4, min_samples=10).fit_predict(X)",
            "use": "formes non-convexes et anomalies",
            "pitfalls": "hyperparametres sensibles a l'echelle",
        },
        {
            "name": "q-learning",
            "intuition": "apprendre une table de valeur action-etat",
            "formula": "Q(s,a)=Q(s,a)+alpha[r+gamma*max_a'Q(s',a')-Q(s,a)]",
            "params": "alpha, gamma, epsilon",
            "code": "Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s2].max() - Q[s,a])",
            "use": "rl discret et strategie par exploration",
            "pitfalls": "exploration insuffisante, convergence lente",
        },
    ]

    frames = [
        "Explique l'algorithme {name} en profondeur avec formule, hyperparametres et code.",
        "Donne une fiche complete sur {name}: intuition, code python, cas d'usage et erreurs frequentes.",
        "Comment fonctionne {name} et quand l'utiliser en production ?",
    ]
    contexts = ["tabulaire", "serie temporelle", "classification binaire", "detection anomalies", "segmentation", "reco"]
    constraints = ["latence basse", "interpretabilite", "cout cloud", "dataset bruite", "classes desequilibrees"]
    deployment_modes = ["batch", "temps reel", "quasi temps reel", "edge", "A/B test", "canary"]
    risk_controls = [
        "monitoring de derive",
        "validation temporelle",
        "seuil adapte au cout metier",
        "regularisation et early stopping",
        "analyse d'erreurs par segment",
        "plan de rollback",
    ]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        algo = algos[i % len(algos)]
        instruction = (
            f"{frames[i % len(frames)].format(name=algo['name'])} "
            f"Contexte: {contexts[i % len(contexts)]}; contrainte: {constraints[i % len(constraints)]}."
        )
        response = (
            f"Algorithme: {algo['name']}\n"
            f"Intuition: {algo['intuition']}\n"
            f"Formule: {algo['formula']}\n"
            f"Hyperparametres cle: {algo['params']}\n"
            "Code Python:\n"
            "```python\n"
            f"{algo['code']}\n"
            "```\n"
            f"Cas d'usage: {algo['use']}\n"
            f"Erreurs frequentes: {algo['pitfalls']}\n"
            f"Mode de deploiement conseille: {deployment_modes[i % len(deployment_modes)]}.\n"
            f"Controle risque prioritaire: {risk_controls[i % len(risk_controls)]}.\n"
            f"Remarque expert: valider sur plusieurs splits et comparer a une baseline simple sous contrainte {constraints[(i + 2) % len(constraints)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux theorie + code + limites. ||| Assistant: Reponse algorithmique complete.",
                "source": "ml_algorithms_deep",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_learning_types_rows(target_rows: int) -> list[dict[str, str]]:
    types = [
        {
            "name": "apprentissage supervise",
            "algos": "regression, random forest, xgboost, svm",
            "example": "predire churn ou score risque",
            "limit": "dependance a labels de qualite",
            "code": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier().fit(X_train, y_train)",
        },
        {
            "name": "apprentissage non supervise",
            "algos": "k-means, dbscan, pca",
            "example": "segmentation client",
            "limit": "evaluation plus difficile sans cible",
            "code": "from sklearn.cluster import KMeans\nlabels = KMeans(n_clusters=4, random_state=42).fit_predict(X)",
        },
        {
            "name": "apprentissage semi-supervise",
            "algos": "pseudo-labeling, consistency training",
            "example": "peu de labels + beaucoup de donnees non annotees",
            "limit": "propagation d'erreurs de pseudo labels",
            "code": "# 1) entrainer sur labels\n# 2) pseudo-labeler unlabeled\n# 3) re-entrainer sur combine",
        },
        {
            "name": "apprentissage par renforcement",
            "algos": "q-learning, policy gradient",
            "example": "agent qui optimise une politique de decision",
            "limit": "cout d'exploration et stabilite apprentissage",
            "code": "# update Q\nQ[s,a] += alpha*(r + gamma*Q[s2].max() - Q[s,a])",
        },
    ]

    frames = [
        "Explique {name} avec algorithmes, cas d'usage, limites et mini-code.",
        "Je veux une synthese complete sur {name} avec schema de pipeline.",
        "Donne un guide pratique pour {name}.",
    ]
    sectors = ["banque", "retail", "industrie", "SaaS", "telecom", "sante"]
    maturities = ["POC", "MVP", "production", "scale-up", "legacy modernization"]
    validation_modes = ["offline CV", "backtesting", "shadow mode", "A/B", "champion-challenger"]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        t = types[i % len(types)]
        instruction = (
            f"{frames[i % len(frames)].format(name=t['name'])} "
            f"Secteur cible: {sectors[i % len(sectors)]}; maturite: {maturities[i % len(maturities)]}."
        )
        response = (
            f"Type: {t['name']}\n"
            f"Algorithmes typiques: {t['algos']}\n"
            f"Cas d'usage: {t['example']}\n"
            f"Limites: {t['limit']}\n"
            "Mini-code:\n"
            "```python\n"
            f"{t['code']}\n"
            "```\n"
            "Schema pipeline (texte):\n"
            f"donnees -> preprocessing -> apprentissage -> evaluation -> decision ({sectors[(i + 1) % len(sectors)]})\n"
            f"Validation recommandee: {validation_modes[i % len(validation_modes)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux comprendre les types d'apprentissage. ||| Assistant: comparaison structuree.",
                "source": "ml_learning_types",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_implementation_rows(target_rows: int) -> list[dict[str, str]]:
    topics = [
        {
            "name": "pipeline sklearn end-to-end",
            "code": "from sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\n\nnum_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\ncat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])\npre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])\nclf = Pipeline([('pre', pre), ('model', LogisticRegression(max_iter=1000))])\nclf.fit(X_train, y_train)",
        },
        {
            "name": "gridsearch tuning",
            "code": "from sklearn.model_selection import GridSearchCV\nparam_grid = {'model__C':[0.1,1,3], 'model__class_weight':[None,'balanced']}\nsearch = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)\nsearch.fit(X_train, y_train)",
        },
        {
            "name": "randomizedsearch tuning",
            "code": "from sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import loguniform\nparam_dist = {'model__C': loguniform(1e-3, 1e1)}\nsearch = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, scoring='f1', cv=5, random_state=42)\nsearch.fit(X_train, y_train)",
        },
        {
            "name": "pytorch training loop minimal",
            "code": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\nmodel = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, n_classes))\nopt = optim.Adam(model.parameters(), lr=1e-3)\nloss_fn = nn.CrossEntropyLoss()\nfor xb, yb in loader:\n    opt.zero_grad()\n    logits = model(xb)\n    loss = loss_fn(logits, yb)\n    loss.backward()\n    opt.step()",
        },
        {
            "name": "tensorflow keras minimal",
            "code": "import tensorflow as tf\nmodel = tf.keras.Sequential([\n    tf.keras.layers.Dense(64, activation='relu'),\n    tf.keras.layers.Dense(n_classes, activation='softmax')\n])\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\nmodel.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)",
        },
        {
            "name": "optuna hyperparameter tuning",
            "code": "import optuna\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score\n\ndef objective(trial):\n    model = RandomForestClassifier(\n        n_estimators=trial.suggest_int('n_estimators', 100, 600),\n        max_depth=trial.suggest_int('max_depth', 3, 20),\n        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),\n        random_state=42,\n        n_jobs=-1,\n    )\n    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()\n\nstudy = optuna.create_study(direction='maximize')\nstudy.optimize(objective, n_trials=40)",
        },
    ]

    frames = [
        "Donne une implementation complete sur {name} avec bonnes pratiques production.",
        "Je veux un exemple executable pour {name}.",
        "Montre comment coder {name} proprement.",
    ]
    stacks = ["sklearn", "pytorch", "tensorflow", "optuna", "mlflow"]
    infra = ["API FastAPI", "batch hebdo", "streaming", "CI/CD", "monitoring drift"]
    data_shapes = ["10k lignes", "100k lignes", "1M lignes", "haute cardinalite", "donnees manquantes", "features mixtes"]
    ops_focus = ["observabilite", "tests", "cout", "latence", "resilience", "gouvernance"]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        t = topics[i % len(topics)]
        instruction = (
            f"{frames[i % len(frames)].format(name=t['name'])} "
            f"Stack: {stacks[i % len(stacks)]}; cible: {infra[i % len(infra)]}; donnees: {data_shapes[i % len(data_shapes)]}."
        )
        response = (
            f"Implementation: {t['name']}\n"
            "Code:\n"
            "```python\n"
            f"{t['code']}\n"
            "```\n"
            "Conseils pro:\n"
            "- Verifier les splits et la fuite de donnees.\n"
            "- Logger metriques train/validation.\n"
            "- Versionner modele + donnees + hyperparametres.\n"
            f"- Ajouter tests de robustesse et monitoring post-deploiement ({infra[(i + 2) % len(infra)]}).\n"
            f"Priorite ops: {ops_focus[i % len(ops_focus)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux du code exploitable. ||| Assistant: exemple complet et pratique.",
                "source": "ml_implementation",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_visualizations_rows(target_rows: int) -> list[dict[str, str]]:
    viz = [
        {
            "name": "courbe ROC",
            "code": "from sklearn.metrics import RocCurveDisplay\nimport matplotlib.pyplot as plt\n\nRocCurveDisplay.from_estimator(model, X_test, y_test)\nplt.title('Courbe ROC')\nplt.tight_layout()\nplt.show()",
        },
        {
            "name": "precision-recall curve",
            "code": "from sklearn.metrics import PrecisionRecallDisplay\nimport matplotlib.pyplot as plt\n\nPrecisionRecallDisplay.from_estimator(model, X_test, y_test)\nplt.title('Precision-Recall')\nplt.tight_layout()\nplt.show()",
        },
        {
            "name": "matrice de confusion",
            "code": "from sklearn.metrics import ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\nConfusionMatrixDisplay.from_estimator(model, X_test, y_test)\nplt.title('Matrice de confusion')\nplt.tight_layout()\nplt.show()",
        },
        {
            "name": "feature importance",
            "code": "import pandas as pd\nimport matplotlib.pyplot as plt\n\nimp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)\nimp.iloc[::-1].plot(kind='barh', figsize=(8,6))\nplt.title('Top feature importances')\nplt.tight_layout()\nplt.show()",
        },
        {
            "name": "learning curve",
            "code": "from sklearn.model_selection import learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n\ntrain_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='f1')\nplt.plot(train_sizes, np.mean(train_scores, axis=1), label='train')\nplt.plot(train_sizes, np.mean(val_scores, axis=1), label='validation')\nplt.legend()\nplt.title('Learning curve')\nplt.tight_layout()\nplt.show()",
        },
        {
            "name": "frontiere de decision 2D",
            "code": "import numpy as np\nimport matplotlib.pyplot as plt\n\nxx, yy = np.meshgrid(\n    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 250),\n    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 250),\n)\nZ = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)\nplt.contourf(xx, yy, Z, alpha=0.25)\nplt.scatter(X[:,0], X[:,1], c=y, s=18, edgecolor='k')\nplt.title('Decision boundary')\nplt.tight_layout()\nplt.show()",
        },
    ]

    frames = [
        "Montre un exemple complet pour tracer {name}.",
        "Comment visualiser {name} en Python ?",
        "Donne le code pour {name} avec interpretation.",
    ]
    business_angles = ["fraude", "churn", "credit", "maintenance", "recommandation", "NLP"]
    interpretation_focus = [
        "choix du seuil",
        "diagnostic faux positifs",
        "diagnostic faux negatifs",
        "stabilite du modele",
        "tradeoff precision/recall",
        "impact metier",
    ]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        v = viz[i % len(viz)]
        instruction = (
            f"{frames[i % len(frames)].format(name=v['name'])} "
            f"Cas metier: {business_angles[i % len(business_angles)]}; focus: {interpretation_focus[i % len(interpretation_focus)]}."
        )
        response = (
            f"Visualisation: {v['name']}\n"
            "Code Python executable:\n"
            "```python\n"
            f"{v['code']}\n"
            "```\n"
            "Interpretation rapide:\n"
            "- Lire la forme globale de la courbe/du graphe.\n"
            "- Detecter zones de faiblesse (faux positifs/faux negatifs).\n"
            f"- Ajuster seuil, features ou modele selon le signal observe dans {business_angles[(i + 1) % len(business_angles)]}.\n"
            f"Angle d'analyse: {interpretation_focus[(i + 2) % len(interpretation_focus)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux un graphe reel et du code. ||| Assistant: code matplotlib/sklearn pret a executer.",
                "source": "ml_visualizations",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_applications_real_rows(target_rows: int) -> list[dict[str, str]]:
    apps = [
        {
            "name": "fraude bancaire",
            "goal": "detecter transactions suspectes en quasi temps reel",
            "data": "transactions, device, geoloc, historique compte",
            "models": "xgboost + regles metier",
            "kpi": "PR-AUC, recall fraude, faux positifs par 1000",
        },
        {
            "name": "churn telecom",
            "goal": "predire risque de depart a 30 jours",
            "data": "usage, incidents, facturation, interactions support",
            "models": "logistic regression, random forest, calibrated xgboost",
            "kpi": "uplift retention, recall top-decile, cout action",
        },
        {
            "name": "scoring client",
            "goal": "estimer risque credit et potentiel commercial",
            "data": "profil, historique paiement, comportement digital",
            "models": "gbdt + modeles interpretable de controle",
            "kpi": "AUC, calibration, pertes attendues",
        },
        {
            "name": "prevision de ventes",
            "goal": "predire volumes par produit et magasin",
            "data": "historique ventes, promo, saisonnalite, stock",
            "models": "lightgbm, prophet, modeles hybrides",
            "kpi": "WAPE/MAE, rupture evitee, surstock evite",
        },
        {
            "name": "maintenance predictive",
            "goal": "anticiper pannes equipements critiques",
            "data": "capteurs IoT, logs machine, maintenance historique",
            "models": "anomaly detection + classification panne",
            "kpi": "downtime evite, recall incidents critiques",
        },
        {
            "name": "NLP support client",
            "goal": "classifier tickets et suggerer reponses",
            "data": "texte ticket, historique resolution, tags",
            "models": "embeddings + classifieur transformer",
            "kpi": "F1 macro, temps moyen resolution, CSAT",
        },
        {
            "name": "vision qualite industrielle",
            "goal": "detecter defauts produit sur ligne",
            "data": "images camera ligne, labels defauts",
            "models": "CNN/ViT + detecteur objet",
            "kpi": "mAP, taux de defaut manque, latence inference",
        },
    ]

    constraints = [
        "latence < 120 ms",
        "budget cloud limite",
        "fort desequilibre de classes",
        "donnees bruitees",
        "explicabilite obligatoire",
        "derive hebdomadaire",
        "deploiement multi-pays",
    ]
    errors = [
        "seuil non adapte au cout metier",
        "fuite de donnees temporelles",
        "absence de calibration",
        "monitoring incomplet par segment",
        "pipeline train/inference incoherent",
        "absence de rollback",
    ]
    deploy = ["batch quotidien", "temps reel", "micro-batch 15 min", "canary", "champion-challenger"]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        app = apps[i % len(apps)]
        instruction = (
            f"Donne un plan ML complet pour {app['name']} en production. "
            f"Contrainte: {constraints[i % len(constraints)]}."
        )
        response = (
            f"Use case: {app['name']}\n"
            f"Objectif: {app['goal']}\n"
            f"Donnees utiles: {app['data']}\n"
            f"Modeles candidats: {app['models']}\n"
            "Pipeline de bout en bout:\n"
            "1) Contrat de donnees (schema/fraicheur/completude).\n"
            "2) Feature engineering robuste + split adapte.\n"
            "3) Baseline puis modele principal + calibration.\n"
            "4) Seuil metier et simulation de cout des erreurs.\n"
            "5) Deploiement progressif, monitoring, rollback.\n"
            f"KPI cles: {app['kpi']}\n"
            f"Erreur frequente: {errors[i % len(errors)]}.\n"
            f"Mode de deploiement conseille: {deploy[i % len(deploy)]}."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux un cas reel exploitable. ||| Assistant: plan actionnable complet avec contraintes production.",
                "source": "ml_applications_real",
            }
        )
        i += 1

    return rows[:target_rows]


def build_ml_business_applied_rows(target_rows: int) -> list[dict[str, str]]:
    use_cases = [
        {
            "name": "churn",
            "objective": "predire la probabilité de depart client a J+30",
            "target": "client_resilie_30j",
            "kpi": "recall@precision_cible, uplift retention, cout actions",
            "actions": "prioriser les clients a risque pour offers de retention",
        },
        {
            "name": "scoring client",
            "objective": "estimer le risque de defaut et le potentiel de valeur",
            "target": "defaut_12m ou score appétence",
            "kpi": "AUC, calibration, taux d'acceptation, pertes attendues",
            "actions": "ajuster les seuils selon appetit au risque",
        },
        {
            "name": "fraude",
            "objective": "detecter transactions suspectes en quasi temps reel",
            "target": "transaction_frauduleuse",
            "kpi": "PR-AUC, recall, faux positifs par 1k transactions",
            "actions": "bloquer, verifier manuellement ou demander authentification forte",
        },
        {
            "name": "prevision de ventes",
            "objective": "prevoir volumes de ventes par produit/site",
            "target": "ventes_journalieres",
            "kpi": "WAPE/MAE, biais par segment, rupture de stock evitee",
            "actions": "ajuster approvisionnement et planning commercial",
        },
        {
            "name": "maintenance predictive",
            "objective": "anticiper pannes equipements avant arret",
            "target": "panne_7j",
            "kpi": "recall incident critique, downtime evite, cout intervention",
            "actions": "planifier maintenance proactive et stock de pieces",
        },
    ]

    constraints = [
        "latence inference < 120 ms",
        "budget cloud limite",
        "drift donnees hebdomadaire",
        "fort desequilibre de classes",
        "qualite labels variable",
        "exigence d'explicabilite metier",
    ]
    error_patterns = [
        "fuite temporelle dans les features",
        "seuil unique non aligne aux segments",
        "mauvaise gestion des valeurs manquantes",
        "absence de recalibrage des probabilites",
        "monitoring incomplet sur les populations rares",
        "pipeline train/inference incoherent",
    ]
    tradeoffs = [
        "precision vs recall",
        "latence vs complexite modele",
        "cout intervention vs risque residuel",
        "performance globale vs robustesse segment rare",
        "explicabilite vs performance brute",
    ]
    deployment_modes = ["batch quotidien", "temps reel", "micro-batch 15 min", "champion-challenger", "canary"]

    rows: list[dict[str, str]] = []
    i = 0
    while len(rows) < target_rows:
        case = use_cases[i % len(use_cases)]
        constraint = constraints[i % len(constraints)]
        error = error_patterns[i % len(error_patterns)]
        tradeoff = tradeoffs[i % len(tradeoffs)]
        mode = deployment_modes[i % len(deployment_modes)]

        instruction = (
            f"Je construis un cas {case['name']} en production. "
            f"Donne un plan actionnable de bout en bout avec controle des risques. "
            f"Contrainte cle: {constraint}."
        )
        response = (
            f"Use case: {case['name']}\n"
            f"Objectif metier: {case['objective']}\n"
            f"Target: {case['target']}\n"
            "Pipeline recommande:\n"
            "1) Contrat de donnees (schema, fraicheur, completude).\n"
            "2) Features metier + anti-leakage + split temporel.\n"
            "3) Baseline simple puis modele principal (XGBoost/RandomForest selon contrainte).\n"
            "4) Calibration + seuil par segment + simulation de cout.\n"
            "5) Deploiement + monitoring + rollback.\n"
            f"KPI metier: {case['kpi']}\n"
            f"Action operationnelle: {case['actions']}\n"
            f"Erreur frequente a eviter: {error}.\n"
            f"Arbitrage principal: {tradeoff}.\n"
            f"Mode de run conseille: {mode}.\n"
            "Controle hebdo: drift features, taux d'alertes, matrice de confusion par segment, cout FP/FN, stabilite des seuils."
        )
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: je veux un cas business realiste. ||| Assistant: plan ML applique avec contraintes prod.",
                "source": "ml_business_applied",
            }
        )
        i += 1

    return rows[:target_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML expert dataset modules for Elibot")
    parser.add_argument("--rows-concepts", type=int, default=1500)
    parser.add_argument("--rows-algorithms", type=int, default=2200)
    parser.add_argument("--rows-learning-types", type=int, default=800)
    parser.add_argument("--rows-implementation", type=int, default=1000)
    parser.add_argument("--rows-visualizations", type=int, default=500)
    parser.add_argument("--rows-business-applied", type=int, default=500)
    parser.add_argument("--rows-applications-real", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-concepts", default="data/processed/chatbot_train_fr_ml_concepts_deep.csv")
    parser.add_argument("--out-algorithms", default="data/processed/chatbot_train_fr_ml_algorithms_deep.csv")
    parser.add_argument("--out-learning-types", default="data/processed/chatbot_train_fr_ml_learning_types.csv")
    parser.add_argument("--out-implementation", default="data/processed/chatbot_train_fr_ml_implementation.csv")
    parser.add_argument("--out-visualizations", default="data/processed/chatbot_train_fr_ml_visualizations.csv")
    parser.add_argument("--out-business-applied", default="data/processed/chatbot_train_fr_ml_business_applied.csv")
    parser.add_argument("--out-applications-real", default="data/processed/chatbot_train_fr_ml_applications_real.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    concepts = build_ml_concepts_deep_rows(max(1, args.rows_concepts))
    algorithms = build_ml_algorithms_deep_rows(max(1, args.rows_algorithms))
    learning_types = build_ml_learning_types_rows(max(1, args.rows_learning_types))
    implementation = build_ml_implementation_rows(max(1, args.rows_implementation))
    visualizations = build_ml_visualizations_rows(max(1, args.rows_visualizations))
    business_applied = build_ml_business_applied_rows(max(1, args.rows_business_applied))
    applications_real = build_ml_applications_real_rows(max(1, args.rows_applications_real))

    out_concepts = Path(args.out_concepts)
    out_algorithms = Path(args.out_algorithms)
    out_learning_types = Path(args.out_learning_types)
    out_implementation = Path(args.out_implementation)
    out_visualizations = Path(args.out_visualizations)
    out_business_applied = Path(args.out_business_applied)
    out_applications_real = Path(args.out_applications_real)

    write_rows(out_concepts, concepts)
    write_rows(out_algorithms, algorithms)
    write_rows(out_learning_types, learning_types)
    write_rows(out_implementation, implementation)
    write_rows(out_visualizations, visualizations)
    write_rows(out_business_applied, business_applied)
    write_rows(out_applications_real, applications_real)

    print(
        {
            "concepts_rows": len(concepts),
            "algorithms_rows": len(algorithms),
            "learning_types_rows": len(learning_types),
            "implementation_rows": len(implementation),
            "visualizations_rows": len(visualizations),
            "business_applied_rows": len(business_applied),
            "applications_real_rows": len(applications_real),
            "out_concepts": str(out_concepts).replace("\\", "/"),
            "out_algorithms": str(out_algorithms).replace("\\", "/"),
            "out_learning_types": str(out_learning_types).replace("\\", "/"),
            "out_implementation": str(out_implementation).replace("\\", "/"),
            "out_visualizations": str(out_visualizations).replace("\\", "/"),
            "out_business_applied": str(out_business_applied).replace("\\", "/"),
            "out_applications_real": str(out_applications_real).replace("\\", "/"),
        }
    )


if __name__ == "__main__":
    main()
