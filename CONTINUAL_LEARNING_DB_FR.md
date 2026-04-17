# Continual Learning DB (FR)

Ce module ajoute un apprentissage continu controle avec MongoDB.

## Fichiers principaux

- `continuous_learning_db_pipeline_fr.py`
- `mongo_migrations_fr.py`

## Base cible (imposee)

- Database: `Elibot`
- Collection: `mon_chatbot`

## Schema documents MongoDB

### Collection `mon_chatbot` (1 document = 1 conversation)

- `_id` (string)
- `user_id` (TEXT)
- `started_at` (TEXT ISO)
- `ended_at` (TEXT ISO)
- `topic` (TEXT)
- `tags` (array)
- `quality_score` (REAL)
- `status` (TEXT: `raw`, `filtered`, `selected`, `trained`)
- `messages` (array)

### Objet `messages[]`

- `id` (TEXT, PK)
- `role` (TEXT: `user`, `assistant`)
- `content` (TEXT)
- `created_at` (TEXT ISO)
- `is_technical` (INTEGER 0/1)
- `is_in_domain` (INTEGER 0/1)
- `has_error` (INTEGER 0/1)
- `quality_score` (REAL 0..1)
- `intent` (TEXT)
- `structure_type` (TEXT: `plain`, `json`, `steps`, `code`)

### Collection `_migrations`

- `_id` (migration id)
- `description`
- `applied_at`
- `result`

## Pipeline controle

1. Ingestion brute JSONL vers DB (`raw`)
2. Scoring + filtrage automatique (`filtered`)
3. Selection des meilleures conversations (`selected`)
4. Export datasets pour fine-tuning

## Commandes

### Migrations MongoDB (obligatoire)

```powershell
.\.venv_gpu\Scripts\python.exe mongo_migrations_fr.py --db-name Elibot --collection-name mon_chatbot
```

### Pipeline complet

```powershell
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode run-all --db-name Elibot --collection-name mon_chatbot
```

### Validation rapide (exemple)

```powershell
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode run-all --db-name Elibot --collection-name mon_chatbot --max-events 1000 --selection-limit 500
```

### Etapes separees

```powershell
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode migrate --db-name Elibot --collection-name mon_chatbot
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode ingest --db-name Elibot --collection-name mon_chatbot
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode score --db-name Elibot --collection-name mon_chatbot
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode select --db-name Elibot --collection-name mon_chatbot
.\.venv_gpu\Scripts\python.exe continuous_learning_db_pipeline_fr.py --mode export --db-name Elibot --collection-name mon_chatbot
```

## Sorties

Le script exporte dans `data/processed/continual_db/`:

- `core_qa.jsonl`
- `agent_actions.jsonl`
- `context_memory.jsonl`
- `style_signature.jsonl`
- `chatbot_train_fr_continual_selected.csv`

## Notes importantes

- Elibot n'apprend pas automatiquement depuis la DB brute.
- Le fine-tuning doit utiliser seulement les exports filtres/selectes.
- Regle recommandee: mini cycles LoRA avec jeux propres et limites.
