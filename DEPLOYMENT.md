# Deploiement Elibot

## 1) Demarrage local (Windows PowerShell)

Installe les dependances de deploiement:

```powershell
& .\.venv_gpu\Scripts\python.exe -m pip install -r .\requirements-deploy.txt
```

Lance l'API:

```powershell
& .\.venv_gpu\Scripts\python.exe -m uvicorn api_server_fr:app --host 0.0.0.0 --port 8000
```

Test rapide:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

## 2) Appels API

### Envoyer un message

```powershell
$body = @{ message = "Bonjour, je m'appelle Sarah." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body $body
```

Le retour contient un `session_id`. Reutilise-le pour garder la memoire:

```powershell
$body = @{ message = "Et ma ville ?"; session_id = "<SESSION_ID>" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body $body
```

### Reset session

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/reset/<SESSION_ID>
```

## 3) Deploiement Docker

Build:

```powershell
docker build -t elibot-api:latest .
```

Run:

```powershell
docker run --rm -p 8000:8000 elibot-api:latest
```

## 4) Variables d'environnement utiles

- `MODEL_DIR` (par defaut `models/chatbot-fr-flan-t5-small-v2-convfix`)
- `HISTORY_MODE` (`user-only` ou `full`)
- `HISTORY_TURNS` (par defaut `4`)
- `MAX_INPUT_LENGTH` (par defaut `512`)
- `MAX_NEW_TOKENS` (par defaut `72`)
- `TEMPERATURE` (par defaut `0`)
- `NO_REPEAT_NGRAM` (par defaut `3`)

Exemple:

```powershell
$env:MODEL_DIR = "models/chatbot-fr-flan-t5-small-v2-convfix"
$env:TEMPERATURE = "0"
& .\.venv_gpu\Scripts\python.exe -m uvicorn api_server_fr:app --host 0.0.0.0 --port 8000
```
