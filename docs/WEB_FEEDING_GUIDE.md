# Web Feeding Guide (Safe RAG)

## Objective
This project should not continuously retrain directly on random internet pages.
Use controlled web ingestion + retrieval (RAG) instead.

## Added Components
- scripts/ingest_web_knowledge.py
- scripts/query_web_knowledge.py
- data/web/allowed_domains.txt
- data/web/sources_allowlist.txt

## Run Ingestion
```powershell
python scripts/ingest_web_knowledge.py \
  --sources-file data/web/sources_allowlist.txt \
  --allow-domains-file data/web/allowed_domains.txt \
  --out-jsonl data/web/web_chunks.jsonl \
  --manifest-json reports/web_ingest_manifest.json
```

## Query Retrieved Knowledge
```powershell
python scripts/query_web_knowledge.py \
  --index-jsonl data/web/web_chunks.jsonl \
  --query "difference between precision and recall" \
  --top-k 5 \
  --out-json reports/web_retrieval_latest.json
```

## Consequences Without Guardrails
- Quality degradation from noisy and contradictory pages.
- Prompt injection risk from malicious page content.
- Legal/licensing exposure if source rights are unclear.
- Privacy risk if user data is sent to external endpoints.
- Operational instability (latency/cost/availability variability).

## Minimal Safety Rules
- Keep strict domain allowlist.
- Never ingest authenticated or private pages.
- Log every ingestion run in manifest JSON.
- Review sources before promoting to production.
- Use retrieval context with citations instead of blind model retraining.
