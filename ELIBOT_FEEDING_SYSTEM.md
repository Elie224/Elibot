# Elibot Feeding System

This document defines how Elibot is fed, updated, and improved over time.

## 1) Four feeding layers

1. Foundation model
- Role: language understanding, reasoning prior, general fluency.
- Strategy: keep as base, do not directly overwrite behavior policy here.

2. Main fine-tuning dataset
- Role: global style, answer structure, baseline behavior consistency.
- Current source: large supervised dataset (phase 1).

3. Signature dataset (premium)
- Role: high-precision domain behavior (Data/AI/Automation), strict tone, internal rules.
- Current source: curated domain signature set (phase 2).

4. External memory (RAG + stores)
- Role: long-term continuity, user/project memory, document retrieval, operational context.
- Current source: logs, summaries, task memory, knowledge docs.

## 2) Continuous learning pipeline (implemented)

Script: `continuous_learning_pipeline_fr.py`

What it does:
- reads chat logs and audit logs
- filters low-quality and out-of-domain events
- applies quality thresholds (native score if present, heuristic fallback)
- deduplicates prompt/response pairs
- writes weekly feedback dataset
- merges signature + feedback into weekly training bundle
- writes a report with filtering stats and update hints

Run (weekly):

```powershell
.\.venv\Scripts\python.exe continuous_learning_pipeline_fr.py \
  --chat-log data/logs/elibot_chat_events.jsonl \
  --audit-log data/logs/elibot_audit.jsonl \
  --signature-dataset data/processed/chatbot_train_fr_signature_v2_domain.csv \
  --out-feedback data/processed/chatbot_train_fr_feedback_weekly.csv \
  --out-bundle data/processed/chatbot_train_fr_weekly_bundle.csv \
  --out-report reports/feeding_pipeline_report.json
```

## 3) Smart selection policy

Keep:
- in-domain responses
- non-degenerate outputs
- sufficiently long/structured responses
- quality score >= threshold

Reject:
- empty/short outputs
- out-of-domain turns
- low-quality flagged generations
- duplicates

Default policy (editable in CLI):
- min response chars: 40
- min quality score: 0.65

## 4) Update cadence

Weekly:
- run `continuous_learning_pipeline_fr.py`
- train lightweight LoRA on weekly bundle
- run fast evaluation set

Monthly:
- refresh signature dataset from best traces
- tighten quality threshold if drift appears

Quarterly:
- add new modules/workflows
- extend internal metrics and evaluation scenarios

## 5) Memory and coherence growth

Short-term:
- active session history + dynamic summary

Mid-term:
- task memory and audit traces

Long-term:
- chat logs + internal metrics + curated weekly dataset

## 6) Operational outputs

Generated files:
- `data/processed/chatbot_train_fr_feedback_weekly.csv`
- `data/processed/chatbot_train_fr_weekly_bundle.csv`
- `reports/feeding_pipeline_report.json`

These outputs are directly usable for continuous fine-tuning and governance review.
