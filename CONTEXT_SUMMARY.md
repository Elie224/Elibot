# Dynamic Context Summary

Elibot now keeps a dynamic summary per session to improve long-conversation coherence.

## Behavior

- Summary is refreshed every `SUMMARY_INTERVAL` user turns (default: 4).
- Summary is injected into prompt context before generation.
- Summary is also logged in chat events.

## Environment variables

- `SUMMARY_INTERVAL` (default `4`)
- `SUMMARY_MAX_CHARS` (default `1200`)

## Inspect a session summary

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/session/<session_id>/summary
```

## Notes

- This module is extractive and lightweight (no extra model).
- It complements memory/profile and knowledge retrieval.
