import os
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock
from threading import Thread

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import automation_engine_fr as automation
import advanced_modules_fr as advanced
import api_key_store_fr as key_store
import chat_with_model_fr as runtime
import context_summarizer_fr as context_summarizer
import external_integrations_fr as external_integrations
import intent_classifier_fr as intent_classifier
from knowledge_retrieval_fr import KnowledgeBase, format_knowledge_context
import response_verifier_fr as verifier


DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "models/chatbot-fr-flan-t5-small-v2-convfix")
DEFAULT_HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
DEFAULT_MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "72"))
DEFAULT_AUTOMATION_MAX_NEW_TOKENS = int(os.getenv("AUTOMATION_MAX_NEW_TOKENS", "220"))
DEFAULT_SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "4"))
DEFAULT_SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "1200"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
DEFAULT_NO_REPEAT_NGRAM = int(os.getenv("NO_REPEAT_NGRAM", "3"))
DEFAULT_HISTORY_MODE = os.getenv("HISTORY_MODE", "user-only")
DEFAULT_KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "3"))
DEFAULT_CHAT_LOG_PATH = os.getenv("CHAT_LOG_PATH", "data/logs/elibot_chat_events.jsonl")
DEFAULT_REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_API_KEYS = os.getenv(
    "API_KEYS",
    "elibot-admin-key:admin,elibot-advanced-key:advanced,elibot-basic-key:basic",
)

DEFAULT_RATE_LIMIT_BASIC = int(os.getenv("RATE_LIMIT_BASIC_PER_MIN", "30"))
DEFAULT_RATE_LIMIT_ADVANCED = int(os.getenv("RATE_LIMIT_ADVANCED_PER_MIN", "120"))
DEFAULT_RATE_LIMIT_ADMIN = int(os.getenv("RATE_LIMIT_ADMIN_PER_MIN", "300"))

DEFAULT_DAILY_QUOTA_BASIC = int(os.getenv("DAILY_QUOTA_BASIC", "800"))
DEFAULT_DAILY_QUOTA_ADVANCED = int(os.getenv("DAILY_QUOTA_ADVANCED", "5000"))
DEFAULT_DAILY_QUOTA_ADMIN = int(os.getenv("DAILY_QUOTA_ADMIN", "20000"))
DEFAULT_INTEGRATION_QUEUE_MAX_SIZE = int(os.getenv("INTEGRATION_QUEUE_MAX_SIZE", "1000"))
DEFAULT_INTEGRATION_MAX_PENDING_PER_PRINCIPAL = int(os.getenv("INTEGRATION_MAX_PENDING_PER_PRINCIPAL", "50"))
DEFAULT_INTEGRATION_JOB_TTL_SECONDS = int(os.getenv("INTEGRATION_JOB_TTL_SECONDS", "172800"))
DEFAULT_INTEGRATION_JOBS_PATH = Path(os.getenv("INTEGRATION_JOBS_PATH", "data/automation/integration_jobs.json"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Tu es Elibot, un assistant specialise en analyse de donnees, IA appliquee et automatisation. "
    "Tu reponds de facon claire et professionnelle. "
    "Tu refuses poliment les sujets hors domaine et rediriges vers une demande technique.",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    profile: dict[str, str]


class AutomationPlanRequest(BaseModel):
    goal: str = Field(..., min_length=4)


class AutomationPlanResponse(BaseModel):
    plan: dict
    source: str


class AutomationRunRequest(BaseModel):
    goal: str | None = None
    plan: dict | None = None
    dry_run: bool = True


class AutomationRunResponse(BaseModel):
    plan: dict
    source: str
    execution: dict


class IntentClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1)


class IntentClassifyResponse(BaseModel):
    intent: str
    confidence: float
    reasons: list[str]


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(default="simple")


class SimulateRequest(BaseModel):
    plan: dict


class ToolSuggestRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ApiKeyCreateRequest(BaseModel):
    role: str = Field(default="basic")
    label: str = Field(default="")


class ApiKeyRevokeRequest(BaseModel):
    key_id: str = Field(..., min_length=8)


class IntegrationExecuteRequest(BaseModel):
    provider: str = Field(..., min_length=2)
    action: str = Field(..., min_length=2)
    payload: dict = Field(default_factory=dict)
    dry_run: bool = True


class IntegrationAsyncRequest(BaseModel):
    provider: str = Field(..., min_length=2)
    action: str = Field(..., min_length=2)
    payload: dict = Field(default_factory=dict)
    dry_run: bool = True


class IntegrationTemplateExecuteRequest(BaseModel):
    template_id: str = Field(..., min_length=3)
    variables: dict = Field(default_factory=dict)
    dry_run: bool = True


class IntegrationBatchItem(BaseModel):
    provider: str = Field(..., min_length=2)
    action: str = Field(..., min_length=2)
    payload: dict = Field(default_factory=dict)


class IntegrationAutomationRunRequest(BaseModel):
    items: list[IntegrationBatchItem] = Field(default_factory=list)
    dry_run: bool = True
    max_actions: int = Field(default=5, ge=1, le=20)
    stop_on_error: bool = True


@dataclass
class SessionState:
    history: list[tuple[str, str]] = field(default_factory=list)
    profile: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    user_turn_count: int = 0


app = FastAPI(title="Elibot API", version="1.0.0")

_state_lock = Lock()
_sessions: dict[str, SessionState] = {}
_log_lock = Lock()
_access_lock = Lock()

_rate_state: dict[str, dict] = {}
_quota_state: dict[str, dict] = {}
_integration_queue: Queue = Queue(maxsize=max(1, DEFAULT_INTEGRATION_QUEUE_MAX_SIZE))
_integration_jobs: dict[str, dict] = {}
_integration_jobs_lock = Lock()

_role_rank = {"basic": 0, "advanced": 1, "admin": 2}
_role_rate_limit = {
    "basic": DEFAULT_RATE_LIMIT_BASIC,
    "advanced": DEFAULT_RATE_LIMIT_ADVANCED,
    "admin": DEFAULT_RATE_LIMIT_ADMIN,
}
_role_daily_quota = {
    "basic": DEFAULT_DAILY_QUOTA_BASIC,
    "advanced": DEFAULT_DAILY_QUOTA_ADVANCED,
    "admin": DEFAULT_DAILY_QUOTA_ADMIN,
}


def _parse_iso_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return 0.0


def _save_integration_jobs_locked() -> None:
    DEFAULT_INTEGRATION_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": _integration_jobs,
    }
    DEFAULT_INTEGRATION_JOBS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _purge_expired_jobs_locked() -> int:
    now_ts = time.time()
    ttl = max(60, DEFAULT_INTEGRATION_JOB_TTL_SECONDS)
    removed = 0

    for job_id, job in list(_integration_jobs.items()):
        status = str(job.get("status") or "")
        if status in {"queued", "running"}:
            continue

        finished_ts = _parse_iso_ts(job.get("finished_at"))
        if not finished_ts:
            created_ts = _parse_iso_ts(job.get("created_at"))
            finished_ts = created_ts
        if finished_ts and (now_ts - finished_ts) > ttl:
            _integration_jobs.pop(job_id, None)
            removed += 1

    return removed


def _load_integration_jobs() -> None:
    if not DEFAULT_INTEGRATION_JOBS_PATH.exists():
        return
    try:
        raw = json.loads(DEFAULT_INTEGRATION_JOBS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    jobs = raw.get("jobs")
    if not isinstance(jobs, dict):
        return

    with _integration_jobs_lock:
        _integration_jobs.update(jobs)
        _purge_expired_jobs_locked()
        _save_integration_jobs_locked()


def _integration_worker_loop() -> None:
    while True:
        try:
            job_id, item = _integration_queue.get(timeout=1.0)
        except Empty:
            continue

        with _integration_jobs_lock:
            job = _integration_jobs.get(job_id)
            if job is None:
                _integration_queue.task_done()
                continue
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            _save_integration_jobs_locked()

        try:
            result = external_integrations.execute_integration(
                provider=item["provider"],
                action=item["action"],
                payload=item["payload"],
                dry_run=bool(item["dry_run"]),
            )
            with _integration_jobs_lock:
                job = _integration_jobs.get(job_id, {})
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
                _integration_jobs[job_id] = job
                _purge_expired_jobs_locked()
                _save_integration_jobs_locked()
        except Exception as exc:
            with _integration_jobs_lock:
                job = _integration_jobs.get(job_id, {})
                job["status"] = "error"
                job["error"] = str(exc)
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
                _integration_jobs[job_id] = job
                _purge_expired_jobs_locked()
                _save_integration_jobs_locked()
        finally:
            _integration_queue.task_done()


_load_integration_jobs()
_integration_worker = Thread(target=_integration_worker_loop, name="elibot-integration-worker", daemon=True)
_integration_worker.start()


def _parse_api_keys(raw: str) -> dict[str, str]:
    # Format: "key1:admin,key2:advanced,key3:basic"
    mapping: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, role = chunk.split(":", 1)
        key = key.strip()
        role = role.strip().lower()
        if key and role in _role_rank:
            mapping[key] = role
    return mapping


_api_key_role_map_env = _parse_api_keys(DEFAULT_API_KEYS)
_bootstrapped_count = key_store.bootstrap_from_env(_api_key_role_map_env)


def _enforce_rate_and_quota(principal: str, role: str) -> None:
    minute_bucket = int(time.time() // 60)
    day_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    per_min_limit = _role_rate_limit.get(role, DEFAULT_RATE_LIMIT_BASIC)
    per_day_quota = _role_daily_quota.get(role, DEFAULT_DAILY_QUOTA_BASIC)

    with _access_lock:
        rs = _rate_state.get(principal)
        if rs is None or rs.get("minute") != minute_bucket:
            rs = {"minute": minute_bucket, "count": 0}
            _rate_state[principal] = rs
        if rs["count"] >= per_min_limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        rs["count"] += 1

        qs = _quota_state.get(principal)
        if qs is None or qs.get("day") != day_bucket:
            qs = {"day": day_bucket, "count": 0}
            _quota_state[principal] = qs
        if qs["count"] >= per_day_quota:
            raise HTTPException(status_code=429, detail="daily quota exceeded")
        qs["count"] += 1


def _authorize(required_role: str, api_key: str | None, request: Request) -> dict:
    client_host = request.client.host if request.client else "unknown"

    role = "basic"
    principal = f"anonymous:{client_host}"

    if not DEFAULT_REQUIRE_API_KEY:
        if api_key:
            rec = key_store.verify_api_key(api_key)
            if rec is not None:
                role = rec["role"]
                principal = f"key:{str(rec.get('id', ''))[:8]}"
            else:
                role = _api_key_role_map_env.get(api_key, "basic")
                principal = f"env:{api_key[:6]}"
        _enforce_rate_and_quota(principal, role)
        return {"principal": principal, "role": role}

    if not api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")

    rec = key_store.verify_api_key(api_key)
    if rec is not None:
        role = rec["role"]
        principal = f"key:{str(rec.get('id', ''))[:8]}"
    else:
        role = _api_key_role_map_env.get(api_key)
        if role is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        principal = f"env:{api_key[:6]}"

    if _role_rank[role] < _role_rank[required_role]:
        raise HTTPException(status_code=403, detail="insufficient role")

    _enforce_rate_and_quota(principal, role)
    return {"principal": principal, "role": role}


def require_access(required_role: str):
    def _dep(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict:
        return _authorize(required_role, x_api_key, request)

    return _dep

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_DIR).to(_device)
_model.eval()
_kb = KnowledgeBase()


def _append_chat_event(event: dict) -> None:
    path = Path(DEFAULT_CHAT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

CHAT_UI_HTML = """
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Elibot Chat</title>
    <style>
        :root {
            --bg: #f5f7fb;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #166534;
            --accent-2: #0f766e;
            --border: #e5e7eb;
        }
        body {
            margin: 0;
            background: radial-gradient(circle at top left, #dff4ec, var(--bg));
            color: var(--text);
            font-family: Segoe UI, Tahoma, sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: stretch;
        }
        .app {
            width: min(900px, 100%);
            display: grid;
            grid-template-rows: auto 1fr auto;
            gap: 10px;
            padding: 16px;
            box-sizing: border-box;
        }
        .header {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .title {
            font-size: 18px;
            font-weight: 700;
            color: var(--accent);
        }
        .session {
            color: var(--muted);
            font-size: 13px;
        }
        .auth {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .auth input {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 8px;
            font-size: 12px;
            width: 180px;
        }
        .chat {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            overflow: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 50vh;
        }
        .msg {
            max-width: 85%;
            padding: 10px 12px;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.35;
        }
        .user {
            align-self: flex-end;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
        }
        .bot {
            align-self: flex-start;
            background: #ecfeff;
            border: 1px solid #bae6fd;
        }
        .composer {
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 8px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px;
        }
        textarea {
            resize: none;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            font: inherit;
            min-height: 44px;
            max-height: 120px;
        }
        button {
            border: none;
            border-radius: 10px;
            padding: 0 14px;
            font: inherit;
            cursor: pointer;
            color: white;
            background: var(--accent);
        }
        .secondary {
            background: var(--accent-2);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <div class="title">Elibot</div>
            <div class="auth">
                <input id="apiKey" placeholder="X-API-Key" />
                <div class="session" id="session">Session: -</div>
            </div>
        </div>
        <div class="chat" id="chat"></div>
        <div class="composer">
            <textarea id="input" placeholder="Ecris ton message..."></textarea>
            <button id="send">Envoyer</button>
            <button class="secondary" id="reset">Reset</button>
        </div>
    </div>
    <script>
        let sessionId = null;
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const resetBtn = document.getElementById('reset');
        const sessionEl = document.getElementById('session');
        const apiKeyInput = document.getElementById('apiKey');

        function authHeaders() {
            const headers = { 'Content-Type': 'application/json' };
            const key = (apiKeyInput.value || '').trim();
            if (key) headers['X-API-Key'] = key;
            return headers;
        }

        function addMessage(text, cls) {
            const el = document.createElement('div');
            el.className = `msg ${cls}`;
            el.textContent = text;
            chat.appendChild(el);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            input.value = '';
            addMessage(message, 'user');
            sendBtn.disabled = true;

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: authHeaders(),
                    body: JSON.stringify({ message, session_id: sessionId }),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erreur API');
                sessionId = data.session_id;
                sessionEl.textContent = `Session: ${sessionId}`;
                addMessage(data.response, 'bot');
            } catch (e) {
                addMessage(`Erreur: ${e.message}`, 'bot');
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }

        async function resetSession() {
            if (!sessionId) {
                chat.innerHTML = '';
                return;
            }
            await fetch(`/reset/${sessionId}`, { method: 'POST', headers: authHeaders() });
            sessionId = null;
            sessionEl.textContent = 'Session: -';
            chat.innerHTML = '';
            addMessage('Session reinitialisee.', 'bot');
        }

        sendBtn.addEventListener('click', sendMessage);
        resetBtn.addEventListener('click', resetSession);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        addMessage("Bonjour, je suis Elibot. Je suis specialise data, IA et automatisation. Pose-moi une question technique.", 'bot');
        input.focus();
    </script>
</body>
</html>
"""


def _get_or_create_session(session_id: str | None) -> tuple[str, SessionState]:
    with _state_lock:
        sid = session_id or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = SessionState()
        return sid, _sessions[sid]


@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    return CHAT_UI_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": _device, "model_dir": DEFAULT_MODEL_DIR}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _ctx: dict = Depends(require_access("basic"))) -> ChatResponse:
    sid, state = _get_or_create_session(request.session_id)
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    user_prefs = advanced.update_user_preferences(sid, user_text)
    if user_prefs:
        for key, value in user_prefs.items():
            if key not in {"updated_at"}:
                state.profile[key] = str(value)

    runtime.update_profile_from_user_text(user_text, state.profile)
    in_domain = runtime.is_in_domain_query(user_text)
    intent = intent_classifier.classify_intent(user_text)
    skill = advanced.detect_skill(user_text)
    tone = advanced.infer_tone(user_text, state.profile, intent)

    direct = runtime.maybe_rule_reply(user_text, state.profile)
    verifier_issues: list[str] = []
    corrected_by_verifier = False

    clarification = advanced.clarification_prompt(user_text)
    if not direct and clarification and intent.get("intent") in {"unknown", "technical_question"}:
        direct = clarification

    # Route explicit automation asks toward workflow planning guidance.
    if not direct and intent["intent"] == "automation_request":
        plan, source = automation.build_workflow_plan(
            goal=user_text,
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            max_input_length=DEFAULT_MAX_INPUT_LENGTH,
            max_new_tokens=DEFAULT_AUTOMATION_MAX_NEW_TOKENS,
        )
        first_steps = [s.get("action", "") for s in plan.get("steps", [])[:3]]
        direct = (
            "J'ai prepare un plan d'automatisation. "
            f"Source: {source}. "
            f"Etapes: {', '.join(first_steps)}. "
            "Utilise /automation/run pour l'executer (dry_run recommande)."
        )

    if direct:
        answer = direct
    else:
        summary_context = context_summarizer.format_summary_context(state.summary)
        knowledge_context = format_knowledge_context(_kb.search(user_text, top_k=DEFAULT_KNOWLEDGE_TOP_K))
        combined_context = "\n".join([x for x in [summary_context, knowledge_context] if x]).strip()

        prompt = runtime.build_prompt(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            history=state.history,
            user_text=user_text,
            history_turns=DEFAULT_HISTORY_TURNS,
            history_mode=DEFAULT_HISTORY_MODE,
            profile=state.profile,
            use_slot_memory=True,
            knowledge_context=combined_context,
        )

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_INPUT_LENGTH,
        ).to(_device)

        use_sampling = DEFAULT_TEMPERATURE > 0
        generate_kwargs = {
            "do_sample": use_sampling,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "no_repeat_ngram_size": max(0, DEFAULT_NO_REPEAT_NGRAM),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "num_beams": 1,
        }
        if use_sampling:
            generate_kwargs["temperature"] = DEFAULT_TEMPERATURE
            generate_kwargs["top_p"] = DEFAULT_TOP_P

        with torch.no_grad():
            output_ids = _model.generate(**inputs, **generate_kwargs)

        answer = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        answer = runtime.clean_generated_text(answer)
        verifier_issues = verifier.detect_quality_issues(user_text, answer, in_domain=in_domain)
        if runtime.is_low_quality_answer(answer) or verifier_issues:
            answer = runtime.fallback_reply(user_text)
            corrected_by_verifier = True

    answer = advanced.apply_tone(answer, tone)
    compliance = advanced.compliance_check(answer, in_domain=in_domain)
    if not compliance.get("ok", True):
        answer = runtime.fallback_reply(user_text)

    state.history.append(("Utilisateur", user_text))
    state.history.append(("Assistant", answer))
    state.user_turn_count += 1

    if state.user_turn_count % max(1, DEFAULT_SUMMARY_INTERVAL) == 0:
        state.summary = context_summarizer.build_dynamic_summary(
            history=state.history,
            profile=state.profile,
            previous_summary=state.summary,
            max_chars=DEFAULT_SUMMARY_MAX_CHARS,
        )

    _append_chat_event(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": sid,
            "user_text": user_text,
            "assistant_text": answer,
            "in_domain": in_domain,
            "intent": intent,
            "skill": skill,
            "tone": tone,
            "used_rule_reply": bool(direct),
            "is_low_quality": runtime.is_low_quality_answer(answer),
            "verifier_issues": verifier_issues,
            "corrected_by_verifier": corrected_by_verifier,
            "compliance": compliance,
            "summary": state.summary,
            "profile": dict(state.profile),
        }
    )

    advanced.update_performance_metrics(
        {
            "in_domain": in_domain,
            "intent": intent,
            "used_rule_reply": bool(direct),
            "corrected_by_verifier": corrected_by_verifier,
        }
    )

    advanced.write_audit(
        {
            "session_id": sid,
            "user_text": user_text,
            "assistant_text": answer,
            "intent": intent,
            "skill": skill,
            "tone": tone,
            "metacognition": advanced.build_metacognition(
                user_text=user_text,
                intent=intent,
                selected_sources=[x.get("source", "") for x in _kb.search(user_text, top_k=DEFAULT_KNOWLEDGE_TOP_K)],
            ),
        }
    )

    return ChatResponse(session_id=sid, response=answer, profile=state.profile)


@app.post("/reset/{session_id}")
def reset_session(session_id: str, _ctx: dict = Depends(require_access("basic"))) -> dict[str, str]:
    with _state_lock:
        if session_id in _sessions:
            del _sessions[session_id]
    return {"status": "reset", "session_id": session_id}


@app.get("/session/{session_id}/summary")
def session_summary(session_id: str, _ctx: dict = Depends(require_access("basic"))) -> dict:
    with _state_lock:
        state = _sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session_id": session_id,
            "summary": state.summary,
            "user_turn_count": state.user_turn_count,
            "history_items": len(state.history),
        }


@app.post("/automation/plan", response_model=AutomationPlanResponse)
def automation_plan(request: AutomationPlanRequest, _ctx: dict = Depends(require_access("advanced"))) -> AutomationPlanResponse:
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal cannot be empty")

    plan, source = automation.build_workflow_plan(
        goal=goal,
        model=_model,
        tokenizer=_tokenizer,
        device=_device,
        max_input_length=DEFAULT_MAX_INPUT_LENGTH,
        max_new_tokens=DEFAULT_AUTOMATION_MAX_NEW_TOKENS,
    )
    return AutomationPlanResponse(plan=plan, source=source)


@app.post("/automation/run", response_model=AutomationRunResponse)
def automation_run(request: AutomationRunRequest, _ctx: dict = Depends(require_access("advanced"))) -> AutomationRunResponse:
    source = "request"
    if request.plan is not None:
        plan = automation.sanitize_plan(request.plan)
    else:
        goal = (request.goal or "").strip()
        if not goal:
            raise HTTPException(status_code=400, detail="provide plan or goal")
        plan, source = automation.build_workflow_plan(
            goal=goal,
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            max_input_length=DEFAULT_MAX_INPUT_LENGTH,
            max_new_tokens=DEFAULT_AUTOMATION_MAX_NEW_TOKENS,
        )

    execution = automation.execute_plan(plan=plan, dry_run=request.dry_run)
    return AutomationRunResponse(plan=plan, source=source, execution=execution)


@app.post("/intent/classify", response_model=IntentClassifyResponse)
def classify_intent(request: IntentClassifyRequest, _ctx: dict = Depends(require_access("basic"))) -> IntentClassifyResponse:
    result = intent_classifier.classify_intent(request.message.strip())
    return IntentClassifyResponse(
        intent=result["intent"],
        confidence=float(result["confidence"]),
        reasons=list(result["reasons"]),
    )


@app.post("/rewrite")
def rewrite_text(request: RewriteRequest, _ctx: dict = Depends(require_access("basic"))) -> dict:
    rewritten = advanced.rewrite_text(request.text, request.mode)
    return {"mode": request.mode, "original": request.text, "rewritten": rewritten}


@app.post("/simulate")
def simulate_plan(request: SimulateRequest, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    return advanced.simulate_workflow(request.plan)


@app.post("/tools/suggest")
def suggest_tools(request: ToolSuggestRequest, _ctx: dict = Depends(require_access("basic"))) -> dict:
    tools = advanced.tool_suggestions(request.message)
    return {"message": request.message, "tools": tools}


@app.get("/audit/recent")
def recent_audit(limit: int = 20, _ctx: dict = Depends(require_access("admin"))) -> dict:
    items = advanced.read_recent_audit(limit=max(1, min(limit, 200)))
    return {"count": len(items), "items": items}


@app.get("/metrics/performance")
def performance_metrics(_ctx: dict = Depends(require_access("admin"))) -> dict:
    return advanced.get_performance_metrics()


@app.get("/metrics/performance.csv", response_class=PlainTextResponse)
def performance_metrics_csv(_ctx: dict = Depends(require_access("admin"))) -> str:
        metrics = advanced.get_performance_metrics()
        by_intent = metrics.get("by_intent", {})

        lines = ["metric,value"]
        lines.append(f"total,{metrics.get('total', 0)}")
        lines.append(f"corrected_by_verifier,{metrics.get('corrected_by_verifier', 0)}")
        lines.append(f"rule_reply,{metrics.get('rule_reply', 0)}")
        lines.append(f"out_of_domain,{metrics.get('out_of_domain', 0)}")
        lines.append(f"updated_at,{metrics.get('updated_at', '')}")
        for intent_name, count in sorted(by_intent.items()):
                lines.append(f"intent_{intent_name},{count}")
        return "\n".join(lines)


@app.get("/dashboard/metrics", response_class=HTMLResponse)
def metrics_dashboard(_ctx: dict = Depends(require_access("admin"))) -> str:
        metrics = advanced.get_performance_metrics()
        by_intent = metrics.get("by_intent", {})

        rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(by_intent.items())
        )
        html = f"""
        <html>
            <head>
                <title>Elibot Metrics Dashboard</title>
                <style>
                    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; }}
                    h1 {{ margin-bottom: 8px; }}
                    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
                    th {{ background: #f3f4f6; }}
                </style>
            </head>
            <body>
                <h1>Elibot Metrics</h1>
                <div class=\"card\">
                    <div><strong>Total:</strong> {metrics.get('total', 0)}</div>
                    <div><strong>Corrected by verifier:</strong> {metrics.get('corrected_by_verifier', 0)}</div>
                    <div><strong>Rule replies:</strong> {metrics.get('rule_reply', 0)}</div>
                    <div><strong>Out of domain:</strong> {metrics.get('out_of_domain', 0)}</div>
                    <div><strong>Updated:</strong> {metrics.get('updated_at', '')}</div>
                </div>
                <div class=\"card\">
                    <h3>By intent</h3>
                    <table>
                        <thead><tr><th>Intent</th><th>Count</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                <div><a href=\"/metrics/performance.csv\">Download CSV</a></div>
            </body>
        </html>
        """
        return html


@app.get("/user/{session_id}/preferences")
def user_preferences(session_id: str, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    prefs = advanced.get_user_preferences(session_id)
    return {"session_id": session_id, "preferences": prefs}


@app.get("/integrations/providers")
def integrations_providers(_ctx: dict = Depends(require_access("basic"))) -> dict:
    return {
        "providers": external_integrations.provider_status(),
    }


@app.post("/integrations/execute")
def integrations_execute(request: IntegrationExecuteRequest, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    try:
        result = external_integrations.execute_integration(
            provider=request.provider,
            action=request.action,
            payload=request.payload,
            dry_run=bool(request.dry_run),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"integration error: {exc}")

    advanced.write_audit(
        {
            "event": "integration_execute",
            "provider": request.provider,
            "action": request.action,
            "dry_run": request.dry_run,
            "principal": _ctx.get("principal"),
            "role": _ctx.get("role"),
        }
    )
    return result


@app.post("/integrations/execute-async")
def integrations_execute_async(request: IntegrationAsyncRequest, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    principal = _ctx.get("principal", "unknown")

    with _integration_jobs_lock:
        _purge_expired_jobs_locked()
        principal_pending = sum(
            1
            for job in _integration_jobs.values()
            if job.get("principal") == principal and job.get("status") in {"queued", "running"}
        )
        if principal_pending >= max(1, DEFAULT_INTEGRATION_MAX_PENDING_PER_PRINCIPAL):
            raise HTTPException(status_code=429, detail="too many pending integration jobs for principal")

    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    job_payload = {
        "provider": request.provider,
        "action": request.action,
        "payload": request.payload,
        "dry_run": bool(request.dry_run),
    }

    with _integration_jobs_lock:
        _integration_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "principal": principal,
            "role": _ctx.get("role"),
            **job_payload,
        }
        _save_integration_jobs_locked()

    try:
        _integration_queue.put_nowait((job_id, job_payload))
    except Full:
        with _integration_jobs_lock:
            _integration_jobs.pop(job_id, None)
        raise HTTPException(status_code=429, detail="integration queue is full")

    advanced.write_audit(
        {
            "event": "integration_execute_async",
            "job_id": job_id,
            "provider": request.provider,
            "action": request.action,
            "dry_run": request.dry_run,
            "principal": principal,
            "role": _ctx.get("role"),
        }
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
    }


@app.get("/integrations/jobs/{job_id}")
def integration_job_status(job_id: str, _ctx: dict = Depends(require_access("basic"))) -> dict:
    with _integration_jobs_lock:
        _purge_expired_jobs_locked()
        job = _integration_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        principal = _ctx.get("principal")
        role = _ctx.get("role", "basic")
        if role != "admin" and job.get("principal") != principal:
            raise HTTPException(status_code=403, detail="job access denied")
        return dict(job)


@app.post("/integrations/jobs/purge")
def integration_jobs_purge(_ctx: dict = Depends(require_access("admin"))) -> dict:
    with _integration_jobs_lock:
        removed = _purge_expired_jobs_locked()
        _save_integration_jobs_locked()
        remaining = len(_integration_jobs)
    return {
        "status": "ok",
        "removed": removed,
        "remaining": remaining,
    }


@app.get("/integrations/metrics")
def integrations_metrics(_ctx: dict = Depends(require_access("admin"))) -> dict:
    with _integration_jobs_lock:
        _purge_expired_jobs_locked()
        _save_integration_jobs_locked()
        queued = sum(1 for job in _integration_jobs.values() if job.get("status") == "queued")
        running = sum(1 for job in _integration_jobs.values() if job.get("status") == "running")
        finished = sum(1 for job in _integration_jobs.values() if job.get("status") in {"done", "error"})

    return {
        "metrics": external_integrations.get_integration_metrics(),
        "provider_health": external_integrations.get_provider_health(),
        "queue": {
            "size": _integration_queue.qsize(),
            "max_size": DEFAULT_INTEGRATION_QUEUE_MAX_SIZE,
        },
        "jobs": {
            "total": queued + running + finished,
            "queued": queued,
            "running": running,
            "finished": finished,
            "ttl_seconds": DEFAULT_INTEGRATION_JOB_TTL_SECONDS,
            "storage_path": str(DEFAULT_INTEGRATION_JOBS_PATH),
        },
    }


@app.get("/dashboard/integrations", response_class=HTMLResponse)
def integrations_dashboard(_ctx: dict = Depends(require_access("admin"))) -> str:
    with _integration_jobs_lock:
        _purge_expired_jobs_locked()
        _save_integration_jobs_locked()
        queued_jobs = sum(1 for job in _integration_jobs.values() if job.get("status") == "queued")
        running_jobs = sum(1 for job in _integration_jobs.values() if job.get("status") == "running")
        finished_jobs = sum(1 for job in _integration_jobs.values() if job.get("status") in {"done", "error"})

    metrics = external_integrations.get_integration_metrics()
    health = external_integrations.get_provider_health()
    by_provider = metrics.get("by_provider", {})

    rows = []
    for provider in sorted(by_provider.keys()):
        m = by_provider.get(provider, {})
        h = health.get(provider, {})
        rows.append(
            "<tr>"
            f"<td>{provider}</td>"
            f"<td>{m.get('total', 0)}</td>"
            f"<td>{m.get('success', 0)}</td>"
            f"<td>{m.get('error', 0)}</td>"
            f"<td>{m.get('last_latency_ms', 0)}</td>"
            f"<td>{h.get('consecutive_failures', 0)}</td>"
            f"<td>{h.get('last_error', '')}</td>"
            "</tr>"
        )
    table_rows = "".join(rows)

    return f"""
    <html>
      <head>
        <title>Elibot Integrations Dashboard</title>
        <style>
          body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; }}
          .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
          th {{ background: #f3f4f6; }}
        </style>
      </head>
      <body>
        <h1>Elibot Integrations</h1>
        <div class="card">
          <div><strong>Total:</strong> {metrics.get('total', 0)}</div>
          <div><strong>Success:</strong> {metrics.get('success', 0)}</div>
          <div><strong>Error:</strong> {metrics.get('error', 0)}</div>
          <div><strong>Updated:</strong> {metrics.get('updated_at', '')}</div>
          <div><strong>Queue:</strong> {_integration_queue.qsize()} / {DEFAULT_INTEGRATION_QUEUE_MAX_SIZE}</div>
                    <div><strong>Jobs queued/running/finished:</strong> {queued_jobs} / {running_jobs} / {finished_jobs}</div>
                    <div><strong>Jobs TTL (s):</strong> {DEFAULT_INTEGRATION_JOB_TTL_SECONDS}</div>
        </div>
        <div class="card">
          <h3>By provider</h3>
          <table>
            <thead>
              <tr>
                <th>Provider</th><th>Total</th><th>Success</th><th>Error</th>
                <th>Last latency ms</th><th>Consecutive failures</th><th>Last error</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
      </body>
    </html>
    """


@app.get("/integrations/templates")
def integrations_templates(_ctx: dict = Depends(require_access("basic"))) -> dict:
    return {
        "templates": external_integrations.integration_templates(),
    }


@app.post("/integrations/execute-template")
def integrations_execute_template(request: IntegrationTemplateExecuteRequest, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    try:
        built = external_integrations.build_template_request(
            template_id=request.template_id,
            variables=request.variables,
        )
        result = external_integrations.execute_integration(
            provider=built["provider"],
            action=built["action"],
            payload=built["payload"],
            dry_run=bool(request.dry_run),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"integration template error: {exc}")

    advanced.write_audit(
        {
            "event": "integration_execute_template",
            "template_id": request.template_id,
            "dry_run": request.dry_run,
            "principal": _ctx.get("principal"),
            "role": _ctx.get("role"),
        }
    )
    return {
        "template": built,
        "execution": result,
    }


@app.post("/automation/run-integrations")
def automation_run_integrations(request: IntegrationAutomationRunRequest, _ctx: dict = Depends(require_access("advanced"))) -> dict:
    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    safe_items = request.items[: request.max_actions]
    results: list[dict] = []
    errors = 0

    for idx, item in enumerate(safe_items, start=1):
        try:
            run_result = external_integrations.execute_integration(
                provider=item.provider,
                action=item.action,
                payload=item.payload,
                dry_run=bool(request.dry_run),
            )
            results.append({
                "index": idx,
                "status": "ok",
                "provider": item.provider,
                "action": item.action,
                "result": run_result,
            })
        except Exception as exc:
            errors += 1
            results.append(
                {
                    "index": idx,
                    "status": "error",
                    "provider": item.provider,
                    "action": item.action,
                    "error": str(exc),
                }
            )
            if request.stop_on_error:
                break

    advanced.write_audit(
        {
            "event": "automation_run_integrations",
            "dry_run": request.dry_run,
            "requested_actions": len(request.items),
            "executed_actions": len(results),
            "errors": errors,
            "principal": _ctx.get("principal"),
            "role": _ctx.get("role"),
        }
    )

    return {
        "dry_run": request.dry_run,
        "requested_actions": len(request.items),
        "executed_actions": len(results),
        "max_actions": request.max_actions,
        "stop_on_error": request.stop_on_error,
        "errors": errors,
        "results": results,
    }


@app.get("/access/status")
def access_status(_ctx: dict = Depends(require_access("basic"))) -> dict:
    principal = _ctx.get("principal", "unknown")
    role = _ctx.get("role", "basic")
    minute_bucket = int(time.time() // 60)
    day_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with _access_lock:
        rs = _rate_state.get(principal, {"minute": minute_bucket, "count": 0})
        qs = _quota_state.get(principal, {"day": day_bucket, "count": 0})

    return {
        "principal": principal,
        "role": role,
        "rate_limit_per_min": _role_rate_limit.get(role, DEFAULT_RATE_LIMIT_BASIC),
        "rate_used_current_min": rs["count"] if rs.get("minute") == minute_bucket else 0,
        "daily_quota": _role_daily_quota.get(role, DEFAULT_DAILY_QUOTA_BASIC),
        "daily_used": qs["count"] if qs.get("day") == day_bucket else 0,
    }


@app.get("/admin/keys")
def admin_list_keys(include_inactive: bool = False, _ctx: dict = Depends(require_access("admin"))) -> dict:
    return {
        "count": len(key_store.list_api_keys(include_inactive=include_inactive)),
        "keys": key_store.list_api_keys(include_inactive=include_inactive),
    }


@app.post("/admin/keys")
def admin_create_key(request: ApiKeyCreateRequest, _ctx: dict = Depends(require_access("admin"))) -> dict:
    role = request.role.strip().lower()
    if role not in _role_rank:
        raise HTTPException(status_code=400, detail="invalid role")

    created = key_store.create_api_key(role=role, label=request.label)
    return {
        "warning": "api_key is shown once. store it securely.",
        "key": created,
    }


@app.post("/admin/keys/revoke")
def admin_revoke_key(request: ApiKeyRevokeRequest, _ctx: dict = Depends(require_access("admin"))) -> dict:
    ok = key_store.revoke_api_key(request.key_id.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="key not found or already revoked")
    return {"status": "revoked", "key_id": request.key_id.strip()}
