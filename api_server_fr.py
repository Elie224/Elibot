import os
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import chat_with_model_fr as runtime


DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "models/chatbot-fr-flan-t5-small-v2-convfix")
DEFAULT_HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
DEFAULT_MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "72"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
DEFAULT_NO_REPEAT_NGRAM = int(os.getenv("NO_REPEAT_NGRAM", "3"))
DEFAULT_HISTORY_MODE = os.getenv("HISTORY_MODE", "user-only")
DEFAULT_CHAT_LOG_PATH = os.getenv("CHAT_LOG_PATH", "data/logs/elibot_chat_events.jsonl")
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


@dataclass
class SessionState:
    history: list[tuple[str, str]] = field(default_factory=list)
    profile: dict[str, str] = field(default_factory=dict)


app = FastAPI(title="Elibot API", version="1.0.0")

_state_lock = Lock()
_sessions: dict[str, SessionState] = {}
_log_lock = Lock()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_DIR).to(_device)
_model.eval()


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
            <div class="session" id="session">Session: -</div>
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
                    headers: { 'Content-Type': 'application/json' },
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
            await fetch(`/reset/${sessionId}`, { method: 'POST' });
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
def chat(request: ChatRequest) -> ChatResponse:
    sid, state = _get_or_create_session(request.session_id)
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    runtime.update_profile_from_user_text(user_text, state.profile)
    in_domain = runtime.is_in_domain_query(user_text)

    direct = runtime.maybe_rule_reply(user_text, state.profile)
    if direct:
        answer = direct
    else:
        prompt = runtime.build_prompt(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            history=state.history,
            user_text=user_text,
            history_turns=DEFAULT_HISTORY_TURNS,
            history_mode=DEFAULT_HISTORY_MODE,
            profile=state.profile,
            use_slot_memory=True,
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
        if runtime.is_low_quality_answer(answer):
            answer = runtime.fallback_reply(user_text)

    state.history.append(("Utilisateur", user_text))
    state.history.append(("Assistant", answer))

    _append_chat_event(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": sid,
            "user_text": user_text,
            "assistant_text": answer,
            "in_domain": in_domain,
            "used_rule_reply": bool(direct),
            "is_low_quality": runtime.is_low_quality_answer(answer),
            "profile": dict(state.profile),
        }
    )

    return ChatResponse(session_id=sid, response=answer, profile=state.profile)


@app.post("/reset/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    with _state_lock:
        if session_id in _sessions:
            del _sessions[session_id]
    return {"status": "reset", "session_id": session_id}
