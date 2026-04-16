import os
import uuid
from dataclasses import dataclass, field
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
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
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Tu es Elibot, un assistant francophone humain, naturel, poli et coherent. "
    "Tu reponds avec chaleur, de maniere claire et concise, sans etre robotique.",
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

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_DIR).to(_device)
_model.eval()


def _get_or_create_session(session_id: str | None) -> tuple[str, SessionState]:
    with _state_lock:
        sid = session_id or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = SessionState()
        return sid, _sessions[sid]


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

    state.history.append(("Utilisateur", user_text))
    state.history.append(("Assistant", answer))

    return ChatResponse(session_id=sid, response=answer, profile=state.profile)


@app.post("/reset/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    with _state_lock:
        if session_id in _sessions:
            del _sessions[session_id]
    return {"status": "reset", "session_id": session_id}
