import gradio as gr
import re
import torch
import uuid
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DOMAIN_TOPICS = [
    "analyse de donnees",
    "machine learning",
    "ia appliquee",
    "automatisation",
    "pipelines",
    "api",
]

IN_DOMAIN_KEYWORDS = {
    "data", "donnee", "donnees", "dataset", "csv", "json", "excel", "table", "sql",
    "pandas", "numpy", "analyse", "nettoyage", "feature", "visualisation", "statistique",
    "ml", "ia", "ai", "machine learning", "modele", "model", "entrainement", "evaluation",
    "classification", "regression", "prediction", "prompt", "llm", "token", "embedding",
    "pipeline", "workflow", "automatisation", "script", "python", "fastapi", "api", "docker",
}

OUT_DOMAIN_KEYWORDS = {
    "medecine", "medical", "maladie", "diagnostic", "traitement", "politique", "election",
    "religion", "psychologie", "depression", "amour", "relation", "sexe", "voyance", "astrologie",
    "finance personnelle", "pari", "bet", "casino", "juridique", "avocat",
}

MODEL_ID = "Elie224/Elibot"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.0
TOP_P = 0.9
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 3
HISTORY_TURNS = 4
HISTORY_MODE = "user-only"
SYSTEM_PROMPT = (
    "Tu es Elibot, un assistant specialise en analyse de donnees, IA appliquee et automatisation. "
    "Tu reponds de facon claire, concise et professionnelle. "
    "Tu refuses poliment les sujets hors domaine et rediriges vers une demande technique."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(model_id):
    # Some Spaces/transformers versions fail if extra_special_tokens is a list in tokenizer config.
    try:
        return AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})
    except Exception:
        return AutoTokenizer.from_pretrained(model_id)


TOKENIZER = load_tokenizer(MODEL_ID)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
MODEL.eval()


def clean_generated_text(text):
    value = " ".join(text.strip().split())
    if not value:
        return value

    parts = re.split(r"(?<=[.!?])\s+", value)
    dedup_parts = []
    for part in parts:
        if not part:
            continue
        if dedup_parts and part.lower() == dedup_parts[-1].lower():
            continue
        dedup_parts.append(part)
    value = " ".join(dedup_parts)

    tokens = value.split()
    collapsed = []
    run_token = ""
    run_count = 0
    for tok in tokens:
        key = tok.lower()
        if key == run_token:
            run_count += 1
        else:
            run_token = key
            run_count = 1
        if run_count <= 2:
            collapsed.append(tok)
    value = " ".join(collapsed)

    if value and value[-1] not in ".!?":
        value += "."
    return value


def _normalize_text(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", value.lower())).strip()


def is_in_domain_query(user_text):
    text = _normalize_text(user_text or "")
    if not text:
        return True
    if text in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc", "merci"}:
        return True
    if any(k in text for k in OUT_DOMAIN_KEYWORDS):
        return False
    return any(k in text for k in IN_DOMAIN_KEYWORDS)


def out_of_domain_reply():
    topics = ", ".join(DOMAIN_TOPICS)
    return (
        "Je suis specialise en "
        f"{topics}. "
        "Je ne traite pas les sujets hors de ce cadre. "
        "Pose une question technique (ex: pipeline ML, API FastAPI, nettoyage de dataset) et je t'aide."
    )


def is_low_quality_answer(answer):
    text = " ".join((answer or "").strip().lower().split())
    if not text:
        return True

    bad_patterns = [
        "je ne peux pas te dire que tu veux dire",
        "je veux dire que je n'ai pas besoin",
        "reheatreheatreheat",
        "synchronoussynchronous",
        "municipal municipal",
    ]
    if any(p in text for p in bad_patterns):
        return True

    # Many duplicated words strongly indicate degeneration.
    tokens = text.split()
    if len(tokens) >= 12:
        uniq_ratio = len(set(tokens)) / max(1, len(tokens))
        if uniq_ratio < 0.55:
            return True

    if text in {"je ne sais pas.", "je sais pas.", "ok.", "d'accord."}:
        return True

    return False


def fallback_reply(user_text, profile):
    q = (user_text or "").lower().strip()

    if not is_in_domain_query(q):
        return out_of_domain_reply()

    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    if ("comment" in q and "t'appelle" in q) or ("ton nom" in q) or ("qui es tu" in q) or ("qui es-tu" in q):
        return "Je m'appelle Elibot."

    if "je comprends pas" in q or "je ne comprends pas" in q:
        return "Pas de souci. Dis-moi juste ce que tu veux faire, en une phrase simple."

    if "merci" in q:
        return "Avec plaisir."

    if "prenom" in profile:
        return f"D'accord {profile['prenom']}, peux-tu reformuler en une phrase simple ?"

    return "Je n'ai pas bien compris. Peux-tu reformuler en une phrase simple ?"


def update_profile_from_user_text(user_text, profile):
    text = user_text.strip()

    name_match = re.search(
        r"(?:je m'appelle|mon prenom est|mon prénom est|je suis)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        profile["prenom"] = name_match.group(1)

    city_match = re.search(
        r"(?:je viens de|j'habite a|j'habite à|je vis a|je vis à|ma ville est)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if city_match:
        profile["ville"] = city_match.group(1)

    sport_match = re.search(
        r"(?:j'aime le|j'aime la|j'adore le|j'adore la|mon sport prefere est le|mon sport prefere est la|mon sport prefere c'est le|mon sport prefere c'est la|mon sport préféré est le|mon sport préféré est la|mon sport préféré c'est le|mon sport préféré c'est la)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
        text,
        flags=re.IGNORECASE,
    )
    if not sport_match:
        sport_match = re.search(
            r"(?:mon sport prefere|mon sport préféré)\s*(?:est|c'est)?\s*(?:le|la)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-']+)",
            text,
            flags=re.IGNORECASE,
        )
    if sport_match:
        profile["sport"] = sport_match.group(1)

    goal_match = re.search(
        r"(?:objectif|courir|course|entrainement|entraînement|seance|séance)?.{0,20}?(\d+)\s*minutes",
        text,
        flags=re.IGNORECASE,
    )
    if goal_match:
        profile["objectif_minutes"] = goal_match.group(1)


def build_memory_lines(profile):
    lines = []
    if "prenom" in profile:
        lines.append(f"Memoire: Le prenom utilisateur est {profile['prenom']}.")
    if "ville" in profile:
        lines.append(f"Memoire: La ville utilisateur est {profile['ville']}.")
    if "sport" in profile:
        lines.append(f"Memoire: Le sport prefere utilisateur est {profile['sport']}.")
    if "objectif_minutes" in profile:
        lines.append(f"Memoire: L'objectif sport utilisateur est {profile['objectif_minutes']} minutes.")
    return lines


def maybe_rule_reply(user_text, profile):
    q = user_text.lower().strip()
    q_norm = re.sub(r"[^a-zà-öø-ÿ0-9\s]", "", q)

    if not is_in_domain_query(q):
        return out_of_domain_reply()

    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    if "que fais tu" in q_norm or "tu fais quoi" in q_norm or "ton domaine" in q:
        return (
            "Je suis specialise en analyse de donnees, IA appliquee et automatisation. "
            "Je peux t'aider sur pipelines, modeles, API, debugging Python et workflows techniques."
        )

    asks_name = (
        ("prenom" in q)
        or ("prénom" in q)
        or ("comment je m'appelle" in q)
        or ("qui suis-je" in q)
        or ("comment tu t'appelle" in q)
        or ("comment tu t'appelles" in q)
        or ("ton nom" in q)
    )
    asks_city = ("ville" in q) or ("j'habite" in q) or ("je vis" in q)
    asks_sport = ("sport" in q) or ("j'aime" in q)
    asks_goal = ("objectif" in q) or ("course" in q) or ("entrain" in q)
    asks_advice = ("conseil" in q) or ("demain" in q)

    if (
        ("qui es-tu" in q)
        or ("qui es tu" in q)
        or ("ton nom" in q)
        or ("comment tu t'appelle" in q)
        or ("comment tu t'appelles" in q)
    ):
        return "Je m'appelle Elibot. Je suis la pour discuter avec toi de facon naturelle et utile."

    if asks_name and asks_city:
        if "prenom" in profile and "ville" in profile:
            return f"Si je me souviens bien, tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return None

    if asks_name and "prenom" in profile:
        return f"Bien sur: tu t'appelles {profile['prenom']}."

    if asks_city and "ville" in profile:
        return f"Tu m'avais dit que tu viens de {profile['ville']}."

    if asks_advice and "sport" in profile:
        return (
            f"Top, on fait simple pour demain: 30 minutes de {profile['sport']}, "
            "5 minutes d'echauffement au debut, puis etirements en fin de seance."
        )

    if asks_sport and "sport" in profile:
        return f"Ton sport prefere, c'est le {profile['sport']}."

    if asks_goal and "objectif_minutes" in profile:
        return f"Ton objectif, c'est de courir {profile['objectif_minutes']} minutes."

    if ("qui suis-je" in q or "qui je suis" in q) and "prenom" in profile:
        if "ville" in profile:
            return f"Tu t'appelles {profile['prenom']} et tu viens de {profile['ville']}."
        return f"Tu t'appelles {profile['prenom']}."

    return None


def build_prompt(history, user_text, profile):
    lines = [f"Systeme: {SYSTEM_PROMPT}"]
    lines.extend(build_memory_lines(profile))

    if history:
        recent = history[-HISTORY_TURNS:]
        for user_msg, bot_msg in recent:
            lines.append(f"Utilisateur: {user_msg}")
            if HISTORY_MODE == "full":
                lines.append(f"Assistant: {bot_msg}")

    lines.append(f"Utilisateur: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat(message, history):
    history = history or []
    state = {
        "history": list(history),
        "profile": {},
    }
    for h_user, _ in state["history"]:
        update_profile_from_user_text(h_user, state["profile"])

    user_text = (message or "").strip()
    if not user_text:
        return state["history"], state["history"]

    update_profile_from_user_text(user_text, state["profile"])
    direct = maybe_rule_reply(user_text, state["profile"])
    if direct:
        answer = direct
        state["history"].append((user_text, answer))
        return state["history"], state["history"]

    prompt = build_prompt(state["history"], user_text, state["profile"])

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)

    use_sampling = TEMPERATURE > 0
    gen_kwargs = {
        "do_sample": use_sampling,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM,
        "num_beams": 1,
    }
    if use_sampling:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **gen_kwargs)

    answer = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    answer = clean_generated_text(answer)
    if not answer or is_low_quality_answer(answer):
        answer = fallback_reply(user_text, state["profile"])

    state["history"].append((user_text, answer))
    return state["history"], state["history"]


def handle_submit(message, history):
        new_chat, new_state = chat(message, history)
        return new_chat, new_state, ""


def _format_tasks(tasks):
    if not tasks:
        return "Aucune tache pour le moment."

    lines = []
    for t in tasks:
        lines.append(
            f"- [{t['status']}] {t['title']} (restant: {t['remaining_steps']}, erreurs: {len(t['errors'])})"
        )
    return "\n".join(lines)


def _task_choices(tasks):
    return [f"{t['task_id']} | {t['title']}" for t in tasks]


def add_task_ui(title, steps_text, tasks):
    items = list(tasks or [])
    clean_title = (title or "").strip()
    if not clean_title:
        return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None), "", ""

    steps = [x.strip() for x in (steps_text or "").split(",") if x.strip()]
    items.append(
        {
            "task_id": str(uuid.uuid4())[:8],
            "title": clean_title,
            "steps": steps,
            "done_steps": [],
            "remaining_steps": len(steps),
            "status": "active",
            "errors": [],
        }
    )
    return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None), "", ""


def mark_task_done_ui(selected_task, tasks):
    items = list(tasks or [])
    if not selected_task:
        return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None)

    task_id = selected_task.split("|", 1)[0].strip()
    for t in items:
        if t["task_id"] == task_id:
            t["status"] = "done"
            t["remaining_steps"] = 0
            break
    return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None)


def mark_task_error_ui(selected_task, error_text, tasks):
    items = list(tasks or [])
    if not selected_task:
        return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None), ""

    task_id = selected_task.split("|", 1)[0].strip()
    for t in items:
        if t["task_id"] == task_id:
            t["status"] = "blocked"
            if (error_text or "").strip():
                t["errors"].append(error_text.strip())
            break
    return items, _format_tasks(items), gr.update(choices=_task_choices(items), value=None), ""


APP_CSS = """
.app-shell {
    max-width: 1200px;
    margin: 0 auto;
    border-radius: 20px;
    border: 1px solid #d7e4da;
    background: linear-gradient(180deg, #f8fffb 0%, #f1f7f4 100%);
    box-shadow: 0 20px 40px rgba(18, 52, 38, 0.08);
    padding: 18px;
}
.hero-title h1 {
    margin: 0;
    color: #14532d;
    letter-spacing: 0.2px;
}
.hero-sub {
    color: #3f4f46;
    margin-top: 6px;
}
.quick-row .gr-button {
    border-radius: 999px !important;
    border: 1px solid #b6d8c4 !important;
    background: #e7f5ec !important;
    color: #0f5132 !important;
}
#send-btn {
    background: #166534 !important;
    color: white !important;
}
#reset-btn {
    background: #0f766e !important;
    color: white !important;
}
.task-card {
    border: 1px solid #d7e4da;
    border-radius: 14px;
    background: #f6fbf8;
    padding: 10px;
    display: block !important;
}
"""


with gr.Blocks(title="Elibot", css=APP_CSS, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML('<div class="hero-title"><h1>Elibot</h1></div><div class="hero-sub">Assistant specialise en data, IA appliquee et automatisation.</div>')
        gr.Markdown("UI version: tasks-panel-v2")

        state = gr.State([])
        tasks_state = gr.State([])

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=620):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=430,
                    bubble_full_width=False,
                    show_copy_button=True,
                )

                with gr.Row(elem_classes=["quick-row"]):
                    quick_1 = gr.Button("Explique un pipeline ML", size="sm")
                    quick_2 = gr.Button("Corrige ce code pandas", size="sm")
                    quick_3 = gr.Button("Architecture API FastAPI", size="sm")
                    quick_4 = gr.Button("Automatiser un workflow CSV", size="sm")

                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ecris ton message ici...",
                    lines=2,
                    max_lines=4,
                )

                with gr.Row():
                    send = gr.Button("Envoyer", elem_id="send-btn", variant="primary")
                    clear = gr.Button("Reinitialiser", elem_id="reset-btn")

            with gr.Column(scale=2, min_width=320, elem_classes=["task-card"], visible=True):
                gr.Markdown("### Taches")
                task_title = gr.Textbox(label="Nouvelle tache", placeholder="Ex: Preparer pipeline")
                task_steps = gr.Textbox(label="Etapes (virgules)", placeholder="collecte, nettoyage, entrainement")
                add_task_btn = gr.Button("Ajouter tache", variant="primary")
                task_select = gr.Dropdown(label="Selectionner tache", choices=[])
                task_error = gr.Textbox(label="Erreur tache", placeholder="Optionnel")
                with gr.Row():
                    task_done_btn = gr.Button("Marquer terminee")
                    task_block_btn = gr.Button("Signaler erreur")
                task_view = gr.Markdown("Aucune tache pour le moment.")

        send.click(handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg])
        msg.submit(handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg])
        clear.click(lambda: ([], [], ""), inputs=None, outputs=[chatbot, state, msg])

        quick_1.click(lambda: "Peux-tu expliquer un pipeline machine learning de bout en bout ?", outputs=[msg]).then(
                handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg]
        )
        quick_2.click(lambda: "Voici un script pandas lent, comment l'optimiser ?", outputs=[msg]).then(
                handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg]
        )
        quick_3.click(lambda: "Propose une architecture FastAPI pour servir un modele ML.", outputs=[msg]).then(
                handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg]
        )
        quick_4.click(lambda: "Comment automatiser un workflow de nettoyage CSV en Python ?", outputs=[msg]).then(
                handle_submit, inputs=[msg, state], outputs=[chatbot, state, msg]
        )

        add_task_btn.click(
            add_task_ui,
            inputs=[task_title, task_steps, tasks_state],
            outputs=[tasks_state, task_view, task_select, task_title, task_steps],
        )

        task_done_btn.click(
            mark_task_done_ui,
            inputs=[task_select, tasks_state],
            outputs=[tasks_state, task_view, task_select],
        )

        task_block_btn.click(
            mark_task_error_ui,
            inputs=[task_select, task_error, tasks_state],
            outputs=[tasks_state, task_view, task_select, task_error],
        )


demo.launch()
