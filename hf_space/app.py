import gradio as gr
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    "Tu es Elibot, un assistant francophone humain, naturel, poli et coherent. "
    "Tu reponds avec chaleur, de maniere claire et concise, sans etre robotique."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
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

    if q in {"bonjour", "salut", "hello", "bonsoir", "coucou"}:
        return "Salut, ravi de te parler. Comment je peux t'aider aujourd'hui ?"

    asks_name = ("prenom" in q) or ("prénom" in q) or ("comment je m'appelle" in q) or ("qui suis-je" in q)
    asks_city = ("ville" in q) or ("j'habite" in q) or ("je vis" in q)
    asks_sport = ("sport" in q) or ("j'aime" in q)
    asks_goal = ("objectif" in q) or ("course" in q) or ("entrain" in q)
    asks_advice = ("conseil" in q) or ("demain" in q)

    if ("qui es-tu" in q) or ("qui es tu" in q) or ("ton nom" in q) or ("comment tu t'appelles" in q):
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
    if not answer:
        answer = "Je suis la pour t'aider. Peux-tu reformuler en une phrase simple ?"

    state["history"].append((user_text, answer))
    return state["history"], state["history"]


with gr.Blocks(title="Elibot") as demo:
    gr.Markdown("# Elibot\nDiscute avec Elibot en francais.")
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Message", placeholder="Ecris ton message ici...", lines=2)
    state = gr.State([])

    send = gr.Button("Envoyer")
    clear = gr.Button("Reinitialiser")

    send.click(chat, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state])
    clear.click(lambda: ([], []), inputs=None, outputs=[chatbot, state])


demo.launch()
