import argparse
import ast
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DELIM = " +++$+++ "

PLACEHOLDER_PATTERNS = [
    re.compile(r"\bORD-\d+\b"),
    re.compile(r"\bINV-\d+\b"),
    re.compile(r"\bITM-\d+\b"),
    re.compile(r"\bE\d{3,5}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{3,5}\s?EUR\b"),
]


def make_pair_record(
    conversation_id,
    turn_index,
    prompt,
    response,
    source_dataset,
    movie_id="",
    movie_title="",
    movie_year="",
    genres="",
    speaker_id="",
    speaker_name="",
    speaker_gender="?",
    listener_id="",
    listener_name="",
    listener_gender="?",
    is_pretranslated_fr="0",
):
    return {
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "movie_id": movie_id,
        "movie_title": movie_title,
        "movie_year": movie_year,
        "genres": genres,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "speaker_gender": speaker_gender,
        "listener_id": listener_id,
        "listener_name": listener_name,
        "listener_gender": listener_gender,
        "prompt_en": prompt,
        "response_en": response,
        "source_dataset": source_dataset,
        "is_pretranslated_fr": is_pretranslated_fr,
    }


def clean_text(text: str) -> str:
    text = text or ""
    text = html.unescape(text)
    text = text.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = text.strip("\"'")
    return text


def normalize_translation_text(text: str):
    normalized = text
    placeholders = []

    def repl(match):
        token = f"ZZPH{len(placeholders)}ZZ"
        placeholders.append((token, match.group(0)))
        return token

    for pattern in PLACEHOLDER_PATTERNS:
        normalized = pattern.sub(repl, normalized)

    return normalized, placeholders


def restore_translation_text(translated_text: str, placeholders):
    out = translated_text
    for token, value in placeholders:
        out = out.replace(token, value)
    return out


def parse_titles(path: Path):
    movies = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) < 6:
                continue
            movie_id, title, year, imdb_rating, imdb_votes, genres = parts[:6]
            try:
                parsed_genres = ast.literal_eval(genres)
                if isinstance(parsed_genres, list):
                    genres_text = ", ".join(str(g) for g in parsed_genres)
                else:
                    genres_text = str(parsed_genres)
            except Exception:
                genres_text = genres
            movies[movie_id] = {
                "movie_title": clean_text(title),
                "movie_year": year,
                "imdb_rating": imdb_rating,
                "imdb_votes": imdb_votes,
                "genres": clean_text(genres_text),
            }
    return movies


def parse_characters(path: Path):
    chars = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) < 6:
                continue
            char_id, char_name, movie_id, movie_title, gender, credit_pos = parts[:6]
            chars[char_id] = {
                "character_name": clean_text(char_name),
                "character_gender": gender,
                "character_credit_pos": credit_pos,
                "movie_id": movie_id,
                "movie_title_from_char": clean_text(movie_title),
            }
    return chars


def parse_lines(path: Path):
    lines = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.rstrip("\n").split(DELIM)
            if len(parts) < 5:
                continue
            line_id, character_id, movie_id, character_name, text = parts[:5]
            lines[line_id] = {
                "line_id": line_id,
                "character_id": character_id,
                "movie_id": movie_id,
                "character_name_line": clean_text(character_name),
                "text": clean_text(text),
            }
    return lines


def parse_conversations(path: Path):
    conversations = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f, start=1):
            parts = raw.rstrip("\n").split(DELIM)
            if len(parts) < 4:
                continue
            char1, char2, movie_id, utterance_list = parts[:4]
            try:
                line_ids = ast.literal_eval(utterance_list)
                if not isinstance(line_ids, list):
                    continue
            except Exception:
                continue
            conversations.append(
                {
                    "conversation_id": f"conv_{idx}",
                    "char1": char1,
                    "char2": char2,
                    "movie_id": movie_id,
                    "line_ids": line_ids,
                }
            )
    return conversations


def extract_dialog_turns(raw_dialog: str):
    raw_dialog = (raw_dialog or "").strip()
    if not raw_dialog:
        return []

    # Preferred path for test.csv format: ['utt1' 'utt2' ...] (adjacent literals).
    if raw_dialog.startswith("[") and raw_dialog.endswith("]"):
        inner = raw_dialog[1:-1].strip()
        candidate_tokens = re.split(r"(?<=[\"'])\s+(?=[\"'])", inner, flags=re.S)
        candidate_turns = []
        for tok in candidate_tokens:
            t = tok.strip().strip(",")
            if len(t) >= 2 and t[0] == t[-1] and t[0] in {'\"', "'"}:
                t = t[1:-1]
            t = clean_text(t)
            if t:
                candidate_turns.append(t)
        if len(candidate_turns) >= 2:
            return candidate_turns

    try:
        parsed = ast.literal_eval(raw_dialog)
        if isinstance(parsed, list):
            turns = [clean_text(str(x)) for x in parsed]
            turns = [t for t in turns if t]
            if len(turns) >= 2:
                return turns
            if len(turns) == 1:
                # Some rows use adjacent Python string literals, which ast joins
                # into one long string. Keep parsing with regex fallback.
                raw_dialog = turns[0]
        elif isinstance(parsed, str):
            raw_dialog = parsed
    except Exception:
        pass

    # Fallback: pull quoted segments when the list string is malformed.
    matches = re.findall(
        r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"",
        raw_dialog,
        flags=re.S,
    )
    turns = []
    for g1, g2 in matches:
        text = clean_text(g1 or g2)
        if text:
            turns.append(text)
    return turns


def parse_test_csv_pairs(path: Path):
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            turns = extract_dialog_turns(row.get("dialog", ""))
            if len(turns) < 2:
                continue

            for i in range(len(turns) - 1):
                prompt = clean_text(turns[i])
                response = clean_text(turns[i + 1])

                if not prompt or not response:
                    continue
                if len(prompt) > 300 or len(response) > 300:
                    continue

                rows.append(
                    make_pair_record(
                        conversation_id=f"testcsv_{idx}",
                        turn_index=i,
                        prompt=prompt,
                        response=response,
                        source_dataset="test_csv_dialog_act_emotion",
                    )
                )
    return rows


def get_row_text(row):
    for key in ("text", "question", "answer", "utterance", "sentence"):
        if key in row and row.get(key):
            return clean_text(row.get(key, ""))

    values = [clean_text(str(v)) for v in row.values() if clean_text(str(v))]
    if len(values) >= 2:
        return values[1]
    if values:
        return values[0]
    return ""


def parse_question_answer_csv_pairs(path: Path):
    is_fr_source = path.stem.lower().endswith("_fr")

    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows

        cols = {c.strip().lower() for c in reader.fieldnames if c}
        if "question" not in cols or "answer" not in cols:
            return rows

        for idx, row in enumerate(reader, start=1):
            prompt = clean_text(row.get("question", ""))
            response = clean_text(row.get("answer", ""))
            if not prompt or not response:
                continue
            if len(prompt) > 300 or len(response) > 300:
                continue

            rows.append(
                make_pair_record(
                    conversation_id=f"qa_{path.stem}_{idx}",
                    turn_index=0,
                    prompt=prompt,
                    response=response,
                    source_dataset=f"qa_csv:{path.name}",
                    is_pretranslated_fr="1" if is_fr_source else "0",
                )
            )

    return rows


def parse_dialog_csv_pairs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        cols = {c.strip().lower() for c in reader.fieldnames if c}
        if "dialog" not in cols:
            return rows

        for idx, row in enumerate(reader, start=1):
            turns = extract_dialog_turns(row.get("dialog", ""))
            if len(turns) < 2:
                continue

            for i in range(len(turns) - 1):
                prompt = clean_text(turns[i])
                response = clean_text(turns[i + 1])
                if not prompt or not response:
                    continue
                if len(prompt) > 300 or len(response) > 300:
                    continue

                rows.append(
                    make_pair_record(
                        conversation_id=f"dialog_{path.stem}_{idx}",
                        turn_index=i,
                        prompt=prompt,
                        response=response,
                        source_dataset=f"dialog_csv:{path.name}",
                    )
                )
    return rows


def parse_input_target_csv_pairs(input_path: Path, target_path: Path):
    rows = []
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as fi, target_path.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as ft:
        input_reader = csv.DictReader(fi)
        target_reader = csv.DictReader(ft)

        for idx, (in_row, tg_row) in enumerate(zip(input_reader, target_reader), start=1):
            prompt = get_row_text(in_row)
            response = get_row_text(tg_row)

            if not prompt or not response:
                continue
            if len(prompt) > 300 or len(response) > 300:
                continue

            rows.append(
                make_pair_record(
                    conversation_id=f"input_target_{input_path.stem}_{idx}",
                    turn_index=0,
                    prompt=prompt,
                    response=response,
                    source_dataset=f"input_target_csv:{input_path.name}+{target_path.name}",
                )
            )
    return rows


def discover_additional_csv_pairs(root: Path, prefer_french_synthetic: bool = True):
    all_pairs = []
    source_counts = {}
    consumed = set()

    csv_files = sorted(root.glob("*.csv"))
    csv_names = {p.name.lower() for p in csv_files}
    has_fr_synthetic = "synthetic_realistic_qa_fr.csv" in csv_names

    for csv_path in csv_files:
        name = csv_path.name.lower()
        if name.endswith(".crdownload"):
            continue
        if csv_path.name in consumed:
            continue

        # If a native French synthetic source exists, skip the English synthetic
        # duplicate so we don't spend CPU time translating redundant content.
        if prefer_french_synthetic and has_fr_synthetic and name == "synthetic_realistic_qa.csv":
            continue

        if name.startswith("input"):
            suffix = csv_path.stem[5:]
            target_name = f"target{suffix}.csv"
            target_path = root / target_name
            if target_path.exists():
                pairs = parse_input_target_csv_pairs(csv_path, target_path)
                if pairs:
                    all_pairs.extend(pairs)
                    source_counts[f"input_target_csv:{csv_path.name}+{target_path.name}"] = len(pairs)
                consumed.add(csv_path.name)
                consumed.add(target_path.name)
                continue

        pairs = parse_question_answer_csv_pairs(csv_path)
        if pairs:
            all_pairs.extend(pairs)
            source_counts[f"qa_csv:{csv_path.name}"] = len(pairs)
            consumed.add(csv_path.name)
            continue

        pairs = parse_dialog_csv_pairs(csv_path)
        if pairs:
            all_pairs.extend(pairs)
            source_counts[f"dialog_csv:{csv_path.name}"] = len(pairs)
            consumed.add(csv_path.name)
            continue

    return all_pairs, source_counts


def build_pairs(lines_map, conversations, chars_map, movies_map):
    pairs = []
    seen = set()
    for conv in conversations:
        line_ids = conv["line_ids"]
        if len(line_ids) < 2:
            continue
        for i in range(len(line_ids) - 1):
            src = lines_map.get(line_ids[i])
            tgt = lines_map.get(line_ids[i + 1])
            if not src or not tgt:
                continue

            prompt = clean_text(src["text"])
            response = clean_text(tgt["text"])

            if not prompt or not response:
                continue
            if len(prompt) > 300 or len(response) > 300:
                continue

            dedup_key = (prompt.lower(), response.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            movie = movies_map.get(src["movie_id"], {})
            src_char = chars_map.get(src["character_id"], {})
            tgt_char = chars_map.get(tgt["character_id"], {})

            pairs.append(
                make_pair_record(
                    conversation_id=conv["conversation_id"],
                    turn_index=i,
                    prompt=prompt,
                    response=response,
                    source_dataset="cornell_movie_dialogs",
                    movie_id=src["movie_id"],
                    movie_title=movie.get("movie_title", ""),
                    movie_year=movie.get("movie_year", ""),
                    genres=movie.get("genres", ""),
                    speaker_id=src["character_id"],
                    speaker_name=src_char.get("character_name", src.get("character_name_line", "")),
                    speaker_gender=src_char.get("character_gender", "?"),
                    listener_id=tgt["character_id"],
                    listener_name=tgt_char.get("character_name", tgt.get("character_name_line", "")),
                    listener_gender=tgt_char.get("character_gender", "?"),
                )
            )
    return pairs


def dedup_pairs(rows):
    seen = set()
    out = []
    for row in rows:
        key = (row.get("prompt_en", "").lower(), row.get("response_en", "").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_translation_cache(cache_path: Path):
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_translation_cache(cache_path: Path, cache_data):
    cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_text_mymemory(text: str, cache_data, pause_s: float = 0.1, retries: int = 2):
    text = clean_text(text)
    if not text:
        return ""
    key = f"en|fr::{text}"
    if key in cache_data:
        return cache_data[key]

    query = urllib.parse.quote(text)
    url = f"https://api.mymemory.translated.net/get?q={query}&langpair=en|fr"

    translated = ""
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                translated = payload.get("responseData", {}).get("translatedText", "")
                translated = clean_text(translated)
                if translated:
                    break
        except Exception:
            translated = ""
        time.sleep(pause_s)

    cache_data[key] = translated
    return translated


def ensure_argos_en_fr_package():
    import argostranslate.package
    import argostranslate.translate

    installed_languages = argostranslate.translate.get_installed_languages()
    for lang in installed_languages:
        if lang.code == "en":
            for target in installed_languages:
                if target.code == "fr":
                    if lang.get_translation(target):
                        return

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = None
    for pkg in available_packages:
        if pkg.from_code == "en" and pkg.to_code == "fr":
            package_to_install = pkg
            break

    if package_to_install is None:
        raise RuntimeError("No Argos EN->FR package available")

    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)


def build_argos_translator():
    import argostranslate.translate

    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((x for x in installed_languages if x.code == "en"), None)
    to_lang = next((x for x in installed_languages if x.code == "fr"), None)
    if from_lang is None or to_lang is None:
        raise RuntimeError("Argos languages EN/FR not installed")
    return from_lang.get_translation(to_lang)


def translate_argos_parallel(unique_texts, cache, cache_path, workers=6, checkpoint_step=50):
    translator = build_argos_translator()

    translated_texts = {}
    text_payload = {}
    normalized_to_texts = {}

    for text in unique_texts:
        normalized, placeholders = normalize_translation_text(text)
        text_payload[text] = (normalized, placeholders)
        normalized_to_texts.setdefault(normalized, []).append(text)

    to_translate = []
    for normalized in normalized_to_texts.keys():
        key = f"en|fr::{normalized}"
        if key not in cache or not cache.get(key, ""):
            to_translate.append(normalized)

    def _translate_one(normalized_text):
        try:
            out = clean_text(translator.translate(normalized_text))
        except Exception:
            out = ""
        return normalized_text, out

    completed = 0
    total = len(to_translate)
    if total == 0:
        return translated_texts

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_translate_one, normalized) for normalized in to_translate]
        for fut in as_completed(futures):
            normalized, translated = fut.result()
            key = f"en|fr::{normalized}"
            cache[key] = translated
            completed += 1

            if completed % checkpoint_step == 0:
                save_translation_cache(cache_path, cache)
                print(f"Translated unique texts (new): {completed}/{total}", flush=True)

    raw_unique = len(unique_texts)
    normalized_unique = len(normalized_to_texts)
    print(
        f"Translation normalization: raw_unique={raw_unique}, template_unique={normalized_unique}",
        flush=True,
    )

    for text in unique_texts:
        normalized, placeholders = text_payload[text]
        translated_template = cache.get(f"en|fr::{normalized}", "")
        if not translated_template:
            translated_texts[text] = text
            continue
        translated_texts[text] = restore_translation_text(translated_template, placeholders)

    return translated_texts


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Prepare merged EN/FR chatbot datasets from Cornell movie corpus.")
    parser.add_argument("--workspace", default=".", help="Workspace folder containing Cornell files")
    parser.add_argument("--max-translate-rows", type=int, default=5000, help="Rows to translate to French")
    parser.add_argument("--translate-all", action="store_true", help="Translate all rows (can take very long)")
    parser.add_argument(
        "--translator",
        choices=["mymemory", "argos"],
        default="mymemory",
        help="Translation backend: mymemory API or local argos",
    )
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers for Argos translation")
    parser.add_argument(
        "--prefer-french-synthetic",
        action="store_true",
        default=True,
        help="When synthetic_realistic_qa_fr.csv exists, skip synthetic_realistic_qa.csv to reduce translation workload",
    )
    parser.add_argument(
        "--no-prefer-french-synthetic",
        action="store_false",
        dest="prefer_french_synthetic",
        help="Include both French and English synthetic datasets",
    )
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines_path = root / "movie_lines.txt"
    conv_path = root / "movie_conversations.txt"
    chars_path = root / "movie_characters_metadata.txt"
    titles_path = root / "movie_titles_metadata.txt"
    movies = parse_titles(titles_path)
    chars = parse_characters(chars_path)
    lines = parse_lines(lines_path)
    conversations = parse_conversations(conv_path)

    cornell_pairs = build_pairs(lines, conversations, chars, movies)
    additional_pairs, additional_counts = discover_additional_csv_pairs(
        root, prefer_french_synthetic=args.prefer_french_synthetic
    )
    pairs = dedup_pairs(cornell_pairs + additional_pairs)

    en_path = out_dir / "chatbot_pairs_en.csv"
    en_fields = [
        "conversation_id",
        "turn_index",
        "movie_id",
        "movie_title",
        "movie_year",
        "genres",
        "speaker_id",
        "speaker_name",
        "speaker_gender",
        "listener_id",
        "listener_name",
        "listener_gender",
        "prompt_en",
        "response_en",
        "source_dataset",
        "is_pretranslated_fr",
    ]
    write_csv(en_path, pairs, en_fields)

    cache_path = out_dir / "translation_cache_en_fr.json"
    cache = load_translation_cache(cache_path)

    limit = len(pairs) if args.translate_all else min(args.max_translate_rows, len(pairs))
    selected_rows = pairs[:limit]

    unique_texts = set()
    for row in selected_rows:
        if row.get("is_pretranslated_fr") == "1":
            continue
        if row.get("prompt_en"):
            unique_texts.add(row["prompt_en"])
        if row.get("response_en"):
            unique_texts.add(row["response_en"])

    translated_texts = {}

    if args.translator == "argos":
        ensure_argos_en_fr_package()
        translated_texts = translate_argos_parallel(
            unique_texts,
            cache,
            cache_path,
            workers=max(1, args.workers),
            checkpoint_step=50,
        )
        save_translation_cache(cache_path, cache)

        # Ensure texts already in cache but not in translated_texts are also resolved.
        for idx, text in enumerate(unique_texts, start=1):
            if text not in translated_texts:
                translated_texts[text] = cache.get(f"en|fr::{text}", "")
            if idx % 2000 == 0:
                print(f"Resolved unique texts: {idx}/{len(unique_texts)}", flush=True)
    else:
        for idx, text in enumerate(unique_texts, start=1):
            translated = translate_text_mymemory(text, cache)
            translated_texts[text] = translated or text
            if idx % 200 == 0:
                save_translation_cache(cache_path, cache)
                print(f"Translated unique texts: {idx}/{len(unique_texts)}", flush=True)

    fr_rows = []
    for row in selected_rows:
        if row.get("is_pretranslated_fr") == "1":
            p_fr = row["prompt_en"]
            r_fr = row["response_en"]
        else:
            p_fr = translated_texts.get(row["prompt_en"], row["prompt_en"])
            r_fr = translated_texts.get(row["response_en"], row["response_en"])

        fr_rows.append(
            {
                **row,
                "prompt_fr": p_fr,
                "response_fr": r_fr,
                "language": "fr",
            }
        )

    save_translation_cache(cache_path, cache)

    fr_path = out_dir / "chatbot_pairs_fr.csv"
    fr_fields = en_fields + ["prompt_fr", "response_fr", "language"]
    write_csv(fr_path, fr_rows, fr_fields)

    train_rows = []
    for row in fr_rows:
        train_rows.append(
            {
                "instruction": row["prompt_fr"],
                "response": row["response_fr"],
                "source": row.get("source_dataset", "unknown"),
                "conversation_id": row["conversation_id"],
                "movie_title": row["movie_title"],
                "genres": row["genres"],
            }
        )

    train_path = out_dir / "chatbot_train_fr.csv"
    train_fields = ["instruction", "response", "source", "conversation_id", "movie_title", "genres"]
    write_csv(train_path, train_rows, train_fields)

    report = {
        "total_pairs_en": len(pairs),
        "cornell_pairs_en": len(cornell_pairs),
        "additional_csv_pairs_en": len(additional_pairs),
        "additional_csv_source_counts": additional_counts,
        "translated_rows": len(fr_rows),
        "outputs": {
            "en": str(en_path),
            "fr": str(fr_path),
            "train": str(train_path),
            "cache": str(cache_path),
        },
    }

    report_path = out_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
