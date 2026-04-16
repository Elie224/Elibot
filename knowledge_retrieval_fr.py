import math
import re
from pathlib import Path


KNOWLEDGE_DIR = Path("data/knowledge")
MAX_SNIPPET_CHARS = 480


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self.docs: list[dict] = []
        self.idf: dict[str, float] = {}
        self._load()

    @staticmethod
    def _normalize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", text)
        tokens = [t for t in text.split() if len(t) >= 2]
        return tokens

    def _load(self) -> None:
        if not self.knowledge_dir.exists():
            return

        files = []
        for pattern in ("*.md", "*.txt"):
            files.extend(sorted(self.knowledge_dir.glob(pattern)))

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            for idx, chunk in enumerate(self._split_chunks(text), start=1):
                tokens = self._normalize(chunk)
                if not tokens:
                    continue
                tf = {}
                for tok in tokens:
                    tf[tok] = tf.get(tok, 0) + 1
                self.docs.append(
                    {
                        "id": f"{file_path.name}:{idx}",
                        "source": str(file_path),
                        "text": chunk.strip(),
                        "tf": tf,
                        "token_count": len(tokens),
                    }
                )

        self._build_idf()

    def _split_chunks(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        buf = []
        buf_len = 0
        for para in paragraphs:
            if buf_len + len(para) > MAX_SNIPPET_CHARS and buf:
                chunks.append("\n\n".join(buf))
                buf = [para]
                buf_len = len(para)
            else:
                buf.append(para)
                buf_len += len(para)
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

    def _build_idf(self) -> None:
        df = {}
        n_docs = max(1, len(self.docs))
        for doc in self.docs:
            for tok in doc["tf"].keys():
                df[tok] = df.get(tok, 0) + 1

        self.idf = {tok: math.log((1 + n_docs) / (1 + d)) + 1.0 for tok, d in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_tokens = self._normalize(query)
        if not q_tokens or not self.docs:
            return []

        q_tf = {}
        for tok in q_tokens:
            q_tf[tok] = q_tf.get(tok, 0) + 1

        scored = []
        for doc in self.docs:
            score = 0.0
            for tok, count in q_tf.items():
                if tok not in doc["tf"]:
                    continue
                score += (count * doc["tf"][tok]) * self.idf.get(tok, 0.0)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]


def format_knowledge_context(results: list[dict]) -> str:
    if not results:
        return ""

    lines = ["Contexte technique interne:"]
    for i, item in enumerate(results, start=1):
        snippet = " ".join(item["text"].split())
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)
