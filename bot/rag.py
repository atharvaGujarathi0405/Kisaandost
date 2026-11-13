"""Keyword retrieval (BM25) over the farm documents in data/rag/chunks.jsonl (built by scripts/build_rag_index.py).

No vector database and no embedding API on purpose: the documents are English, the agent (bot/agent.py) writes the search query
in English itself, and BM25 on a few thousand chunks is fast, free and easy to explain. If recall turns out too low on real
questions, add embeddings then (see EXPERIMENT_LOG P11 for the retrieval check).
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

CHUNKS = Path(os.environ.get("KD_RAG_CHUNKS", Path(__file__).resolve().parent.parent / "data" / "rag" / "chunks.jsonl"))
K1, B = 1.5, 0.75
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with can may should".split())
_idx: dict = {}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+|[ऀ-ॿ]+", text.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)  # crude plural strip: pests -> pest
    return out


def load() -> dict:
    if not _idx:
        rows = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()] if CHUNKS.exists() else []
        docs = [Counter(tokens(r["title"] + " " + r["text"])) for r in rows]
        df = Counter(t for d in docs for t in d)
        n = len(rows)
        _idx.update(rows=rows, docs=docs, lens=[sum(d.values()) for d in docs], avg=(sum(sum(d.values()) for d in docs) / n) if n else 1.0,
                    idf={t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()})
    return _idx


def available() -> bool:
    return bool(load()["rows"])


def search(query: str, crop: str = "", k: int = 5) -> list[dict]:
    """Top-k chunks for the query. If crop is one of cotton/soybean/tur/orange, chunks about that crop rank first,
    chunks that name no crop are kept at a discount and chunks about other crops are pushed down."""
    ix = load()
    q = tokens(query)
    scored = []
    for i, d in enumerate(ix["docs"]):
        s = sum(ix["idf"].get(t, 0) * d[t] * (K1 + 1) / (d[t] + K1 * (1 - B + B * ix["lens"][i] / ix["avg"])) for t in q if t in d)
        if s and crop:
            c = ix["rows"][i]["crops"]
            s *= 1.0 if crop in c else 0.6 if not c else 0.25
        if s:
            scored.append((s, i))
    return [{**ix["rows"][i], "score": round(s, 2)} for s, i in sorted(scored, reverse=True)[:k]]
