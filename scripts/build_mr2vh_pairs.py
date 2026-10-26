"""Build Marathi -> Varhadi training pairs by back-translating the local VAHNT verses.

  python -m scripts.build_mr2vh_pairs --limit 12     # inspect a few first
  python -m scripts.build_mr2vh_pairs                # all verses, resumable, then writes the splits

Target side = real published Varhadi (VAHNT). Source side = plain Marathi written by a Groq model from
that Varhadi, so the pair direction matches the bot: LLM writes Marathi -> model produces Varhadi.
Sends VAHNT text to Groq (the user approved this on 2026-10-26; VAHNT licence is unverified, research only).
Output stays in data/translation/ (gitignored).

Splits are by CHAPTER, never by verse. Test/val chapters never reach training.
"""
import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import groq

from bot import server  # loads .env and gives us the shared Varhadi marker list

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "transcripts" / "vahnt_text"
OUT = ROOT / "data" / "translation"
# Only 120b by default: on 8 test verses gpt-oss-20b and qwen3.8-27b changed meaning (wrong subject, dropped phrases).
MODELS = os.environ.get("KD_BT_MODELS", "openai/gpt-oss-120b").split(",")
PROMPT_VERSION = "bt-v1"
BATCH = 8
TEST = ["JHN_001", "JHN_005", "LUK_003", "MAT_005", "MRK_002"]  # JHN_001 = project test chapter (P2_READY.md)
VAL = ["JHN_002", "LUK_004", "MAT_006", "MRK_003"]              # JHN_002 = project val chapter
# Varhadi-only forms: if one survives in the "Marathi" side, the model copied instead of translating.
MARKERS = {"नाई", "हाय", "हायत", "हायस", "तुमी", "आमी", "मले", "तुले", "त्याले", "तिले", "तुमाले", "आमाले",
           "त्यायले", "त्यायनं", "तवा", "जवा", "मंग", "कावून", "अन्", "पायजे", "म्हतलं", "देलं", "लय", "माह्या", "तुह्या"}
INSTR = (
    "Each numbered item is a sentence in Varhadi, a dialect of Marathi from Vidarbha. Rewrite each one in plain, "
    "standard, spoken Marathi (not Varhadi, no other dialect). Keep the meaning, names, numbers and sentence "
    "structure; change only dialect words and forms. Do not shorten, summarise, explain or add. Answer with one "
    "line per item, exactly `[[n]] <marathi sentence>`, nothing else."
)


def chapter_of(ref: str) -> str:
    """'JHN.3.2' -> 'JHN_003' (the stored chapter_id looks like 'JHN.3.VAHNT', which the split lists don't use)."""
    book, ch = re.match(r"([A-Z0-9]{3})\.(\d+)\.", ref).groups()
    return f"{book}_{int(ch):03d}"


def verses() -> list[dict]:
    out = []
    for f in sorted(TEXT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out += [{"ref": u["ref"], "chapter": chapter_of(u["ref"]), "varhadi": u["text"].strip()}
                for u in d["units"] if u["kind"] == "verse" and u["text"].strip()]
    return out


def done() -> dict[str, dict]:
    p = OUT / "backtrans.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l] if p.exists() else []
    return {r["ref"]: r for r in rows if r.get("prompt_version") == PROMPT_VERSION}


_blocked_until: dict[str, float] = {}  # model -> time.time() when its rate limit lifts


def _extra(model: str) -> dict:
    return {"reasoning_effort": "low", "include_reasoning": False} if "gpt-oss" in model else {"reasoning_effort": "none"}


def translate(client: groq.Groq, batch: list[dict]) -> tuple[str, dict[int, str]]:
    """Try each model in turn; a model that is rate limited is skipped until its retry-after passes.

    Groq's free tier caps each model separately (daily), so rotating keeps the job moving."""
    body = "\n".join(f"[[{i}]] {v['varhadi']}" for i, v in enumerate(batch, 1))
    for _ in range(200):
        ready = [m for m in MODELS if _blocked_until.get(m, 0) <= time.time()]
        if not ready:
            wait = min(_blocked_until.values()) - time.time()
            print(f"  all models rate limited, sleeping {wait:.0f}s")
            time.sleep(max(wait, 1))
            continue
        model = ready[0]
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_completion_tokens=4096, **_extra(model),
                messages=[{"role": "system", "content": INSTR}, {"role": "user", "content": body}])
            text = r.choices[0].message.content or ""
            return model, {int(n): s.strip() for n, s in re.findall(r"\[\[(\d+)\]\]\s*(.+)", text)}
        except groq.RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 20)) + 1
            _blocked_until[model] = time.time() + wait
            print(f"  {model} rate limited for {wait:.0f}s")
        except groq.APIConnectionError:
            time.sleep(5)
    raise RuntimeError("Groq kept failing; rerun to resume")


def backtranslate(limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = groq.Groq()
    have = done()
    todo = [v for v in verses() if v["ref"] not in have][:limit]
    print(f"{len(have)} done, {len(todo)} to do, models {MODELS}")
    with open(OUT / "backtrans.jsonl", "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            model, got = translate(client, batch)
            for n, v in enumerate(batch, 1):
                if n in got:
                    f.write(json.dumps({**v, "marathi": got[n], "model": model, "prompt_version": PROMPT_VERSION},
                                       ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}  ({len(got)}/{len(batch)} parsed)")


def keep(r: dict) -> str | None:
    """Return a reason to drop the pair, or None to keep it."""
    m, v = r["marathi"], r["varhadi"]
    if re.search(r"[A-Za-z]", m) and not re.search(r"[A-Za-z]", v):
        return "latin"
    if not 0.6 <= len(m) / len(v) <= 1.6:
        return "length"
    if set(re.findall(r"[ऀ-ॿ]+", m)) & MARKERS:
        return "copied_varhadi"
    if m == v:
        return "identical"
    return None


def build_splits() -> None:
    rows, dropped = list(done().values()), Counter()
    split = {"train": [], "val": [], "test": []}
    for r in rows:
        r["chapter"] = chapter_of(r["ref"])  # also fixes rows written before chapter_of existed
        why = keep(r)
        if why:
            dropped[why] += 1
            continue
        s = "test" if r["chapter"] in TEST else "val" if r["chapter"] in VAL else "train"
        split[s].append({"ref": r["ref"], "chapter": r["chapter"], "source": r["marathi"], "target": r["varhadi"]})
    for s, items in split.items():
        (OUT / f"{s}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    print(f"back-translated {len(rows)}; dropped {dict(dropped)}; " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))
    leak = {c for s in ("train",) for c in {x["chapter"] for x in split[s]}} & set(TEST + VAL)
    assert not leak, f"held-out chapter in train: {leak}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--splits-only", action="store_true")
    a = ap.parse_args()
    if not a.splits_only:
        backtranslate(a.limit)
    build_splits()
