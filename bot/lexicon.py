"""Word-by-word Marathi -> Varhadi replacement learned from aligned sentence pairs (scripts/build_word_lexicon.py).

Two layers, both mined from real pairs (scripture + the user's sentence datasets), kept only above thresholds tuned on validation data:
  spans    whole words and 2-3 word phrases ("मी" -> "म्या", "पडत आहेत" -> "पडतायत"), exact match on consecutive words
  suffixes ending swaps seen on many different words ("-ावे" -> "-ावं"), applied to words the lexicon does not know
Applied last, to Marathi words that the model and the hand-checked rules left unchanged. Local file (the pairs come from restricted
text): bot/models/lexicon.json, gitignored. KD_LEXICON=0 turns it off.
"""
import json
import os
import re
from pathlib import Path

PATH = Path(os.environ.get("KD_LEXICON_FILE", Path(__file__).resolve().parent / "models" / "lexicon.json"))
_WORD = re.compile("[ऀ-ॿ]+")
_lex: dict = {}


def load() -> dict:
    if not _lex:
        _lex.update(json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {"spans": {}, "suffixes": [], "protect": []})
    return _lex


def available() -> bool:
    return os.environ.get("KD_LEXICON", "1") != "0" and bool(load()["spans"] or load()["suffixes"])


def apply(text: str, lex: dict | None = None) -> str:
    lex = lex or load()
    spans, suffixes, protect = lex["spans"], lex["suffixes"], set(lex.get("protect", []))
    ms = list(_WORD.finditer(text))
    out, last, i = [], 0, 0
    while i < len(ms):
        hit = None
        for n in (3, 2, 1):
            grp = ms[i:i + n]
            if len(grp) < n or any(text[a.end():b.start()] != " " for a, b in zip(grp, grp[1:])):
                continue
            key = " ".join(m.group(0) for m in grp)
            if key in spans:
                hit = (n, spans[key])
                break
        if not hit:
            w = ms[i].group(0)
            for sx, sy in suffixes:           # longest suffix first (the file is sorted that way)
                if w.endswith(sx) and len(w) - len(sx) >= 2 and w not in protect:
                    hit = (1, w[:-len(sx)] + sy)
                    break
        if hit:
            n, repl = hit
            out.append(text[last:ms[i].start()] + repl)
            last = ms[i + n - 1].end()
            i += n
        else:
            i += 1
    return "".join(out) + text[last:]
