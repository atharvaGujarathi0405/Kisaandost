"""Marathi -> Varhadi ByT5 translator, fine-tuned on VAHNT pairs (EXPERIMENT_LOG.md P3a, pilot).

Loaded lazily and optional: without the model files (KD_MR2VH_DIR, default bot/models/mr2vh/) the bot uses the
word-swap rules only. torch/transformers are imported only when the model is actually used.
Model files are gitignored: VAHNT licence is unverified, so the fine-tuned weights are not published.
"""
import collections
import difflib
import json
import os
import re
import sys
import threading
from pathlib import Path

MODEL_DIR = Path(os.environ.get("KD_MR2VH_DIR", Path(__file__).resolve().parent / "models" / "mr2vh"))
BEAMS = int(os.environ.get("KD_MR2VH_BEAMS", "1"))  # 1 vs 4 beams: 24-verse CPU check WER 0.343 vs 0.338 at half the time
PREFIX = "mr2vh: "  # must match experiments/varhadi_adaptation/mr2vh.py
MAX_BYTES = 1024
# Length guard: target/source character ratio over the TRAINING pairs was min 0.77, p99 1.27, max 1.52.
# Outside it the output is not trusted (the pilot's repetition loops were all far above 1.52).
RATIO_MIN, RATIO_MAX = 0.77, 1.52

# Words that flip a farm instruction if lost or added (नका = do not). Counts must survive conversion.
NEGATORS = {"नाही", "नाई", "नका", "नको", "नये", "नाहीत", "नव्हता", "नव्हती", "नव्हते", "नसेल", "नसते", "नसतो"}
# Suffix swaps the data-derived list proposed but a person judged wrong (-ने is the agent, -ले the dative).
SUFFIX_VETO = {("ने", "ले")}

_state: dict = {}
_lock = threading.Lock()


def guard(source: str, prediction: str, fallback: str) -> str:
    ok = RATIO_MIN <= len(prediction) / max(len(source), 1) <= RATIO_MAX
    return prediction if ok else fallback


def available() -> bool:
    return os.environ.get("KD_MR2VH", "1") != "0" and (MODEL_DIR / "model.safetensors").exists()


def load() -> None:
    with _lock:
        if _state:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        _state["model"] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).eval()
        _state["torch"] = torch
        torch.set_num_threads(os.cpu_count() or 4)


def _generate(sentences: list[str]) -> list[str]:
    load()
    tok, model, torch = _state["tok"], _state["model"], _state["torch"]
    x = tok([PREFIX + s for s in sentences], return_tensors="pt", padding=True, truncation=True, max_length=MAX_BYTES)
    longest = max(len(s.encode()) for s in sentences)
    with _lock, torch.no_grad():  # one generation at a time: CPU-bound, and torch modules are not request-safe
        out = model.generate(**x, max_new_tokens=min(MAX_BYTES, int(longest * 1.6) + 8), num_beams=BEAMS)
    return tok.batch_decode(out, skip_special_tokens=True)


def _core(t: str) -> str:
    return re.sub(r"[^ऀ-ॿ\w]", "", t)


def _ops(a: list[str], b: list[str]):
    return difflib.SequenceMatcher(None, [_core(x) for x in a], [_core(x) for x in b], autojunk=False).get_opcodes()


def build_allowed(train_jsonl: Path, out: Path = None, min_count: int = 2) -> dict:
    """Word substitutions (Marathi -> Varhadi) seen at least min_count times in the TRAINING pairs, plus the
    suffix swaps they imply. On the pilot test set, model substitutions found in this list matched the real
    Varhadi 71% of the time, unlisted ones 21% (and several changed the meaning)."""
    pairs = collections.Counter()
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a, b = r["source"].split(), r["target"].split()
        for op, i1, i2, j1, j2 in _ops(a, b):
            if op == "replace" and i2 - i1 == j2 - j1:
                pairs.update((_core(x), _core(y)) for x, y in zip(a[i1:i2], b[j1:j2]))
    allowed = sorted(p for p, c in pairs.items() if c >= min_count)
    suf = collections.Counter((x[-2:], y[-2:]) for x, y in allowed if x[:-2] == y[:-2] and len(x) > 3)
    data = {"pairs": [list(p) for p in allowed],
            "suffixes": sorted(list(p) for p, c in suf.items() if c >= 3 and p not in SUFFIX_VETO)}
    if out:
        out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def constrain(source: str, out: str, allowed: dict) -> str:
    """Keep the model's word changes only if they were seen in training; otherwise keep the source word.

    Inserted, deleted or merged words are kept as the model wrote them (checked only by negation_ok)."""
    pairs = {tuple(p) for p in allowed["pairs"]}
    suffixes = {tuple(p) for p in allowed["suffixes"]}
    a, b = source.split(), out.split()
    res: list[str] = []
    for op, i1, i2, j1, j2 in _ops(a, b):
        if op == "replace" and i2 - i1 == j2 - j1:
            for s, t in zip(a[i1:i2], b[j1:j2]):
                cs, ct = _core(s), _core(t)
                ok = (cs, ct) in pairs or (cs[:-2] == ct[:-2] and (cs[-2:], ct[-2:]) in suffixes)
                res.append(t if ok else s)
        else:
            res += b[j1:j2]
    return " ".join(res)


def negation_ok(source: str, out: str) -> bool:
    """Negation words and digits must not be lost or invented (the rules map नाही -> नाई, so compare after that)."""
    count = lambda t: collections.Counter(w for w in re.findall(r"[ऀ-ॿ]+|\d+", t) if w in NEGATORS or w.isdigit())
    norm = lambda t: t.replace("नाही", "नाई")
    return count(norm(source)) == count(norm(out))


def _allowed() -> dict | None:
    f = MODEL_DIR / "allowed_subs.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def translate_text(text: str, fallback, generate=_generate) -> str:
    """Translate sentence by sentence (training pairs were single verses), guarding each one.

    fallback(sentence) -> str is used when the output fails the length guard or the negation check (the word-swap rules).
    The model's word changes are filtered through constrain() first."""
    parts = re.split(r"(?<=[.?!।])(\s+)", text)
    todo = [i for i, p in enumerate(parts) if re.search("[ऀ-ॿ]", p) and len(p) <= (MAX_BYTES - 16) // 3]
    if not todo:
        return text
    allowed = _allowed()
    if allowed is None:  # without the training-derived list the model cannot be trusted word by word
        return text
    for i, pred in zip(todo, generate([parts[i] for i in todo])):
        src, pred = parts[i], pred.strip()
        pred = guard(src, pred, None)
        pred = None if pred is None else constrain(src, pred, allowed)
        parts[i] = pred if pred is not None and negation_ok(src, pred) else fallback(src)
    return "".join(parts)


if __name__ == "__main__":  # python -m bot.mr2vh_model data/translation/train.jsonl
    d = build_allowed(Path(sys.argv[1]), MODEL_DIR / "allowed_subs.json")
    print(len(d["pairs"]), "substitution pairs,", len(d["suffixes"]), "suffix swaps ->", MODEL_DIR / "allowed_subs.json")
