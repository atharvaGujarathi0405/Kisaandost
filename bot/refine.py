"""Gemini proofreads the Varhadi draft that our fine-tuned model + rules produced (EXPERIMENT_LOG P12 has the measurement).

For each sentence Gemini sees the standard-Marathi source, our draft, the attested word list and a few real Varhadi example pairs
from the user's sentence datasets that look most like the sentence. It may only fix dialect forms and grammar/agreement; every
result is checked (negations and digits unchanged, no Roman letters, sane length) and the draft is kept when a check fails or
Gemini is unreachable. Needs GEMINI_API_KEY. OFF by default (KD_REFINE=1 turns it on) until EXPERIMENT_LOG P12 shows it helps: it costs one more Gemini request per answer.
"""
import csv
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

from bot import agent, mr2vh_model, rag

ROOT = Path(__file__).resolve().parent.parent
FILES = ["varhadi_sentences.csv", "kisaandost_varhadi_agriculture_queries_300.csv", "kisaandost_varhadi_general_600_sentences.csv"]
K, BATCH, MAX_EXAMPLES = 4, 12, 30
_pool: list = []
FAILED = 0  # Gemini calls that failed since start; scripts/eval_refine.py prints it so a quota error cannot pass for "no change"

INSTR = """You proofread Varhadi, the Marathi dialect of Vidarbha, for farmers.
Each numbered item gives SOURCE (standard Marathi) and DRAFT (a program's Varhadi version of it, may contain mistakes).
Return the corrected Varhadi for every item.
- Keep the meaning, numbers, names, negations (नाही -> नाई, नका) and sentence order exactly as in SOURCE. Do not add, drop or explain anything.
- Fix mistakes in the DRAFT: wrong gender/number agreement (for example a neuter form with a masculine plural noun), a swapped word that changes the meaning, broken endings.
- Use Varhadi forms from the WORD LIST and the EXAMPLES. If you are not sure that a Varhadi form is real, keep the DRAFT word or the SOURCE word. Never invent a word.
- Devanagari only, no Roman letters, no quotes added.
Answer with one line per item, exactly `[[n]] <corrected Varhadi sentence>`, nothing else."""


def enabled() -> bool:
    return os.environ.get("KD_REFINE", "0") == "1" and agent.enabled()


def examples() -> list[tuple[str, str, Counter]]:
    """(marathi, varhadi, tokens) from the user's sentence datasets; identical pairs are skipped. Empty when the files are absent."""
    if not _pool:
        seen = set()
        for f in FILES:
            p = ROOT / "datasets" / f
            if not p.exists():
                continue
            for row in list(csv.reader(open(p, encoding="utf-8-sig")))[1:]:
                v, m = (row + ["", ""])[:2]
                v, m = v.strip(), m.strip()
                if v and m and v != m and (m, v) not in seen:
                    seen.add((m, v))
                    _pool.append((m, v, Counter(rag.tokens(m))))
    return _pool


def nearest(src: str, pool, k=K, exclude=()) -> list[tuple[str, str]]:
    q = Counter(rag.tokens(src))
    scored = sorted(((sum((q & d).values()) / math.sqrt(sum(d.values()) or 1), m, v) for m, v, d in pool if m not in exclude), reverse=True)
    return [(m, v) for s, m, v in scored[:k] if s > 0]


def _ok(src: str, out: str) -> bool:
    return bool(out) and mr2vh_model.negation_ok(src, out) and not re.search("[A-Za-z]", out.replace(src, "")) \
        and mr2vh_model.RATIO_MIN <= len(out) / max(len(src), 1) <= mr2vh_model.RATIO_MAX


def refine(pairs: list[tuple[str, str]], glossary: dict, client=None, pool=None, use_examples=True) -> list[str]:
    """pairs = [(source Marathi, draft Varhadi)]; returns the proofread sentences (the draft where Gemini failed or a check failed)."""
    from google.genai import types

    pool = examples() if pool is None else pool
    client = client or agent.make_client()
    words = ", ".join(f"{k}->{v}" for k, v in glossary.items())
    out = [d for _, d in pairs]
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        shots = []
        for src, _ in chunk if use_examples else []:
            shots += [e for e in nearest(src, pool, exclude={s for s, _ in chunk}) if e not in shots]
        ex = "\n".join(f"Marathi: {m}\nVarhadi: {v}" for m, v in shots[:MAX_EXAMPLES])
        body = "\n".join(f"[[{n}]] SOURCE: {s} | DRAFT: {d}" for n, (s, d) in enumerate(chunk, 1))
        text = f"{INSTR}\n\nWORD LIST (standard -> Varhadi): {words}\n\nEXAMPLES of real Varhadi:\n{ex or '(none)'}\n\nITEMS:\n{body}"
        try:
            r = agent.generate(client, [types.Content(role="user", parts=[types.Part(text=text)])],
                               types.GenerateContentConfig(temperature=0.1, max_output_tokens=4096))
            got = {int(n): t.strip() for n, t in re.findall(r"\[\[(\d+)\]\]\s*(.+)", r.text or "")}
        except Exception as e:  # Gemini down or over quota: keep our drafts, but count and log it
            global FAILED
            FAILED += 1
            logging.warning("refine: Gemini call failed (%s %s)", type(e).__name__, getattr(e, "code", ""))
            continue
        for n, (src, _) in enumerate(chunk, 1):
            if n in got and _ok(src, got[n]):
                out[i + n - 1] = got[n]
    return out


def _split(t: str) -> list[str]:
    return re.split(r"(?<=[.?!।])(\s+)", t)


def refine_text(source: str, draft: str, glossary: dict, client=None) -> str:
    """Sentence-wise refine of a whole reply. Needs the same sentence split in source and draft, else the draft is returned."""
    sp, dp = _split(source), _split(draft)
    if len(sp) != len(dp):
        return draft
    idx = [i for i, p in enumerate(sp) if re.search("[ऀ-ॿ]", p) and re.search("[ऀ-ॿ]", dp[i])]
    for i, r in zip(idx, refine([(sp[i].strip(), dp[i].strip()) for i in idx], glossary, client)):
        dp[i] = r
    return "".join(dp)
