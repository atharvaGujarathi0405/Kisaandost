"""Does Gemini proofreading of the v5 Varhadi draft lower the error against the held-out Varhadi? (EXPERIMENT_LOG P12)

  python -m scripts.eval_refine                 # needs GEMINI_API_KEY in .env and data/translation_v5/test_*pred_v5.jsonl
Sets: test_new (119 held-out sentences from the user's datasets) and the 206 scripture test verses. Draft = exactly what the bot would
produce without Gemini (v5 prediction -> length guard -> attested-change filter -> negation check -> rules). The examples shown to
Gemini never include a test_new sentence.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "varhadi_adaptation"))
from asr_baseline.evaluate import _scores  # noqa: E402
from bot import mr2vh_model as M, refine, server  # noqa: E402

V5 = ROOT / "data" / "translation_v5"
GLOSS = {**server.FARM_RULES, **server.STYLE["enforce"], **server.STYLE["enforce_by_rule"]}


def rd(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def draft(r, allowed):
    src, p = r["source"], M.guard(r["source"], r["prediction"].strip(), None)
    if p is not None:
        p = M.constrain(src, p, allowed)
    return server.varhadify(p if p is not None and M.negation_ok(src, p) else server.varhadify(src))


def run(name, rows, variants, pool):
    allowed = M._allowed()
    pairs = [(r["source"], draft(r, allowed)) for r in rows]
    ref = [r["target"] for r in rows]
    res = {"draft (v5, no Gemini)": [d for _, d in pairs]}
    for label, use in variants:
        res[label] = refine.refine(pairs, GLOSS, pool=pool, use_examples=use)
    print(f"\n== {name}: {len(rows)} sentences")
    for k, h in res.items():
        s = _scores(ref, h)
        print(f"  {k:34s} WER {s['wer']:.3f}  CER {s['cer']:.3f}  sentences changed vs draft: {sum(a != b for a, b in zip(h, res['draft (v5, no Gemini)']))}")
    print(f"  Gemini calls failed so far: {refine.FAILED}" + ("   <-- THE NUMBERS ABOVE ARE NOT VALID" if refine.FAILED else ""))
    last = list(res.values())[-1]
    shown = 0
    for r, d, g in zip(rows, res["draft (v5, no Gemini)"], last):
        if d != g and shown < 6:
            print(f"   SRC {r['source']}\n   REF {r['target']}\n   DRAFT {d}\n   GEMINI {g}\n"); shown += 1


if __name__ == "__main__":
    new = rd(V5 / "test_new_pred_v5.jsonl")
    held = {r["source"] for r in new}
    run("test_new (user datasets, held out)", new, [("Gemini proofread, with examples", True), ("Gemini proofread, no examples", False)],
        [e for e in refine.examples() if e[0] not in held])
    run("scripture test", rd(V5 / "test_pred_v5.jsonl"), [("Gemini proofread, with examples", True)], refine.examples())
