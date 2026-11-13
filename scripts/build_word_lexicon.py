"""Mine the word-by-word replacement lexicon (bot/lexicon.py), tune it on validation data, test it on held-out sentences.

  python -m scripts.build_word_lexicon            # report only (mined from TRAIN pairs, tuned on val, scored on test and test_new)
  python -m scripts.build_word_lexicon --write    # then mine from train+val+test_new and write bot/models/lexicon.json
Validation drafts are rules-only (the model has no val predictions); test drafts are the real v5 output through the bot's filter layer.
"""
import argparse
import collections
import difflib
import itertools
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "varhadi_adaptation"))
from asr_baseline.evaluate import _scores  # noqa: E402
from bot import lexicon, mr2vh_model, server  # noqa: E402
from scripts.eval_refine import V5, draft, rd  # noqa: E402

W = re.compile("[ऀ-ॿ]+")
OUT = ROOT / "bot" / "models" / "lexicon.json"
FARM = Path(os.environ.get("KD_FARM_SAMPLE", ""))  # optional: JSON [[marathi, current bot output], ...] to see the effect on farm text


def uniq(rows):
    seen, out = set(), []
    for r in rows:
        if (r["source"], r["target"]) not in seen:
            seen.add((r["source"], r["target"]))
            out.append(r)
    return out


def mine(rows):
    """span pairs (source words -> target words) with counts, n-gram occurrence counts, token counts."""
    pairs, occ, tok = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        a, b = W.findall(r["source"]), W.findall(r["target"])
        tok.update(a)
        for n in (1, 2, 3):
            for i in range(len(a) - n + 1):
                occ[tuple(a[i:i + n])] += 1
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "replace" and 0 < i2 - i1 <= 3 and 0 < j2 - j1 <= 3 and abs((i2 - i1) - (j2 - j1)) <= 1:
                pairs[(tuple(a[i1:i2]), tuple(b[j1:j2]))] += 1
    return pairs, occ, tok


_mined: dict = {}


def build(rows, min_count, min_share):
    if id(rows) not in _mined:  # the sweep builds 36 lexicons from the same rows: mine them once
        _mined[id(rows)] = mine(rows)
    pairs, occ, tok = _mined[id(rows)]
    best = {}
    for (x, y), c in pairs.items():
        if c >= min_count and c / occ[x] >= min_share and x != y and (x not in best or c > best[x][1]):
            best[x] = (" ".join(y), c)
    spans = {" ".join(x): v[0] for x, v in best.items()}
    return {"spans": spans, "suffixes": [], "protect": sorted(w for w in tok if w in spans)}


def suffixes(rows, share, stems_min=5):
    """Ending swaps seen on many different stems, as a share of all running words with that ending. Mined from the SPOKEN-style rows
    only: the scripture keeps "-ते" 99% of the time while the user's datasets change it to "-तं" 82% of the time, so mixing them
    would learn neither."""
    if id(rows) not in _mined:
        _mined[id(rows)] = mine(rows)
    pairs, _, tok = _mined[id(rows)]
    suf, stems = collections.Counter(), collections.defaultdict(set)
    for (x, y), c in pairs.items():
        if len(x) == len(y) == 1:
            p0 = len(os.path.commonprefix([x[0], y[0]]))
            for p in (p0, p0 - 1, p0 - 2):          # the changed tail alone ("े -> ं") and with 1-2 letters before it ("ते -> तं")
                if p >= 2 and x[0][p:] != y[0][p:]:
                    suf[(x[0][p:], y[0][p:])] += c
                    stems[(x[0][p:], y[0][p:])].add(x[0][:p])
    rules = []
    for (sx, sy), c in suf.items():
        total = sum(n for w, n in tok.items() if w.endswith(sx) and len(w) - len(sx) >= 2)
        if sx and len(stems[(sx, sy)]) >= stems_min and total and c / total >= share and len(sx) <= 5:
            rules.append([sx, sy])
    return sorted(rules, key=lambda r: -len(r[0]))


def wer(ref, hyp):
    s = _scores(ref, hyp)
    return s["wer"], s["cer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    train, val = uniq(rd(V5 / "train.jsonl")), rd(V5 / "val.jsonl")
    ref_v, base_v = [r["target"] for r in val], [server.varhadify(r["source"]) for r in val]
    print(f"train {len(train)}  val {len(val)}  rules-only val WER {wer(ref_v, base_v)[0]:.3f}")
    print("\nsweep on VALIDATION (rules-only drafts + word/phrase lexicon)")
    results = []
    for c, s in itertools.product((1, 2, 3), (0.3, 0.5, 0.7, 0.9)):
        lex = build(train, c, s)
        results.append((wer(ref_v, [lexicon.apply(t, lex) for t in base_v])[0], c, s, len(lex["spans"])))
    for w, c, s, ns in sorted(results)[:5]:
        print(f"  min_count={c} min_share={s}: WER {w:.3f}  spans {ns}")
    _, c, s, _ = min(results)
    print(f"chosen on validation: min_count={c} min_share={s}")
    lex = build(train, c, s)
    spoken = [r for r in train if not re.match(r"[A-Z]{3}\.", r["ref"])]   # the user's spoken-style datasets only (see suffixes())
    vnew = [r for r in val if not re.match(r"[A-Z]{3}\.", r["ref"])]
    rv = [r["target"] for r in vnew]
    best = (wer(rv, [lexicon.apply(server.varhadify(r["source"]), lex) for r in vnew])[0], 9.9)
    print(f"\nsuffix share sweep, validation = {len(vnew)} sentences from the user's datasets (spans only: WER {best[0]:.3f})")
    for ss in (0.4, 0.5, 0.6, 0.7, 0.8):
        cand = {**lex, "suffixes": suffixes(spoken, ss)}
        w = wer(rv, [lexicon.apply(server.varhadify(r["source"]), cand) for r in vnew])[0]
        print(f"  suffix share >= {ss}: WER {w:.3f}  rules {len(cand['suffixes'])}")
        best = min(best, (w, ss))
    ss = best[1]
    lex["suffixes"] = suffixes(spoken, ss) if ss <= 1 else []
    print("chosen suffix share", ss if ss <= 1 else "none", "-> rules:", lex["suffixes"])
    allowed = mine_allowed = mr2vh_model._allowed()
    for name, path in (("TEST new (119 held-out sentences, user datasets)", "test_new_pred_v5.jsonl"), ("TEST scripture (206 held-out verses)", "test_pred_v5.jsonl")):
        rows = rd(V5 / path)
        ref, d = [r["target"] for r in rows], [draft(r, allowed) for r in rows]
        only_spans = {**lex, "suffixes": []}
        print(f"\n{name}")
        for label, h in (("v5 draft (no lexicon)", d), ("+ word/phrase lexicon", [lexicon.apply(t, only_spans) for t in d]),
                         ("+ lexicon + suffix rules", [lexicon.apply(t, lex) for t in d])):
            print(f"  {label:26s} WER {wer(ref, h)[0]:.3f}  CER {wer(ref, h)[1]:.3f}")
    if FARM.is_file():
        farm = json.loads(FARM.read_text(encoding="utf-8"))
        tot = ch0 = ch1 = 0
        for src, out in farm:
            a_, b_, c_ = W.findall(src), W.findall(out), W.findall(lexicon.apply(out, lex))
            tot += len(a_); ch0 += sum(x != y for x, y in zip(a_, b_)) if len(a_) == len(b_) else 0
            ch1 += sum(x != y for x, y in zip(a_, c_)) if len(a_) == len(c_) else 0
        print(f"\nFARM sentences ({len(farm)}): words changed {100 * ch0 / tot:.1f}% now -> {100 * ch1 / tot:.1f}% with the lexicon")
    if a.write:
        allrows = uniq(rd(V5 / "train.jsonl") + rd(V5 / "val.jsonl") + rd(V5 / "test_new.jsonl"))
        full = build(allrows, c, s)
        full["suffixes"] = suffixes([r for r in allrows if not re.match(r"[A-Z]{3}\.", r["ref"])], ss) if ss <= 1 else []
        OUT.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {OUT}: {len(full['spans'])} spans, {len(full['suffixes'])} suffix rules (mined from train+val+test_new)")


if __name__ == "__main__":
    main()
