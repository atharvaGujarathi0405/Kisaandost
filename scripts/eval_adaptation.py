"""Measure the Marathi -> Varhadi layer (bot.server.varhadify) against the local VAHNT gold text.

  python -m scripts.eval_adaptation            (from the repo root)

Three checks, all on local gitignored data, none needs a model or an API key:
  1. closeness   WER of standard-Marathi ASR vs the VAHNT text, before and after varhadify().
                 Lower after = the layer moves standard Marathi toward published Varhadi.
  2. leakage     varhadify() applied to VAHNT itself. It should change almost nothing; whatever it
                 changes is a rule false positive or a place VAHNT uses the standard form.
  3. agri audit  tokens varhadify() changes in standard-Marathi farm text (data/agri_text/sources.tsv),
                 so verb or adjective false positives can be spotted by eye.
  4. coverage    -ला / -ांना tokens in farm text that varhadify() leaves alone: candidates for new rules, but
                 many are verbs (आला, झाला), so a person decides.
Caveat: the rules were chosen from this same VAHNT data, so (1) is an in-sample, optimistic number, and
VAHNT is scripture, not farm talk. Only a native speaker's review is real validation.
"""
import csv
import re
from collections import Counter
from pathlib import Path

from asr_baseline.evaluate import _scores
from asr_baseline.utils import strip_punctuation
from bot.server import varhadify
from varhadi_data import gold

ROOT = Path(__file__).resolve().parent.parent
AGRI = ROOT / "data" / "agri_text" / "sources.tsv"


def changed(before: str, after: str) -> list[tuple[str, str]]:
    return [(a, b) for a, b in zip(strip_punctuation(before).split(), strip_punctuation(after).split()) if a != b]


def main() -> None:
    rows = [r for r in gold.read_gold() if r["candidate_text"].strip() and r["raw_asr"].strip()]
    ref = [r["candidate_text"] for r in rows]
    asr = [r["raw_asr"] for r in rows]
    before, after = _scores(ref, asr), _scores(ref, [varhadify(t) for t in asr])
    print(f"1. closeness ({len(rows)} segments, {before['reference_words']} ref words)")
    print(f"   WER vs VAHNT: raw ASR {before['wer']:.3f} -> varhadify {after['wer']:.3f}   "
          f"CER {before['cer']:.3f} -> {after['cer']:.3f}")

    leaks = Counter(pair for t in ref for pair in changed(t, varhadify(t)))
    n_tok = sum(len(strip_punctuation(t).split()) for t in ref)
    print(f"2. leakage: {sum(leaks.values())} of {n_tok} VAHNT tokens changed; top: {leaks.most_common(12)}")

    if AGRI.exists():
        sents = [r["marathi"] for r in csv.DictReader(open(AGRI, encoding="utf-8"), delimiter="\t")]
        ch = Counter(pair for s in sents for pair in changed(s, varhadify(s)))
        n = sum(len(strip_punctuation(s).split()) for s in sents)
        print(f"3. agri audit: {sum(ch.values())} of {n} tokens in {len(sents)} sentences changed "
              f"({len(ch)} distinct)")
        for (a, b), k in ch.most_common(40):
            print(f"   {a} -> {b}  x{k}")
        left = Counter(w for s in sents for w in strip_punctuation(varhadify(s)).split() if re.search(r"(ला|ांना)$", w))
        print(f"4. coverage: {sum(left.values())} unconverted -ला/-ांना tokens ({len(left)} distinct): {left.most_common(40)}")


if __name__ == "__main__":
    main()
