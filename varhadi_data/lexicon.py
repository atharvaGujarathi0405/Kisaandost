"""Candidate lexicon mining: (published Varhadi form, Marathi-ASR rendering) pairs.

Each gold row holds the PUBLISHED VAHNT text (`candidate_text`) and what the
standard-Marathi baseline heard from the same span (`raw_asr`). A token substitution
between them is an ERROR OBSERVATION: the ASR rendering is neither Varhadi nor reliably
standard Marathi. So `standard_marathi` is ALWAYS left empty for a native speaker to fill,
and `verified` is always false. Nothing here is gold or a dialect mapping.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from asr_baseline.utils import strip_punctuation

from . import gold
from .config import DATA_DIR

SOURCE = "mined-vahnt-v1"
FIELDS = ["varhadi_form", "asr_rendering", "standard_marathi", "count", "n_segments", "n_chapters",
          "consistency", "example_segment_ids", "triage", "verified", "source"]


def load_pairs(data_dir: Path = DATA_DIR, triage_filter: str | None = None) -> list[dict]:
    """Gold rows that have both published text and a baseline prediction for the same span.
    The baseline output for a span is stored on the same row (`raw_asr`), so the join is per row."""
    return [row for row in gold.read_gold(data_dir)
            if row["candidate_text"].strip() and row["raw_asr"].strip()
            and (triage_filter is None or row["triage_status"] == triage_filter)]


def align_tokens(a: str, b: str) -> tuple[list[tuple[str, str]], Counter]:
    """1:1 token substitutions between published text `a` and ASR `b` (punctuation stripped,
    nothing else normalised). Returns (pairs, skipped) where skipped counts insert, delete
    and unequal-length replace blocks (word merges/splits), which are NOT paired."""
    ta, tb = strip_punctuation(a).split(), strip_punctuation(b).split()
    pairs, skipped = [], Counter()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op in ("insert", "delete"):
            skipped[op] += 1
        elif op == "replace" and i2 - i1 != j2 - j1:
            skipped["unequal_replace"] += 1  # ponytail: n:m blocks dropped; pair them if merges/splits matter
        elif op == "replace":
            pairs += [(x, y) for x, y in zip(ta[i1:i2], tb[j1:j2]) if x != y]
    return pairs, skipped


def mine(rows: list[dict]) -> tuple[list[dict], dict]:
    """Aggregate substitutions over all rows, ranked by count desc, then consistency desc."""
    count, segs, chaps, triage, examples = Counter(), defaultdict(set), defaultdict(set), defaultdict(Counter), defaultdict(list)
    skipped = Counter()
    for row in rows:
        pairs, skip = align_tokens(row["candidate_text"], row["raw_asr"])
        skipped += skip
        for pair in pairs:
            count[pair] += 1
            segs[pair].add(row["segment_id"])
            chaps[pair].add(row["source_chapter"])
            triage[pair][row["triage_status"] or "UNTRIAGED"] += 1
            if row["segment_id"] not in examples[pair] and len(examples[pair]) < 3:
                examples[pair].append(row["segment_id"])
    per_form = Counter()
    for (form, _), n in count.items():
        per_form[form] += n
    out = [
        {
            "varhadi_form": form,
            "asr_rendering": asr,
            "standard_marathi": "",  # NEVER filled here: native-speaker input only
            "count": n,
            "n_segments": len(segs[form, asr]),
            "n_chapters": len(chaps[form, asr]),
            "consistency": round(n / per_form[form], 3),
            "example_segment_ids": ";".join(examples[form, asr]),
            "triage": ";".join(f"{k}:{v}" for k, v in triage[form, asr].most_common()),
            "verified": "false",
            "source": SOURCE,
        }
        for (form, asr), n in count.items()
    ]
    out.sort(key=lambda r: (-r["count"], -r["consistency"], r["varhadi_form"], r["asr_rendering"]))
    stats = {"segments": len(rows), "substitutions": sum(count.values()), "skipped": dict(skipped),
             "unique_forms": len(per_form), "unique_pairs": len(count)}
    return out, stats
