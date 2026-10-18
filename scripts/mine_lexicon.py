#!/usr/bin/env python3
"""Mine candidate (published Varhadi form, Marathi-ASR rendering) pairs from the gold candidates.

    python scripts/mine_lexicon.py
    python scripts/mine_lexicon.py --triage-filter HIGH_CONFIDENCE_CANDIDATE --min-count 5

Reads data/transcripts/gold_transcripts.csv only (no model, no network). The output stays in
data/transcripts/ (gitignored: VAHNT licence unverified, remote is public). `standard_marathi`
is always empty and `verified` always false; see varhadi_data/lexicon.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from varhadi_data import lexicon
from varhadi_data.config import DATA_DIR

TRIAGE = ["HIGH_CONFIDENCE_CANDIDATE", "NEEDS_REVIEW", "REJECT_CANDIDATE"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-consistency", type=float, default=0.5)
    parser.add_argument("--triage-filter", choices=TRIAGE, help="only mine segments with this triage status")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "transcripts/lexicon_candidates.tsv")
    args = parser.parse_args(argv)

    rows, stats = lexicon.mine(lexicon.load_pairs(triage_filter=args.triage_filter))
    by_count = [r for r in rows if r["count"] >= args.min_count]
    by_cons = [r for r in rows if r["consistency"] >= args.min_consistency]
    kept = [r for r in by_count if r["consistency"] >= args.min_consistency]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=lexicon.FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)

    print(f"segments processed:      {stats['segments']}" + (f" (triage={args.triage_filter})" if args.triage_filter else ""))
    print(f"substitutions found:     {stats['substitutions']}")
    print(f"skipped (not paired):    {stats['skipped']}")
    print(f"unique varhadi forms:    {stats['unique_forms']}")
    print(f"unique pairs:            {stats['unique_pairs']}")
    print(f"  count >= {args.min_count}:            {len(by_count)}")
    print(f"  consistency >= {args.min_consistency}:    {len(by_cons)}")
    print(f"  both (rows written):   {len(kept)}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
