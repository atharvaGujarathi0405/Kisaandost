"""Draft Varhadi versions of the Marathi farming sentences in data/agri_text/sources.tsv.

  python -m scripts.draft_agri_varhadi [--limit 5]       -> data/agri_text/varhadi_draft.tsv

Uses the same Marathi->Varhadi pipeline as the bot (fine-tuned ByT5 with its guards, then the corpus word rules; bot/server.to_varhadi).
Every row is a MACHINE DRAFT (status=machine_draft): no native speaker has checked it, and the model was trained on scripture text, not on
farming language. Native review (corrections, `reviewed_varhadi`, status=verified) is what turns this into a dataset. The source file is not
modified. Output stays under data/agri_text/ (gitignored: the sentences are Wikipedia text).
"""
import argparse
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "agri_text" / "sources.tsv"
OUT = ROOT / "data" / "agri_text" / "varhadi_draft.tsv"
FIELDS = ["id", "crop", "article", "marathi", "varhadi_draft", "changed_words", "status", "reviewed_varhadi", "reviewer"]


def changed(a: str, b: str) -> int:
    """Words that differ position by position (the two texts usually have the same length)."""
    x, y = a.split(), b.split()
    return sum(1 for p, q in zip(x, y) if p != q) + abs(len(x) - len(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from bot import mr2vh_model, server

    print("model available:", mr2vh_model.available(), flush=True)
    rows = list(csv.DictReader(open(SRC, encoding="utf-8"), delimiter="\t"))
    rows = rows[: a.limit] if a.limit else rows
    done = {}
    if OUT.exists() and not a.limit:                       # resume after an interruption
        done = {r["id"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"), delimiter="\t")}
    t0, out = time.time(), []
    for i, r in enumerate(rows):
        if r["id"] in done:
            out.append(done[r["id"]]); continue
        try:
            v = server.to_varhadi(r["marathi"])
        except Exception as e:                              # keep going; the row is marked and can be redone
            v = ""; print("failed", r["id"], type(e).__name__, flush=True)
        out.append({"id": r["id"], "crop": r["crop"], "article": r["article"], "marathi": r["marathi"], "varhadi_draft": v,
                    "changed_words": changed(r["marathi"], v) if v else "", "status": "machine_draft" if v else "failed",
                    "reviewed_varhadi": "", "reviewer": ""})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
            _save(out)
    _save(out)
    ok = [r for r in out if r["status"] == "machine_draft"]
    same = sum(1 for r in ok if r["varhadi_draft"].strip() == r["marathi"].strip())
    print(f"rows {len(out)}, drafted {len(ok)}, identical to Marathi {same}, avg changed words {sum(int(r['changed_words']) for r in ok) / max(len(ok), 1):.1f}")


def _save(rows) -> None:
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
