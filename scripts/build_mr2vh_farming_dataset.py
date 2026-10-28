"""Build one Marathi -> Varhadi farming dataset with the adaptation layer (the same two stages the bot uses).

  python -m scripts.build_mr2vh_farming_dataset      -> data/agri_text/mr2vh_farming_dataset.csv

Marathi sources: the Wikipedia farming sentences (data/agri_text/sources.tsv), the Marathi disease advice the bot shows
(bot/disease_guidance.json) and the Marathi recording prompts (data/prompts/recording_prompts.csv).
Stage 1 = fine-tuned ByT5 model with its guards (bot/mr2vh_model.py), stage 2 = word rules (bot/server.varhadify), so each row shows
what the model changed and what the rules changed. Every row is a MACHINE DRAFT: no native speaker has checked it, and the model was
trained on scripture, not on farming language. `reviewed_varhadi` / `reviewer` stay empty until a native speaker fills them.
"""
import csv
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "agri_text" / "mr2vh_farming_dataset.csv"
FIELDS = ["id", "source", "crop", "marathi", "model_output", "varhadi", "model_changed", "rules_changed", "words_changed",
          "flag", "status", "reviewed_varhadi", "reviewer"]
LATIN = re.compile(r"[A-Za-z0-9]")
REFERENCE = re.compile(r"Archived|बीज उपचारकी घरेलू")


def words_changed(a: str, b: str) -> int:
    """Words replaced, added or removed (alignment based, so a shifted word is not counted as many)."""
    x, y = a.split(), b.split()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y).get_opcodes() if op != "equal")


def marathi_rows() -> list[dict]:
    rows = [{"id": r["id"], "source": "wikipedia_agriculture", "crop": r["crop"], "marathi": r["marathi"]}
            for r in csv.DictReader(open(ROOT / "data" / "agri_text" / "sources.tsv", encoding="utf-8"), delimiter="\t")]
    g = json.load(open(ROOT / "bot" / "disease_guidance.json", encoding="utf-8"))
    for crop, classes in g["crops"].items():
        for label, e in classes.items():
            for i, s in enumerate(e["mr"]):
                rows.append({"id": f"guide_{crop}_{label.replace(' ', '_')}_{i}", "source": "disease_guidance", "crop": crop, "marathi": s})
    rows.append({"id": "guide_common", "source": "disease_guidance", "crop": "general", "marathi": g["common_mr"]})
    rows += [{"id": f"guide_limits_{c}", "source": "disease_guidance", "crop": c, "marathi": s} for c, s in g["limits_mr"].items()]
    rows += [{"id": r["prompt_id"], "source": "recording_prompts", "crop": r["crop"], "marathi": r["text"]}
             for r in csv.DictReader(open(ROOT / "data" / "prompts" / "recording_prompts.csv", encoding="utf-8-sig"))]
    return rows


def main() -> None:
    from bot import mr2vh_model, server

    if not mr2vh_model.available():                       # without the model this would silently be rules only
        raise SystemExit("mr2vh model files not found in bot/models: refusing to build a rules-only dataset")
    out = []
    for i, r in enumerate(marathi_rows()):
        text = server.tidy(r["marathi"])
        model_out = mr2vh_model.translate_text(text, fallback=server.varhadify)
        final = server.varhadify(model_out)
        out.append({**r, "marathi": text, "model_output": model_out, "varhadi": final,
                    "model_changed": words_changed(text, model_out), "rules_changed": words_changed(model_out, final),
                    "words_changed": words_changed(text, final),
                    "flag": "wikipedia_reference_line" if REFERENCE.search(text) else "latin_or_digits" if LATIN.search(text) else "",
                    "status": "machine_draft", "reviewed_varhadi": "", "reviewer": ""})
        if (i + 1) % 50 == 0:
            print(i + 1, flush=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(out)
    same = sum(1 for r in out if r["varhadi"].strip() == r["marathi"].strip())
    print(f"rows {len(out)}; identical to Marathi {same}; model changed {sum(1 for r in out if r['model_changed'])}; "
          f"rules changed {sum(1 for r in out if r['rules_changed'])}")
    for s in sorted({r["source"] for r in out}):
        print(" ", s, sum(1 for r in out if r["source"] == s))


if __name__ == "__main__":
    main()
