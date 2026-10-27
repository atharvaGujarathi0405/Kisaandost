"""Precompute the Varhadi advice in bot/disease_guidance.json with the trained Marathi->Varhadi model.

  python -m scripts.build_disease_guidance          (about 10 minutes on CPU; run it when nothing else is using the CPU)

Why precompute: replies are instant, and a native speaker can read and correct exactly what farmers will be shown
(edit the `vh` lines in the JSON; this script keeps a `vh` line you changed by hand only if you pass --keep-edits).
"""
import argparse
import json
from pathlib import Path

from bot import server

PATH = Path(__file__).resolve().parent.parent / "bot" / "disease_guidance.json"


def main(keep_edits: bool) -> None:
    g = json.loads(PATH.read_text(encoding="utf-8"))
    done = 0
    for crop in g["crops"].values():
        for e in crop.values():
            fresh = [server.to_varhadi(x) for x in e["mr"]]
            if keep_edits and e.get("vh") and e.get("vh_edited"):
                continue
            e["vh"] = fresh
            done += 1
            print(".", end="", flush=True)
    g["common_vh"] = server.to_varhadi(g["common_mr"])
    g["limits_vh"] = {k: server.to_varhadi(v) for k, v in g["limits_mr"].items()}
    PATH.write_text(json.dumps(g, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{done} entries converted -> {PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-edits", action="store_true", help="skip entries with vh_edited: true")
    main(ap.parse_args().keep_edits)
