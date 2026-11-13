"""Training data for the next Marathi -> Varhadi fine-tune (v4): the real scripture pairs PLUS synthetic farm pairs.

  python -m scripts.build_mr2vh_v4_data      # writes data/translation_v4/{train,val,test}.jsonl and mr2vh_v4_data.zip

train = real VAHNT pairs (unchanged) + synthetic pairs; val and test are the SAME real held-out chapters as before, so
v4 and v3 are scored on identical verses. Synthetic pairs (target = the rule-based Varhadi, NOT native text):
  farm      each farm sentence -> varhadify(sentence)  (grammar rules + farm KB + dictionary swaps)
  word      carrier sentences with every dictionary swap and every standard farm term (the latter map to themselves)
They teach the model the swaps and to leave farm terms alone; they cannot add Varhadi that no dictionary or text contains.
"""
import csv
import json
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import server  # noqa: E402  (varhadify = the same rules the bot applies)
from scripts.build_farm_kb import FARM_TERMS  # noqa: E402

SRC, OUT = ROOT / "data" / "translation", ROOT / "data" / "translation_v4"
CARRIERS = ["हे {} आहे.", "तुमच्याकडे {} आहे का?", "मी {} पाहिले.", "शेतात {} दिसते."]


def read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pair(tag, i, src):
    return {"ref": f"{tag}{i}", "chapter": tag, "source": src, "target": server.varhadify(src)}


def main():
    OUT.mkdir(exist_ok=True)
    farm = [r["marathi"].strip() for r in csv.DictReader(open(ROOT / "datasets" / "mr2vh_farming_dataset.csv", encoding="utf-8-sig"))
            if r["marathi"].strip() and len(r["marathi"].encode()) < 900]
    terms = {w.strip() for r in csv.DictReader(open(FARM_TERMS, encoding="utf-8-sig"))
             for w in re.split(r"[/,]", re.sub(r"\(.*?\)", "", r["marathi"])) if w.strip()}
    words = sorted(set(server.FARM_RULES) - {"मग"}) + sorted(terms)
    synth = [pair("farm", i, s) for i, s in enumerate(farm)]
    synth += [pair("word", n, c.format(w)) for n, (w, c) in enumerate((w, c) for w in words for c in CARRIERS)]
    train = read(SRC / "train.jsonl") + synth
    random.Random(13).shuffle(train)
    for name, rows in (("train", train), ("val", read(SRC / "val.jsonl")), ("test", read(SRC / "test.jsonl"))):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with zipfile.ZipFile(ROOT / "data" / "mr2vh_v4_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("train", "val", "test"):
            z.write(OUT / f"{name}.jsonl", f"{name}.jsonl")
    changed = sum(p["source"] != p["target"] for p in synth)
    print(f"real train {len(train) - len(synth)} + synthetic {len(synth)} (farm {len(farm)}, carrier {len(synth) - len(farm)}; "
          f"{changed} differ from their source) = {len(train)}")


if __name__ == "__main__":
    main()
