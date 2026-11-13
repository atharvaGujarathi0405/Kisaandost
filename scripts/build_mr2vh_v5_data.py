"""Training data for the v5 Marathi -> Varhadi fine-tune: real scripture pairs + the user-supplied sentence datasets.

  python -m scripts.build_mr2vh_v5_data      # writes data/translation_v5/{train,val,test,test_new}.jsonl, data/mr2vh_v5_data.zip,
                                             # experiments/varhadi_adaptation/mr2vh_kaggle_v5.ipynb (gitignored: embeds the data)

Sources (datasets/, columns = [Varhadi, standard Marathi]; provenance and native verification unknown, treat as unverified):
  varhadi_sentences.csv, kisaandost_varhadi_agriculture_queries_300.csv, kisaandost_varhadi_general_600_sentences.csv
Pairs where the two sides are identical are dropped (they teach the model to copy; v4 showed that). Splits are by TEMPLATE GROUP
(first 5 Marathi words), so near-duplicate variants never straddle train and test. scripture val/test are unchanged;
test_new = held-out new pairs, scored separately. New train pairs are repeated NEW_REPEAT times (they are about a quarter of the data).
"""
import base64
import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data" / "translation", ROOT / "data" / "translation_v5"
FILES = ["varhadi_sentences.csv", "kisaandost_varhadi_agriculture_queries_300.csv", "kisaandost_varhadi_general_600_sentences.csv"]
NEW_REPEAT = 2


def read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def new_pairs():
    seen, out, dropped = set(), [], 0
    for f in FILES:
        for i, row in enumerate(list(csv.reader(open(ROOT / "datasets" / f, encoding="utf-8-sig")))[1:]):
            v, m = (row + ["", ""])[:2]
            v, m = v.strip(), m.strip()
            if not v or not m or v == m or (m, v) in seen:
                dropped += 1
                continue
            seen.add((m, v))
            out.append({"ref": f"{f[:6]}{i}", "chapter": f[:6], "source": m, "target": v, "group": " ".join(m.split()[:5])})
    return out, dropped


def bucket(group):  # deterministic: <10 test, <15 val, else train
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 100


def main():
    OUT.mkdir(exist_ok=True)
    pairs, dropped = new_pairs()
    test_new = [p for p in pairs if bucket(p["group"]) < 10]
    val_new = [p for p in pairs if 10 <= bucket(p["group"]) < 15]
    train_new = [p for p in pairs if bucket(p["group"]) >= 15]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "group"} for r in rows]
    train = read(SRC / "train.jsonl") + strip(train_new) * NEW_REPEAT
    random.Random(13).shuffle(train)
    splits = {"train": train, "val": read(SRC / "val.jsonl") + strip(val_new), "test": read(SRC / "test.jsonl"), "test_new": strip(test_new)}
    for n, rows in splits.items():
        (OUT / f"{n}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    z = ROOT / "data" / "mr2vh_v5_data.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in splits:
            zf.write(OUT / f"{n}.jsonl", f"{n}.jsonl")
    print(f"new pairs {len(pairs)} (dropped {dropped} identical/empty/duplicate): train {len(train_new)} x{NEW_REPEAT}, val {len(val_new)}, test_new {len(test_new)}; "
          f"scripture train {len(train) - len(train_new) * NEW_REPEAT}; train total {len(train)}")
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
    run = "cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 python mr2vh.py "
    cells = [
        md("# Marathi -> Varhadi ByT5 fine-tune v5 (scripture pairs + user sentence datasets)\nKaggle GPU, Internet ON. Data embedded (VAHNT text + user datasets); keep PRIVATE."),
        code("import os\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n!nvidia-smi -L\n!pip -q install \"transformers>=4.44,<5\" \"accelerate>=0.33\" sentencepiece"),
        code("import base64, os\nos.makedirs('/kaggle/working/data', exist_ok=True)\nopen('/kaggle/working/d.zip','wb').write(base64.b64decode('" + base64.b64encode(z.read_bytes()).decode() + "'))\n!cd /kaggle/working && unzip -oq d.zip -d data && wc -l data/*.jsonl"),
        code("%%writefile /kaggle/working/mr2vh.py\n" + (ROOT / "experiments/varhadi_adaptation/mr2vh.py").read_text(encoding="utf-8")),
        code("!" + run + "train --data data --out mr2vh_model --epochs 5 --batch 2 --accum 8 2>&1 | grep -v Warning | tail -60"),
        code("!" + run + "predict --data data --model mr2vh_model --split test --out test_pred.jsonl 2>&1 | tail -2\n!" + run + "predict --data data --model mr2vh_model --split test_new --out test_new_pred.jsonl 2>&1 | tail -2"),
        code("!cd /kaggle/working/mr2vh_model && zip -q -0 ../mr2vh_final.zip model.safetensors config.json generation_config.json tokenizer_config.json special_tokens_map.json added_tokens.json && ls -la /kaggle/working/mr2vh_final.zip /kaggle/working/test_pred.jsonl /kaggle/working/test_new_pred.jsonl && rm -rf /kaggle/working/mr2vh_model /kaggle/working/data /kaggle/working/d.zip"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "experiments/varhadi_adaptation/mr2vh_kaggle_v5.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
