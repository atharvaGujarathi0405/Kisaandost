"""Marathi -> Varhadi sequence-to-sequence fine-tune (ByT5-small) and scoring.

Data: data/translation/{train,val,test}.jsonl from scripts/build_mr2vh_pairs.py (source = Marathi,
target = real VAHNT Varhadi). Splits are by chapter. NOT trained yet: see docs/varhadi_adaptation_research.md.

Colab (GPU runtime), after uploading the three jsonl files and this script to /content:
  !pip -q install "transformers>=4.44,<5" "accelerate>=0.33" sentencepiece
  !python mr2vh.py train   --data /content --out /content/mr2vh_model
  !python mr2vh.py predict --data /content --model /content/mr2vh_model --split test --out /content/test_pred.jsonl
Download test_pred.jsonl and score it on the PC (no torch needed there):
  python experiments/varhadi_adaptation/mr2vh.py score --data data/translation --pred test_pred.jsonl

Hyper-parameters are common ByT5 fine-tune defaults, not tuned. Measured on a Colab T4 (2026-10-26):
batch 4 (accum 4) ran out of memory at 14.5 GB within 5 steps, so the default is batch 2 x accum 8
(effective 16). About 4.1 s per optimizer step of 16 pairs at batch 4 before it failed.
"""
import argparse
import json
import sys
from pathlib import Path

MODEL = "google/byt5-small"
PREFIX = "mr2vh: "
MAX_BYTES = 1024  # ByT5 tokens are bytes, Devanagari is 3 bytes/char, so this is ~340 characters


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def train(a) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = [r for r in rows if len(r["source"].encode()) < MAX_BYTES - 16 and len(r["target"].encode()) < MAX_BYTES - 16]
            print(f"  kept {len(self.rows)}/{len(rows)} pairs under {MAX_BYTES} bytes")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            x = tok(PREFIX + r["source"], max_length=MAX_BYTES, truncation=True)
            x["labels"] = tok(text_target=r["target"], max_length=MAX_BYTES, truncation=True)["input_ids"]
            return x

    d = Path(a.data)
    tr, va = Pairs(read(d / "train.jsonl")), Pairs(read(d / "val.jsonl"))
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=20, report_to="none",
        warmup_ratio=0.05, weight_decay=0.01, seed=13,
    )
    def collate_fn(feats):  # pad labels with -100 so padding is ignored by the loss
        labels = [f.pop("labels") for f in feats]
        batch = tok.pad(feats, return_tensors="pt")
        width = max(map(len, labels))
        batch["labels"] = torch.tensor([l + [-100] * (width - len(l)) for l in labels])
        return batch

    Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=collate_fn).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)


def predict(a) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    rows = read(Path(a.data) / f"{a.split}.jsonl")
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            x = tok([PREFIX + r["source"] for r in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_BYTES).to(dev)
            with torch.no_grad():
                out = model.generate(**x, max_new_tokens=MAX_BYTES, num_beams=4)
            for r, p in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                f.write(json.dumps({**r, "prediction": p}, ensure_ascii=False) + "\n")
            print(f"  {min(i + a.batch, len(rows))}/{len(rows)}", file=sys.stderr)


def score(a) -> None:
    """WER/CER against the real VAHNT Varhadi for: no change, rules only (varhadify), fine-tuned model."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from asr_baseline.evaluate import _scores
    from bot.mr2vh_model import guard  # same guard and thresholds the bot uses
    from bot.server import varhadify

    pred = {r["ref"]: r["prediction"] for r in read(Path(a.pred))}
    rows = [r for r in read(Path(a.data) / "test.jsonl") if r["ref"] in pred]
    ref = [r["target"] for r in rows]
    systems = {"no change (Marathi as is)": [r["source"] for r in rows],
               "rules only (varhadify)": [varhadify(r["source"]) for r in rows],
               "fine-tuned ByT5": [pred[r["ref"]] for r in rows],
               "fine-tuned + rules": [varhadify(pred[r["ref"]]) for r in rows],
               "fine-tuned, length-guarded": [guard(r["source"], pred[r["ref"]], varhadify(r["source"])) for r in rows]}
    guarded = sum(guard(r["source"], pred[r["ref"]], None) is None for r in rows)
    print(f"  length guard fell back to rules on {guarded}/{len(rows)} verses")
    print(f"{len(rows)} held-out verses from chapters {sorted({r['chapter'] for r in rows})}  (PILOT: no significance test)")
    for name, hyp in systems.items():
        s = _scores(ref, hyp)
        print(f"  {name:28s} WER {s['wer']:.3f}  CER {s['cer']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=2); t.add_argument("--accum", type=int, default=8)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True); p.add_argument("--model", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True); s.add_argument("--pred", required=True)
    a = ap.parse_args()
    {"train": train, "predict": predict, "score": score}[a.cmd](a)
