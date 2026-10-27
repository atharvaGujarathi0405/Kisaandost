"""Build a text-to-speech training set from the narrator recordings of ONE book (a single consistent voice).

  python -m scripts.build_tts_dataset --book LUK            (writes data/tts_train/<book>/ and zip parts for Colab upload)

Why one book: median pitch per chapter was about 194 Hz in John and 216 Hz in Luke (each book consistent inside itself),
so the two books may have different narrators; a single-speaker voice must be trained on one. The narrator is
"unknown" in our records. PRIVATE RESEARCH USE ONLY: the licence of the recordings is unverified and the voice belongs to
a real person; never publish the trained voice or the clips (data/ is gitignored).

Text is cleaned with bot.tts.clean() so training text looks exactly like inference text (no punctuation, no digits).
Clips are cut from the chapter MP3s, resampled to 16 kHz mono and stored as Ogg/Opus (small enough to upload to Colab).
Held-out chapters (default LUK_003, LUK_004 = the project's test/validation chapters) are NOT in the training set;
their text and the real recordings go to data/tts_train/<book>_heldout/ for an intelligibility check.
"""
import argparse
import csv
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "transcripts" / "gold_transcripts.csv"
OUT = ROOT / "data" / "tts_train"
MIN_S, MAX_S = 1.5, 20.0
PART_BYTES = 8_500_000        # the browser upload tool accepts about 10 MB per call


def vocab() -> set[str]:
    from transformers import AutoTokenizer

    return set(AutoTokenizer.from_pretrained("facebook/mms-tts-mar").get_vocab())


def select(book: str, held: set[str], min_conf: float, v: set[str]):
    from bot.tts import clean

    keep, hold, why = [], [], {}
    for r in csv.DictReader(open(GOLD, encoding="utf-8")):
        if not r["source_chapter"].startswith(book):
            continue
        d = float(r["end_sec"]) - float(r["start_sec"])
        t = r["candidate_text"]
        if not MIN_S <= d <= MAX_S:
            why["length"] = why.get("length", 0) + 1; continue
        if re.search(r"[()\[\]0-9०-९]", t) or "heading" in (r.get("triage_flags") or ""):
            why["brackets/digits/heading"] = why.get("brackets/digits/heading", 0) + 1; continue
        letters = [c for c in t if "ऀ" <= c <= "ॿ" and c.isalpha()]
        if any(c not in v for c in letters):
            why["character not in voice alphabet"] = why.get("character not in voice alphabet", 0) + 1; continue
        if r["triage_status"] != "HIGH_CONFIDENCE_CANDIDATE" and float(r["confidence_score"] or 0) < min_conf:
            why["low confidence"] = why.get("low confidence", 0) + 1; continue
        row = {"id": r["segment_id"], "chapter": r["source_chapter"], "audio": r["audio_path"], "start": float(r["start_sec"]),
               "dur": d, "text": clean(t, v)}
        (hold if r["source_chapter"] in held else keep).append(row)
    return keep, hold, why


def cut(row, dest: Path) -> None:
    src = ROOT / "audio" / Path(row["audio"]).name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{row['start']:.3f}", "-t", f"{row['dur']:.3f}", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k", str(dest)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="LUK")
    ap.add_argument("--held", default="LUK_003,LUK_004")
    ap.add_argument("--min-conf", type=float, default=0.85)
    a = ap.parse_args()
    keep, hold, why = select(a.book, set(a.held.split(",")), a.min_conf, vocab())
    print(f"train {len(keep)} clips {sum(r['dur'] for r in keep)/3600:.2f} h | held out {len(hold)} clips | dropped {why}")
    base = OUT / a.book.lower(); (base / "wavs").mkdir(parents=True, exist_ok=True)
    with open(base / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["file_name", "text"])
        for r in keep:
            dest = base / "wavs" / f"{r['id']}.ogg"; cut(r, dest); w.writerow([f"wavs/{dest.name}", r["text"]])
    hd = OUT / f"{a.book.lower()}_heldout"; hd.mkdir(exist_ok=True)
    for r in hold:
        cut(r, hd / f"{r['id']}.ogg")
    (hd / "texts.json").write_text(json.dumps([{"id": r["id"], "text": r["text"], "dur": round(r["dur"], 1)} for r in hold], ensure_ascii=False, indent=1), encoding="utf-8")
    # zip parts small enough for the browser upload tool
    files = [base / "metadata.csv"] + sorted((base / "wavs").iterdir()); part, size, n = [], 0, 0
    def flush():
        nonlocal part, size, n
        if part:
            z = OUT / f"{a.book.lower()}_part{n}.zip"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
                for p in part: zf.write(p, p.relative_to(base))
            print(z.name, z.stat().st_size // 1024, "KB"); part, size, n = [], 0, n + 1
    for p in files:
        if size + p.stat().st_size > PART_BYTES: flush()
        part.append(p); size += p.stat().st_size
    flush()


if __name__ == "__main__":
    main()
