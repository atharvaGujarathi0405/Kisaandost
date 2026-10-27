"""Objective intelligibility check for the voices: speak held-out Luke sentences, transcribe them back with Groq Whisper, CER vs the text.

  python -m scripts.eval_tts --n 20                      generic MMS voice (HUB), fine-tuned voice (bot/models/tts), real narrator
  python -m scripts.eval_tts --dirs hub bot/models/tts

A proxy only: Whisper is a Marathi recogniser, and CER says nothing about how the voice SOUNDS. Listen to out/ too.
"""
import argparse
import io
import json
import os
import wave
from pathlib import Path

import jiwer

from bot import server, tts  # server loads .env

ROOT = Path(__file__).resolve().parent.parent
HELD = ROOT / "data" / "tts_train" / "luk_heldout"
OUT = ROOT / "data" / "tts_eval" / "voices"


def whisper(client, wav: bytes, name: str) -> str:
    r = client.audio.transcriptions.create(file=(name, wav), model="whisper-large-v3", language="mr", temperature=0)
    return r.text


def ogg_to_wav(p: Path) -> bytes:
    import subprocess
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "wav", "-"], capture_output=True, check=True).stdout


def cer(refs, hyps) -> float:
    return jiwer.cer(refs, [h or " " for h in hyps])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dirs", nargs="*", default=["hub", str(tts.ROOT / "models" / "tts")])
    a = ap.parse_args()
    import groq
    client = groq.Groq()
    items = json.loads((HELD / "texts.json").read_text(encoding="utf-8"))[: a.n]
    refs = [i["text"] for i in items]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"real narrator": cer(refs, [whisper(client, ogg_to_wav(HELD / f"{i['id']}.ogg"), "r.wav") for i in items])}
    for d in a.dirs:
        tts.LOCAL = Path(d if d != "hub" else "no-such-dir"); tts._state.clear(); tts._cache.clear()
        label = f"{tts.engine()}"
        hyps = []
        for k, i in enumerate(items):
            wav = tts.synthesize(i["text"])
            (OUT / f"{Path(d).name}_{k:02d}.wav").write_bytes(wav)
            hyps.append(whisper(client, wav, "s.wav"))
        res[label] = cer(refs, hyps)
    for k, v in res.items():
        print(f"{k:35s} CER {v:.3f}")


if __name__ == "__main__":
    main()
