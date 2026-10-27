"""Server-side text-to-speech: the bot's voice.

Default voice: facebook/mms-tts-mar (VITS, 36M parameters, CC BY-NC 4.0). It is a MARATHI voice reading Devanagari
text, so Varhadi spellings are pronounced the way Devanagari is normally read. It is NOT a Varhadi voice.
A voice fine-tuned on Varhadi speech would be dropped into bot/models/tts/ (KD_TTS_DIR) and used automatically;
none has been trained (see EXPERIMENT_LOG P6).

The model ignores punctuation, so we split the text ourselves and insert pauses. Output: 16 kHz mono 16-bit WAV.
"""
import hashlib
import io
import os
import re
import threading
import unicodedata
import wave
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LOCAL = Path(os.environ.get("KD_TTS_DIR", ROOT / "models" / "tts"))
HUB = "facebook/mms-tts-mar"
ENABLED = os.environ.get("KD_TTS", "1") != "0"
SAMPLE_RATE = 16000
PAUSE_MS = {"sentence": 380, "clause": 170}
MAX_CHUNK = 200            # characters per model call; longer clauses are cut at a space
CACHE_SIZE = 64

_state: dict = {}
_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()


def engine() -> str:
    return f"local:{LOCAL.name}" if (LOCAL / "model.safetensors").exists() else HUB


def load() -> None:
    with _lock:
        if _state:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel

        src = str(LOCAL) if (LOCAL / "model.safetensors").exists() else HUB
        _state["tok"] = AutoTokenizer.from_pretrained(src)
        _state["model"] = VitsModel.from_pretrained(src).eval()
        _state["torch"] = torch
        assert _state["model"].config.sampling_rate == SAMPLE_RATE, "unexpected sampling rate"


def clean(text: str, vocab) -> str:
    """Keep only characters the voice model knows (it has no digits or punctuation); collapse spaces."""
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    return re.sub(r"\s+", " ", "".join(c if (c in vocab or c == " ") else " " for c in text)).strip()


def split_chunks(text: str) -> list[tuple[str, int]]:
    """[(chunk, pause_after_ms)]: sentence ends get a long pause, commas and colons a short one."""
    out: list[tuple[str, int]] = []
    for sent in re.split(r"(?<=[.?!।])\s+|\n+", text):
        clauses = [c.strip() for c in re.split(r"(?<=[,;:–—])\s+", sent) if re.search(r"\w", c)]
        for ci, clause in enumerate(clauses):
            words, buf = clause.split(), ""
            pieces = []
            for w in words:                                   # cut over-long clauses at a space
                if buf and len(buf) + 1 + len(w) > MAX_CHUNK:
                    pieces.append(buf); buf = w
                else:
                    buf = f"{buf} {w}".strip()
            pieces.append(buf)
            for pi, piece in enumerate(pieces):
                last_piece, last_clause = pi == len(pieces) - 1, ci == len(clauses) - 1
                out.append((piece, PAUSE_MS["sentence"] if last_piece and last_clause else PAUSE_MS["clause"] if last_piece else 60))
    return out


def _generate(chunk: str) -> np.ndarray:
    load()
    tok, model, torch = _state["tok"], _state["model"], _state["torch"]
    chunk = clean(chunk, tok.get_vocab())
    if not chunk:
        return np.zeros(0, dtype=np.float32)
    x = tok(chunk, return_tensors="pt")
    with _lock, torch.no_grad():
        torch.manual_seed(0)                                  # VITS is stochastic: same text, same audio
        return model(**x).waveform[0].numpy().astype(np.float32)


def synthesize(text: str, generate=_generate) -> bytes:
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    parts = []
    for chunk, pause in split_chunks(text):
        w = generate(chunk)
        if len(w):
            parts += [w, np.zeros(int(SAMPLE_RATE * pause / 1000), dtype=np.float32)]
    if not parts:
        raise ValueError("nothing to say")
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SAMPLE_RATE); f.writeframes(pcm.tobytes())
    _cache[key] = buf.getvalue()
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return _cache[key]
