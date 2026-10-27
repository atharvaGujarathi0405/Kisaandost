import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bot import server, tts


def fake_generate(chunk):
    return np.full(1600, 0.5, dtype=np.float32)       # 0.1 s per chunk


def duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000
        return f.getnframes() / f.getframerate()


@pytest.fixture(autouse=True)
def clear_cache():
    tts._cache.clear()


def test_split_gives_longer_pause_at_sentence_end_than_at_comma():
    parts = tts.split_chunks("पाणी द्या, खत टाका. उद्या या\nठीक आहे")
    assert [c for c, _ in parts] == ["पाणी द्या,", "खत टाका.", "उद्या या", "ठीक आहे"]
    assert [p for _, p in parts] == [170, 380, 380, 380]


def test_long_clause_is_cut_at_a_space():
    words = " ".join(["शब्द"] * 120)
    parts = tts.split_chunks(words)
    assert len(parts) > 1 and all(len(c) <= tts.MAX_CHUNK for c, _ in parts)
    assert " ".join(c for c, _ in parts) == words


def test_clean_drops_characters_the_model_does_not_know():
    vocab = set("पाणी द्या") | {" "}
    assert tts.clean("पाणी, द्या! 25 ‍", vocab) == "पाणी द्या"


def test_wav_has_the_right_format_and_pauses():
    wav = tts.synthesize("एक. दोन, तीन", generate=fake_generate)
    # 3 chunks x 0.1 s speech + pauses 0.38 (sentence) + 0.17 (clause) + 0.38 (end)
    assert duration(wav) == pytest.approx(0.3 + 0.38 + 0.17 + 0.38, abs=0.01)


def test_same_text_is_served_from_the_cache():
    calls = []
    tts.synthesize("पाणी द्या", generate=lambda c: calls.append(c) or fake_generate(c))
    tts.synthesize("पाणी द्या", generate=lambda c: calls.append(c) or fake_generate(c))
    assert len(calls) == 1


def test_text_with_nothing_speakable_is_an_error():
    with pytest.raises(ValueError):
        tts.synthesize("...", generate=fake_generate)


def test_endpoint(monkeypatch):
    c = TestClient(server.app)
    monkeypatch.setattr(tts, "synthesize", lambda text: b"RIFFfake")
    r = c.post("/api/tts", json={"text": "पाणी द्या"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFFfake"
    assert c.post("/api/tts", json={"text": ""}).status_code == 422
    assert c.post("/api/tts", json={"text": "अ" * 1501}).status_code == 422
    monkeypatch.setattr(tts, "synthesize", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    assert c.post("/api/tts", json={"text": "पाणी"}).status_code == 503
    monkeypatch.setattr(tts, "synthesize", lambda text: (_ for _ in ()).throw(ValueError("x")))
    assert c.post("/api/tts", json={"text": "पाणी"}).status_code == 422
    info = c.get("/api/tts/info").json()
    assert info["varhadi_voice"] is False and info["engine"]
