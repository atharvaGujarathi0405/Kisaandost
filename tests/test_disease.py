import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bot import disease, server


def b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def textured(size=(400, 400), seed=0, base=120):
    """A sharp, mid-brightness image (random texture) so the photo checks pass."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(np.clip(rng.normal(base, 50, (*size[::-1], 3)), 0, 255).astype("uint8"))


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(disease, "predict", lambda crop, img, k=3: [("Rust", 0.80), ("Healthy", 0.12), ("Downy Mildew", 0.05)])


def test_photo_checks():
    assert disease.photo_problems(textured()) == []
    assert "small" in disease.photo_problems(textured((100, 100)))
    assert "dark" in disease.photo_problems(Image.new("RGB", (400, 400), (5, 5, 5)))
    assert "bright" in disease.photo_problems(Image.new("RGB", (400, 400), (250, 250, 250)))
    assert "blurry" in disease.photo_problems(Image.new("RGB", (400, 400), (120, 130, 110)))  # flat = no detail


def test_load_image_rejects_junk():
    for bad in ["not base64!!", base64.b64encode(b"hello world, not an image" * 10).decode()]:
        with pytest.raises(ValueError):
            disease.load_image(bad)
    big = base64.b64encode(b"\0" * (disease.MAX_BYTES + 1)).decode()
    with pytest.raises(ValueError):
        disease.load_image(big)


def test_load_image_accepts_data_url_and_resizes():
    img = disease.load_image("data:image/jpeg;base64," + b64(textured((3000, 2000))))
    assert max(img.size) == 1024 and img.mode == "RGB"


def test_ok_result_uses_guidance_and_names_rust():
    r = disease.diagnose("soybean", b64(textured()))
    assert r["status"] == "ok" and r["top"][0]["label"] == "Rust" and r["top"][0]["name_mr"].startswith("तांबेरा")
    assert r["advice"] and r["common"] and not r["healthy"]
    assert "फक्त शक्यता" in r["possible"]          # always framed as a possibility


def test_low_confidence_asks_for_a_better_photo(monkeypatch):
    monkeypatch.setattr(disease, "predict", lambda c, i, k=3: [("Rust", 0.30), ("Healthy", 0.29), ("Mosaic Virus", 0.2)])
    r = disease.diagnose("soybean", b64(textured()))
    assert r["status"] == "unsure" and "advice" not in r and r["top"]


def test_bad_photo_is_not_classified():
    r = disease.diagnose("orange", b64(Image.new("RGB", (400, 400), (5, 5, 5))))
    assert r["status"] == "bad_photo" and "dark" in r["problems"] and "top" not in r


def test_unsupported_crops_are_honest():
    for crop in ("tur", "cotton"):
        r = disease.diagnose(crop, b64(textured()))
        assert r["status"] == "unsupported" and "advice" not in r
    with pytest.raises(ValueError):
        disease.diagnose("wheat", b64(textured()))


def test_guidance_has_no_product_names_or_doses():
    import re
    text = " ".join(" ".join(e["mr"]) for crop in disease.GUIDE["crops"].values() for e in crop.values())
    assert not re.search(r"\d+\s*(मिली|ml|ग्रॅम|gm|g\b|लिटर|%|टक्के)", text, re.I)   # no doses
    for banned in ("कॉपर", "मॅन्कोझेब", "इमिडाक्लोप्रिड", "कार्बेन्डाझिम"):
        assert banned not in text


def test_every_classifier_label_has_guidance():
    # the labels the two models can output (from their model cards); an unlisted one would fall into "unsure"
    expect = {"soybean": {"Downy Mildew", "Frogeye Leaf Spot", "Healthy", "Mosaic Virus", "Rust", "Septoria Brown Spot"},
              "orange": {"Bacterial Blight", "Black Spot", "Canker", "Dry Leaf", "Healthy", "Leaf Curl Virus", "Sooty Mould", "Spider Mites"}}
    for crop, labels in expect.items():
        assert labels == set(disease.GUIDE["crops"][crop])


def test_endpoint():
    c = TestClient(server.app)
    r = c.post("/api/diagnose", json={"crop": "soybean", "image": b64(textured())})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert c.post("/api/diagnose", json={"crop": "soybean", "image": "A" * 200}).status_code == 400
    assert c.post("/api/diagnose", json={"crop": "wheat", "image": b64(textured())}).status_code == 400
    assert c.post("/api/diagnose", json={"crop": "soybean", "image": "short"}).status_code == 422


def test_orange_result_always_carries_the_greening_warning():
    r = disease.diagnose("orange", b64(textured()))          # fake model says Rust -> not an orange label -> unsure
    assert r["status"] == "unsure" and "ग्रीनिंग" in r["limits"]
    import pytest as _p
    _p.MonkeyPatch().setattr(disease, "predict", lambda c, i, k=3: [("Canker", 0.9), ("Black Spot", 0.05), ("Healthy", 0.02)])
    r = disease.diagnose("orange", b64(textured()))
    assert r["status"] == "ok" and "ग्रीनिंग" in r["limits"]


def test_healthy_is_never_a_clean_bill_of_health(monkeypatch):
    monkeypatch.setattr(disease, "predict", lambda c, i, k=3: [("Healthy", 0.95), ("Rust", 0.03), ("Mosaic Virus", 0.02)])
    r = disease.diagnose("soybean", b64(textured()))
    text = " ".join(r["advice"])
    assert r["healthy"] and "पक्के" in text and "निरोगी" not in r["top"][0]["name_mr"]


def test_confidence_bar_is_conservative():
    assert disease.MIN_CONF >= 0.6
