"""Crop photo -> POSSIBLE disease + safe advice in Varhadi.

Models: ConvNeXt-Small leaf classifiers by IXR (agritechixr), CC BY 4.0, trained on LeafNet and the authors' photos:
  soybean -> ixrbhii/soybean-disease-convnext   (6 classes)      orange -> ixrbhii/citrus-disease-convnext   (8 classes)
Cotton uses bot/models/cotton/ if that folder exists (not trained yet). Tur has no model and no public dataset.
A closed-set classifier always names one of its own classes, so a disease is reported only above MIN_CONF; below it
we ask for a better photo. The advice comes from bot/disease_guidance.json (unverified, no product names or doses).
"""
import base64
import binascii
import io
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
GUIDE = json.loads((ROOT / "disease_guidance.json").read_text(encoding="utf-8"))
HF_MODELS = {"soybean": "ixrbhii/soybean-disease-convnext", "orange": "ixrbhii/citrus-disease-convnext"}
COTTON_DIR = ROOT / "models" / "cotton"
CROP_MR = {"cotton": "कापूस", "soybean": "सोयाबीन", "orange": "संत्रा", "tur": "तूर"}
MIN_CONF = 0.60          # raised from the model card's 0.45 after the independent test (EXPERIMENT_LOG P4): precision on known diseases 73% -> 76% (soybean), 88% -> 91% (orange)
MAX_BYTES = 6_000_000    # decoded upload
MAX_PIXELS = 25_000_000
MIN_SIDE = 160           # px
DARK, BRIGHT = 45, 240   # mean grey level (128 public photos ranged 76-226)
BLUR = 15.0              # variance of the Laplacian on a 256 px grey copy: no sharp public photo was below 16.7; blur radius 4 mostly falls below it

_cache: dict = {}
_lock = threading.Lock()


def supported(crop: str) -> bool:
    return crop in HF_MODELS or (crop == "cotton" and (COTTON_DIR / "model.safetensors").exists())


def load_image(data: str) -> Image.Image:
    """Decode a base64 (or data-URL) photo. Raises ValueError with a short reason; nothing is stored."""
    if "," in data[:80]:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("not a valid image upload")
    if len(raw) > MAX_BYTES:
        raise ValueError("photo is too large")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_PIXELS:
            raise ValueError("photo is too large")
        img = ImageOps.exif_transpose(img).convert("RGB")  # the photo lives in memory only: it is never saved or logged
    except (OSError, Image.DecompressionBombError):
        raise ValueError("not a valid image")
    img.thumbnail((1024, 1024))
    return img


def photo_problems(img: Image.Image) -> list[str]:
    """Cheap checks that explain most bad predictions: too small, too dark or bright, out of focus."""
    out = []
    if min(img.size) < MIN_SIDE:
        out.append("small")
    g = np.asarray(ImageOps.grayscale(img).resize((256, 256)), dtype=np.float32)
    if g.mean() < DARK:
        out.append("dark")
    elif g.mean() > BRIGHT:
        out.append("bright")
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if lap.var() < BLUR:
        out.append("blurry")
    return out


def _model(crop: str):
    with _lock:
        if crop not in _cache:
            import timm  # imported late: torch is heavy and only needed once a photo arrives

            name = f"local-dir:{COTTON_DIR}" if crop == "cotton" else f"hf-hub:{HF_MODELS[crop]}"
            m = timm.create_model(name, pretrained=True).eval()
            _cache[crop] = (m, timm.data.create_transform(**timm.data.resolve_data_config({}, model=m)),
                            m.pretrained_cfg["label_names"])
        return _cache[crop]


def predict(crop: str, img: Image.Image, k: int = 3) -> list[tuple[str, float]]:
    import torch

    m, tf, labels = _model(crop)
    with torch.no_grad():
        p = m(tf(img).unsqueeze(0)).softmax(-1)[0]
    return [(labels[i], float(p[i])) for i in p.argsort(descending=True)[:k]]


MSG = {  # standard Marathi; run through the word rules into Varhadi
    "unsupported": "या पिकासाठी फोटोवरून रोग ओळखणारे मॉडेल अजून नाही. लक्षणे लिहून विचारा किंवा पान कृषी सहाय्यकाला दाखवा.",
    "bad_photo": "फोटो नीट आला नाही. पान जवळून, दिवसाच्या उजेडात आणि स्थिर हाताने पुन्हा काढा.",
    "unsure": "मला खात्री वाटत नाही. एकाच पानाचा जवळून, स्पष्ट फोटो पुन्हा काढा, किंवा कृषी सहाय्यकाला दाखवा.",
    "reason_small": "फोटो खूप लहान आहे.", "reason_dark": "फोटोत अंधार आहे.", "reason_bright": "फोटोत खूप उजेड आहे.",
    "reason_blurry": "फोटो हलला आहे किंवा स्पष्ट नाही.",
    "possible": "ही फक्त शक्यता आहे, पक्के निदान नाही.",
}


def _vh(text: str) -> str:
    from bot.server import varhadify  # late import: bot.server imports this module

    return varhadify(text)


def diagnose(crop: str, data: str) -> dict:
    if crop not in CROP_MR:
        raise ValueError("unknown crop")
    base = {"crop": crop, "crop_mr": CROP_MR[crop], "possible": _vh(MSG["possible"])}
    if not supported(crop):
        return {**base, "status": "unsupported", "message": _vh(MSG["unsupported"])}
    img = load_image(data)
    problems = photo_problems(img)
    if problems:
        why = " ".join(MSG["reason_" + p] for p in problems)
        return {**base, "status": "bad_photo", "message": _vh(why + " " + MSG["bad_photo"]), "problems": problems}
    top = predict(crop, img)
    entries = GUIDE["crops"][crop]
    items = [{"label": lab, "name_mr": entries.get(lab, {}).get("name_mr", lab),
              "name_en": lab, "p": round(p, 3)} for lab, p in top]
    if top[0][1] < MIN_CONF or top[0][0] not in entries:
        return {**base, "status": "unsure", "top": items, "message": _vh(MSG["unsure"]),
                "limits": GUIDE["limits_vh"].get(crop) or _vh(GUIDE["limits_mr"][crop])}
    e = entries[top[0][0]]
    lines = e.get("vh") or [_vh(x) for x in e["mr"]]
    return {**base, "status": "ok", "top": items, "healthy": top[0][0] == "Healthy", "advice": lines,
            "common": GUIDE.get("common_vh") or _vh(GUIDE["common_mr"]),
            "limits": GUIDE["limits_vh"].get(crop) or _vh(GUIDE["limits_mr"][crop])}
