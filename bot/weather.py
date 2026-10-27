"""Weather for the farmer's place: Open-Meteo (free, no key; https://open-meteo.com, data CC BY 4.0), cached 30 minutes.

Weather is looked up at a point (the village when OpenStreetMap has it, else the taluka centre). The model grid is about 10 km,
so a village and its neighbour get the same numbers; that is honest enough for a farm-planning hint, not for a spray decision.
"""
import datetime
import json
import time
import urllib.parse
import urllib.request

URL = "https://api.open-meteo.com/v1/forecast"
TTL = 1800
WEEKDAYS = ("सोमवार", "मंगळवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार")
_cache: dict = {}


def fetch(lat: float, lon: float, opener=urllib.request.urlopen) -> dict:
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    q = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "timezone": "Asia/Kolkata", "forecast_days": 4,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    })
    with opener(f"{URL}?{q}", timeout=8) as r:
        d = json.load(r)
    c, dy = d["current"], d["daily"]
    out = {
        "now": {"temp": c["temperature_2m"], "humidity": c["relative_humidity_2m"]},
        "days": [{"date": dy["time"][i], "tmax": dy["temperature_2m_max"][i], "tmin": dy["temperature_2m_min"][i],
                  "rain_mm": dy["precipitation_sum"][i] or 0, "rain_pct": dy["precipitation_probability_max"][i] or 0}
                 for i in range(len(dy["time"]))],
    }
    _cache[key] = (time.time(), out)
    return out


def advice(w: dict) -> list[str]:
    """Plain rules on the next two days; general practice only, never a product or dose."""
    out, days = [], w["days"][:2]
    if any(d["rain_mm"] >= 10 or d["rain_pct"] >= 70 for d in days):
        out.append("पाऊस येण्याची शक्यता हाय, फवारणी आन् खत टाकणं पावसानंतर करा. शेतात पाणी साचू देऊ नका.")
    if any(d["tmax"] >= 40 for d in days):
        out.append("खूप ऊन हाय, सकाळी किंवा सांजच्याला पाणी द्या.")
    if any(d["tmin"] <= 10 for d in days):
        out.append("थंडी वाढली हाय, कोवळ्या रोपायचं रक्षण करा.")
    return out


def summary(w: dict) -> str:
    """One text for both the weather card and the model prompt (Marathi words, numbers as digits)."""
    n, lines = w["now"], []
    lines.append(f"आता: {n['temp']:.0f}°C, हवेतला दमटपणा {n['humidity']:.0f}%")
    for i, d in enumerate(w["days"]):
        label = ("आज", "उद्या", "परवा")[i] if i < 3 else WEEKDAYS[datetime.date.fromisoformat(d["date"]).weekday()]
        lines.append(f"{label}: {d['tmin']:.0f}-{d['tmax']:.0f}°C, पावसाची शक्यता {d['rain_pct']:.0f}%, पाऊस {d['rain_mm']:.0f} मिमी")
    return "\n".join(lines)
