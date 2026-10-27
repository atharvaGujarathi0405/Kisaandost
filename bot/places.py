"""Where the farmer is: Vidarbha district -> taluka -> village (bot/places.json, built by scripts/build_vidarbha_places.py).

Villages come from OpenStreetMap contributors (ODbL) and are INCOMPLETE; a village that is not in the list is accepted as typed text
and the weather falls back to the taluka centre. Zone facts are from PDKV Akola's technology inventory (agro-climatic zones).
Soil is stated only in general terms: field-level soil needs the farmer's Soil Health Card.
"""
import functools
import json
import re
from pathlib import Path

FILE = Path(__file__).with_name("places.json")
# PDKV Akola: three rainfall zones. Rainfall = normal annual range; crops = what dominates the kharif season there.
ZONES = {
    "Western Vidarbha": {"mr": "पश्चिम वऱ्हाड", "districts": ["Amravati", "Akola", "Buldhana", "Washim"], "rain": "700-950 mm",
                         "crops": "cotton, soybean, tur, sorghum, green gram, black gram", "soil": "mostly black (cotton) soils, shallow to deep"},
    "Central Vidarbha": {"mr": "मध्य वऱ्हाड", "districts": ["Nagpur", "Wardha", "Yavatmal"], "rain": "950-1250 mm",
                         "crops": "cotton, soybean, tur, sorghum, orange (Nagpur mandarin)", "soil": "black soils mixed with lighter, shallow soils"},
    "Eastern Vidarbha": {"mr": "पूर्व वऱ्हाड", "districts": ["Chandrapur", "Gadchiroli", "Gondia", "Bhandara"], "rain": "1250-1700 mm",
                         "crops": "paddy (main), also pulses and oilseeds, some cotton and soybean", "soil": "lighter red and yellow soils and heavy clay in paddy tracts"},
}
_SAFE = re.compile(r"[^ऀ-ॿA-Za-z0-9 .\-]")


@functools.lru_cache(maxsize=1)
def data() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {"districts": {}}


def overview() -> list[dict]:
    """District and taluka names only (villages are loaded per taluka)."""
    return [{"en": d, "mr": v["mr"], "talukas": [{"en": t["en"], "mr": t["mr"]} for t in v["talukas"]]} for d, v in data()["districts"].items()]


def villages(district: str, taluka: str) -> list[list]:
    t = _taluka(district, taluka)
    return [[v[0], v[1]] for v in t["villages"]] if t else []


def _taluka(district: str, taluka: str):
    d = data()["districts"].get(district)
    return next((t for t in d["talukas"] if t["en"] == taluka), None) if d else None


def zone_of(district: str):
    return next(((n, z) for n, z in ZONES.items() if district in z["districts"]), (None, None))


def clean_text(s: str, n: int = 60) -> str:
    """Free text from the page (village, name) is shown to the model, so keep only letters, digits, space, dot, hyphen."""
    return _SAFE.sub("", s or "").strip()[:n]


def resolve(district: str, taluka: str, village: str = ""):
    """Validated place with coordinates, or None. District and taluka must exist; the village may be any typed text."""
    t = _taluka(district, taluka)
    if not t:
        return None
    village = clean_text(village)
    lat, lon = t["lat"], t["lon"]
    for v in t["villages"]:
        if village and village.casefold() in (v[0].casefold(), v[1].casefold()) and len(v) == 4:
            lat, lon = v[2], v[3]
            break
    return {"district": district, "district_mr": data()["districts"][district]["mr"], "taluka": taluka, "taluka_mr": t["mr"],
            "village": village, "lat": lat, "lon": lon}


CROPS = {"cotton", "soybean", "orange", "tur", "paddy"}


def context(place: dict, weather_text: str = "", crops=()) -> str:
    """Block appended to the model's system prompt."""
    zone, z = zone_of(place["district"])
    where = ", ".join(x for x in (place["village"], f"taluka {place['taluka']}", f"{place['district']} district") if x)
    lines = [f"FARMER'S LOCATION: {where}, Vidarbha, Maharashtra."]
    crops = [c for c in crops if c in CROPS]
    if crops:
        lines.append("The farmer grows: " + ", ".join(crops) + ". Prefer these crops when the question does not name one.")
    if z:
        lines.append(f"Region: {zone}. Normal rainfall about {z['rain']}; main crops: {z['crops']}; soils: {z['soil']} (general, not this farm's soil).")
    if weather_text:
        lines.append("WEATHER at the farmer's place (forecast, not a guarantee):\n" + weather_text)
    lines.append("Use the location and weather to make the answer specific (timing, rain, heat). Quote weather numbers exactly as given and add "
                 "no weather facts that are not listed (no wind, no cloud, no 'cool air' unless listed). Do not claim to know the soil of this farm; "
                 "for soil or fertiliser say to check the Soil Health Card and ask the local कृषी सहाय्यक. If a sentence is not supported above, do not invent it.")
    return "\n".join(lines)
