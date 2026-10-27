"""Build bot/places.json: Vidarbha district -> taluka -> villages (name in Marathi when OpenStreetMap has it, lat/lon).

  python -m scripts.build_vidarbha_places            (about 11 Overpass requests, one per district, with back-off)

Source: OpenStreetMap contributors (ODbL; the UI must credit them). OSM is incomplete: it holds the villages mappers added, not
the full Census list, so the farmer can also type a village that is not in the list. Taluka list checked against Wikipedia's
"List of talukas of Maharashtra". One request per district is deliberate: the public Overpass server asks bots to go easy.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "bot" / "places.json"
RAW = Path(__file__).resolve().parent.parent / "data" / "places_raw"
UA = "KisaanDost-student-project/1.0 (atharvagujarathi0405@gmail.com)"
HOSTS = ["https://overpass-api.de/api/interpreter"]      # the kumi.systems mirror timed out when tried
# OSM `name` of each district relation (admin_level 5) -> Marathi label
DISTRICTS = {"Akola": "अकोला", "Amravati": "अमरावती", "Buldhana": "बुलढाणा", "Washim": "वाशीम", "Yavatmal": "यवतमाळ",
             "Bhandara": "भंडारा", "Chandrapur": "चंद्रपूर", "Gadchiroli": "गडचिरोली", "Gondia": "गोंदिया", "Nagpur": "नागपूर",
             "Wardha": "वर्धा"}
OSM_NAME = {"Gondia": "Gondiya"}                        # where OSM spells a district differently
DEVA = re.compile("[ऀ-ॿ]")
QUERY = """[out:json][timeout:280];
rel["boundary"="administrative"]["admin_level"="5"]["name"="{d}"]->.dist;
.dist map_to_area->.da;
rel(area.da)["boundary"="administrative"]["admin_level"="6"]->.t;
foreach.t->.cur{{
  .cur out tags center;
  .cur map_to_area->.a;
  node(area.a)["place"~"^(village|hamlet|town|city)$"];
  out body;
}}"""


def fetch(d: str) -> list:
    cache = RAW / f"{d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["elements"]
    wait = 20
    for attempt in range(8):
        host = HOSTS[attempt % len(HOSTS)]
        try:
            req = urllib.request.Request(host, data=urllib.parse.urlencode({"data": QUERY.format(d=OSM_NAME.get(d, d))}).encode(), headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = r.read().decode("utf-8")
            data = json.loads(txt)                      # a busy server answers with HTML/markdown, which fails here
            RAW.mkdir(parents=True, exist_ok=True)
            cache.write_text(txt, encoding="utf-8")
            return data["elements"]
        except Exception as e:
            print(f"  {d}: attempt {attempt + 1} failed ({type(e).__name__}); waiting {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait); wait = min(wait * 2, 300)
    raise SystemExit(f"giving up on {d}")


def label(t: dict) -> tuple[str, str]:
    """(shown name, roman name): Marathi if OSM has it, else `name` when it is Devanagari, else the roman name."""
    name = t.get("name") or ""
    roman = t.get("name:en") or ("" if DEVA.search(name) else name)
    mr = t.get("name:mr") or (name if DEVA.search(name) else "")
    return (mr or roman or "").strip(), roman.strip()


def parse(d: str, els: list) -> list:
    talukas, cur = [], None
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "relation":
            c = e.get("center", {})
            shown, roman = label(t)
            cur = {"mr": shown, "en": roman or t.get("name", ""), "lgd": t.get("ref:LGD:subdistrict", ""), "lat": round(c.get("lat", 0), 4), "lon": round(c.get("lon", 0), 4), "villages": {}}
            talukas.append(cur)
        elif cur is not None and e["type"] == "node":
            shown, roman = label(t)
            if shown:
                key = (shown, roman)
                cur["villages"][key] = [shown, roman, round(e["lat"], 4), round(e["lon"], 4)] if "lat" in e else [shown, roman]
    for t in talukas:
        t["villages"] = sorted(t["villages"].values(), key=lambda v: v[0])
    return sorted(talukas, key=lambda t: t["en"])


def main() -> None:
    out = {"source": "OpenStreetMap contributors, ODbL", "districts": {}}
    for en, mr in DISTRICTS.items():
        print(en, flush=True)
        talukas = parse(en, fetch(en))
        out["districts"][en] = {"mr": mr, "talukas": talukas}
        print(f"  {len(talukas)} talukas, {sum(len(t['villages']) for t in talukas)} places", flush=True)
        time.sleep(10)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
