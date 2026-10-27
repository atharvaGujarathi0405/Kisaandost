import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bot import places, server, weather

FAKE = {"districts": {"Akola": {"mr": "अकोला", "talukas": [
    {"en": "Akot", "mr": "अकोट", "lgd": "1", "lat": 21.1, "lon": 77.0,
     "villages": [["आसेगाव", "Asegaon", 21.2, 77.1], ["पाटसूल", "Patsul"]]}]}}}


def fake_client(seen):
    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="पाणी द्या."))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def use(client):
    server.app.dependency_overrides[server.get_client] = lambda: client


@pytest.fixture(autouse=True)
def fake_places(monkeypatch):
    monkeypatch.setattr(places, "data", lambda: FAKE)
    monkeypatch.setattr(server.mr2vh_model, "available", lambda: False)      # never load the 1.2 GB model in tests
    weather._cache.clear()


def open_meteo(*a, **k):
    body = {"current": {"temperature_2m": 31.4, "relative_humidity_2m": 60, "precipitation": 0},
            "daily": {"time": ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"], "temperature_2m_max": [34, 41, 33, 33], "temperature_2m_min": [24, 25, 23, 22],
                      "precipitation_sum": [12, 0, None, 1], "precipitation_probability_max": [80, 10, None, 20]}}
    return io.BytesIO(json.dumps(body).encode())


def test_resolve_needs_a_real_district_and_taluka_but_any_village():
    assert places.resolve("Akola", "Nowhere") is None and places.resolve("Mars", "Akot") is None
    listed = places.resolve("Akola", "Akot", "आसेगाव")
    assert (listed["lat"], listed["lon"]) == (21.2, 77.1)                       # village coordinates when OSM has them
    typed = places.resolve("Akola", "Akot", "माझं गाव <script>")
    assert (typed["lat"], typed["lon"]) == (21.1, 77.0) and "<" not in typed["village"]   # taluka centre, sanitised text


def test_prompt_context_has_zone_and_blocks_invention():
    ctx = places.context(places.resolve("Akola", "Akot", "आसेगाव"), "आज: 24-34°C")
    assert "Western Vidarbha" in ctx and "आज: 24-34°C" in ctx and "Soil Health Card" in ctx


def test_weather_is_cached_and_summarised():
    calls = []
    def opener(*a, **k): calls.append(1); return open_meteo()
    w = weather.fetch(21.1, 77.0, opener); weather.fetch(21.1, 77.0, opener)
    assert len(calls) == 1 and w["days"][2]["rain_mm"] == 0                      # null precipitation becomes 0
    assert "आज: 24-34°C" in weather.summary(w) and "शुक्रवार: 22-33°C" in weather.summary(w)   # 4th day is named by weekday
    assert any("पावसानंतर" in a for a in weather.advice(w)) and any("ऊन" in a for a in weather.advice(w))


def test_chat_sends_location_and_weather_to_the_model(api, monkeypatch):
    monkeypatch.setattr(weather, "fetch", lambda lat, lon: {"now": {"temp": 30, "humidity": 50}, "days": [{"date": "2026-10-27", "tmax": 33, "tmin": 22, "rain_mm": 0, "rain_pct": 5}] * 4})
    seen = {}
    use(fake_client(seen))
    r = api.post("/api/chat", json={"messages": [{"role": "user", "content": "पेरणी?"}], "profile": {"district": "Akola", "taluka": "Akot", "village": "आसेगाव"}})
    assert r.status_code == 200
    system = seen["messages"][0]["content"]
    assert "Akot" in system and "WEATHER" in system and "आज:" in system


def test_chat_survives_weather_outage_and_unknown_place(api, monkeypatch):
    monkeypatch.setattr(weather, "fetch", lambda lat, lon: (_ for _ in ()).throw(OSError("down")))
    seen = {}
    use(fake_client(seen))
    ok = {"messages": [{"role": "user", "content": "पेरणी?"}]}
    assert api.post("/api/chat", json={**ok, "profile": {"district": "Akola", "taluka": "Akot"}}).status_code == 200
    assert "WEATHER" not in seen["messages"][0]["content"] and "Akot" in seen["messages"][0]["content"]
    api.post("/api/chat", json={**ok, "profile": {"district": "Mars", "taluka": "X"}})
    assert "FARMER'S LOCATION" not in seen["messages"][0]["content"]


def test_endpoints(api, monkeypatch):
    assert api.get("/api/places").json()[0]["talukas"][0]["en"] == "Akot"
    assert api.get("/api/places/villages", params={"district": "Akola", "taluka": "Akot"}).json()[0] == ["आसेगाव", "Asegaon"]
    assert api.get("/api/places/villages", params={"district": "x", "taluka": "y"}).json() == []
    assert api.get("/api/weather", params={"district": "x", "taluka": "y"}).status_code == 404
    monkeypatch.setattr(weather, "fetch", lambda lat, lon: (_ for _ in ()).throw(OSError("down")))
    assert api.get("/api/weather", params={"district": "Akola", "taluka": "Akot"}).status_code == 502


@pytest.fixture
def api():
    c = TestClient(server.app)
    yield c
    server.app.dependency_overrides.clear()
