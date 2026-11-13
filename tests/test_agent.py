import json
from types import SimpleNamespace

import pytest

from bot import agent, rag, server

pytest.importorskip("google.genai")

CHUNKS = [
    {"id": "a:1:0", "source": "a", "title": "Pigeonpea guide", "publisher": "ICAR", "url": "https://x/a", "page": 3, "crops": ["tur"],
     "text": "Wilt of pigeonpea is managed by resistant varieties, seed treatment and crop rotation with sorghum."},
    {"id": "b:2:0", "source": "b", "title": "Cotton guide", "publisher": "CICR", "url": "https://x/b", "page": 7, "crops": ["cotton"],
     "text": "Pink bollworm in cotton is monitored with pheromone traps and managed by timely sowing and clean cultivation."},
    {"id": "c:1:0", "source": "c", "title": "General", "publisher": "PDKV", "url": "https://x/c", "page": 1, "crops": [],
     "text": "Soil testing before sowing helps to plan fertiliser use."},
]


@pytest.fixture
def kb(tmp_path, monkeypatch):
    f = tmp_path / "chunks.jsonl"
    f.write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    monkeypatch.setattr(rag, "CHUNKS", f)
    rag._idx.clear()
    yield
    rag._idx.clear()


def test_search_ranks_the_matching_crop_first(kb):
    assert rag.search("wilt management in pigeonpea")[0]["source"] == "a"
    assert rag.search("pink bollworm pheromone traps")[0]["source"] == "b"
    assert rag.search("managed", crop="cotton")[0]["source"] == "b"      # crop filter pushes other crops down
    assert rag.search("zzzz unknown words") == []


def test_split_sources_strips_the_last_line_and_returns_only_cited_passages():
    seen = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    assert agent.split_sources("उत्तर.\nSOURCES: 1,3", seen) == ("उत्तर.", [{"title": "A"}, {"title": "C"}])
    assert agent.split_sources("उत्तर.\nSOURCES: none", seen) == ("उत्तर.", [])
    assert agent.split_sources("उत्तर without a sources line", seen) == ("उत्तर without a sources line", [])


def test_agent_runs_the_search_tool_and_cites_what_it_used(kb):
    calls = []

    def generate_content(model, contents, config):
        tool = config.tools[0]                                # what Gemini would call
        calls.append(tool("pigeonpea wilt management", "tur"))
        assert "[1] Pigeonpea guide (ICAR, p.3)" in calls[0]
        return SimpleNamespace(text="तुरीवरील मर रोगासाठी पीक फेरपालट करा.\nSOURCES: 1")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    reply, sources = agent.answer([{"role": "user", "content": "तुरीला मर रोग आलाय"}], "SYS", client=client)
    assert reply == "तुरीवरील मर रोगासाठी पीक फेरपालट करा."
    assert sources == [{"title": "Pigeonpea guide", "publisher": "ICAR", "url": "https://x/a", "page": 3}]


def test_chat_uses_gemini_agent_then_varhadi_and_returns_sources(monkeypatch, kb):
    monkeypatch.setattr(agent, "enabled", lambda: True)
    monkeypatch.setattr(server.mr2vh_model, "available", lambda: False)
    monkeypatch.setattr(agent, "answer", lambda msgs, system: ("तुम्ही पीक फेरपालट करा.", [{"title": "T", "publisher": "P", "url": "u", "page": 1}]))
    from fastapi.testclient import TestClient
    r = TestClient(server.app).post("/api/chat", json={"messages": [{"role": "user", "content": "तुरीला मर रोग आलाय"}]})
    assert r.status_code == 200
    assert r.json()["reply"] == "तुमी पीक फेरपालट करा."           # rule layer turned it into Varhadi
    assert r.json()["sources"][0]["title"] == "T"


def test_chat_maps_gemini_rate_limit(monkeypatch, kb):
    monkeypatch.setattr(agent, "enabled", lambda: True)

    def boom(msgs, system):
        raise type("E", (Exception,), {"code": 429})()
    monkeypatch.setattr(agent, "answer", boom)
    from fastapi.testclient import TestClient
    r = TestClient(server.app).post("/api/chat", json={"messages": [{"role": "user", "content": "प्रश्न"}]})
    assert r.status_code == 429


@pytest.mark.parametrize("code", [503, 429, 404])
def test_agent_falls_back_to_the_next_model_when_one_is_overloaded_out_of_quota_or_retired(kb, code):
    tried = []

    def generate_content(model, contents, config):
        tried.append(model)
        if len(tried) == 1:
            raise type("E", (Exception,), {"code": code})()
        return SimpleNamespace(text="उत्तर.\nSOURCES: none")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    assert agent.answer([{"role": "user", "content": "प्रश्न"}], "SYS", client=client)[0] == "उत्तर."
    assert tried == agent.MODELS[:2]


def test_refine_keeps_the_draft_when_gemini_fails_or_breaks_a_check(monkeypatch):
    from bot import refine
    monkeypatch.setattr(refine, "FAILED", 0)
    drafts = [("मी आज येणार नाही.", "मी आज येणार नाई."), ("तुम्ही पाणी द्या.", "तुमी पानी द्या.")]
    bad = SimpleNamespace(text="[[1]] मी आज येणार हाय.\n[[2]] तुमी पानी द्या ना.")                 # drops the negation; second is fine
    monkeypatch.setattr(agent, "generate", lambda *a, **k: bad)
    assert refine.refine(drafts, {}, client=object(), pool=[]) == ["मी आज येणार नाई.", "तुमी पानी द्या ना."]
    def down(*a, **k):
        raise type("E", (Exception,), {"code": 429})()
    monkeypatch.setattr(agent, "generate", down)
    assert refine.refine(drafts, {}, client=object(), pool=[]) == [d for _, d in drafts] and refine.FAILED == 1


def test_lexicon_replaces_exact_words_and_phrases_only():
    from bot import lexicon
    lex = {"spans": {"मी": "म्या", "पडत आहेत": "पडतायत", "पाने": "पानं"}, "suffixes": [["डे", "डं"]], "protect": ["मी"]}
    assert lexicon.apply("मी पाने पडत आहेत.", lex) == "म्या पानं पडतायत."
    assert lexicon.apply("मीठ पडत,आहेत", lex) == "मीठ पडत,आहेत"            # not inside a longer word; a comma breaks the phrase
    assert lexicon.apply("झाडे बोंडे", lex) == "झाडं बोंडं"                  # suffix rule on words the lexicon does not know
    assert lexicon.apply("डे", lex) == "डे"                                  # never strips the whole word
