import glob
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bot import server


def fake_client(reply="तुम्ही काय करता आहे?", seen=None):
    def create(**kw):
        if seen is not None:
            seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Tests must not load the 1.2 GB model even when its files exist locally."""
    monkeypatch.setattr(server.mr2vh_model, "available", lambda: False)


@pytest.fixture
def api():
    c = TestClient(server.app)
    yield c
    server.app.dependency_overrides.clear()


def use(client):
    server.app.dependency_overrides[server.get_client] = lambda: client


def post(api, msgs):
    return api.post("/api/chat", json={"messages": msgs})


def test_varhadify_swaps_whole_words_only():
    assert server.varhadify("तुम्ही नाही आहेत, मला सांगा.") == "तुमी नाई हायत, मले सांगा."
    assert server.varhadify("आहेत") == "हायत"            # not partially matched by आहे
    assert server.varhadify("मगर आला") == "मगर आला"       # मग inside मगर untouched; आला is a verb, untouched
    assert server.varhadify("हवामान ठीक आहे") == "हवामान ठीक हाय"


def test_tidy_strips_list_markers():
    assert server.tidy("- एक\n१. दोन\n2) तीन\n\n\n\nचार-पाच") == "एक\nदोन\nतीन\n\nचार-पाच"


def test_reply_is_converted_and_prompt_sent(api):
    seen = {}
    use(fake_client("तुम्ही नाही म्हणालात", seen))
    r = post(api, [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}])
    assert r.status_code == 200 and r.json()["reply"] == "तुमी नाई म्हणालात"
    assert seen["messages"][0] == {"role": "system", "content": server.SYSTEM}
    assert seen["messages"][1:] == [{"role": "user", "content": "q"}]   # leading assistant turn dropped


def test_last_message_must_be_user(api):
    use(fake_client())
    assert post(api, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]).status_code == 422


def test_validation(api):
    use(fake_client())
    assert post(api, [{"role": "system", "content": "x"}]).status_code == 422
    assert post(api, [{"role": "user", "content": "x" * 1001}]).status_code == 422
    assert post(api, []).status_code == 422


def test_empty_model_reply_is_502(api):
    use(fake_client(reply=""))
    assert post(api, [{"role": "user", "content": "q"}]).status_code == 502


def test_missing_key_is_503(api, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert post(api, [{"role": "user", "content": "q"}]).status_code == 503


def test_system_prompt_forbids_dialect_and_doses():
    assert "Do NOT try to write Varhadi" in server.SYSTEM and "dose" in server.SYSTEM


def test_ui_served(api):
    assert "किसानदोस्त" in api.get("/").text


VAHNT = Path(__file__).resolve().parents[1] / "data" / "transcripts" / "vahnt_text"


@pytest.mark.skipif(not VAHNT.exists(), reason="local VAHNT corpus (gitignored) not present")
def test_every_enforced_form_is_attested_in_corpus():
    """Evidence gate: each Varhadi form must occur in the corpus, at least as often as the standard form."""
    c = Counter()
    for f in glob.glob(str(VAHNT / "*.json")):
        for u in json.load(open(f, encoding="utf-8"))["units"]:
            c.update(re.findall("[ऀ-ॿ]+", u["text"]))
    bad = [(k, v, c[k], c[v]) for k, v in server.STYLE["enforce"].items() if c[v] < 1 or c[v] < c[k]]
    assert not bad, bad


ALLOWED = {"pairs": [["तुम्हाला", "तुमाले"], ["आहे", "हाय"], ["कृषी", "कृषी"]], "suffixes": [["ला", "ले"], ["ने", "नं"]]}


def test_constrain_keeps_only_attested_changes():
    from bot import mr2vh_model as m
    # तुम्हाला->तुमाले attested; हवे->हाय is not (meaning change: need -> is) and must revert; suffix ला->ले allowed
    assert m.constrain("तुम्हाला खत हवे", "तुमाले खत हाय", ALLOWED) == "तुमाले खत हवे"
    assert m.constrain("कृषी सहाय्यकाला विचारा", "कृषी सहाय्यकाले विचारा", ALLOWED) == "कृषी सहाय्यकाले विचारा"
    assert m.constrain("देवाने सांगितले", "देवानं सांगितले", ALLOWED) == "देवानं सांगितले"   # ने->नं suffix
    assert m.constrain("देवाने सांगितले", "देवाले सांगितले", ALLOWED) == "देवाने सांगितले"  # ने->ले vetoed/unlisted


def test_constrain_reverts_inserts_deletes_and_merges():
    from bot import mr2vh_model as m
    A = {"pairs": [["कृषी", "कृषी"]], "suffixes": []}
    assert m.constrain("पाणी द्या आज", "पाणी आज", A) == "पाणी द्या आज"              # deleted word comes back
    assert m.constrain("पाणी द्या", "पाणी जरूर द्या", A) == "पाणी द्या"             # inserted word dropped
    assert m.constrain("पाणी देऊ नका", "पाणीदेऊ नका", A) == "पाणी देऊ नका"         # merge/split reverted


def test_negation_and_digits_must_survive():
    from bot import mr2vh_model as m
    assert m.negation_ok("पाणी देऊ नका", "पाणी देऊ नका")
    assert not m.negation_ok("पाणी देऊ नका", "पाणी देऊ")
    assert m.negation_ok("हे नाही", "हे नाई")                  # rules rewrite नाही -> नाई, not a loss
    assert not m.negation_ok("पाणी द्या", "पाणी नका द्या")
    assert not m.negation_ok("10 लिटर पाणी", "लिटर पाणी")


def test_build_allowed_needs_repeats(tmp_path):
    from bot import mr2vh_model as m
    f = tmp_path / "train.jsonl"
    rows = [{"source": "तुम्हाला पाणी", "target": "तुमाले पाणी"}] * 2 + [{"source": "एक तीन", "target": "एक चार"}]
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    d = m.build_allowed(f)
    assert ["तुम्हाला", "तुमाले"] in d["pairs"] and ["तीन", "चार"] not in d["pairs"]


def test_build_allowed_drops_ambiguous_pairs(tmp_path):
    from bot import mr2vh_model as m
    f = tmp_path / "train.jsonl"
    rows = [{"source": "तुम्ही का", "target": "तुमी कावून"}] * 2 + [{"source": "तुम्ही का", "target": "तुमी का"}] * 10
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    d = m.build_allowed(f)
    assert ["तुम्ही", "तुमी"] in d["pairs"]          # changes every time
    assert ["का", "कावून"] not in d["pairs"]         # changes in 2 of 12 occurrences: context dependent


def test_translate_text_is_sentence_wise_and_guarded(monkeypatch):
    from bot import mr2vh_model as m
    monkeypatch.setattr(m, "_allowed", lambda: {"pairs": [], "suffixes": []})
    text = "तुम्ही शेतात जा. पाणी द्या!\nखत टाका?"
    seen = []

    def fake(sents):
        seen.extend(sents)
        # 2nd sentence: runaway output (repetition loop) must fall back to the rules
        return [s + "!" if i != 1 else s * 5 for i, s in enumerate(sents)]

    out = m.translate_text(text, fallback=lambda s: "RULES(" + s + ")", generate=fake)
    assert seen == ["तुम्ही शेतात जा.", "पाणी द्या!", "खत टाका?"]
    assert out == "तुम्ही शेतात जा.! RULES(पाणी द्या!)\nखत टाका?!"


def test_to_varhadi_uses_model_then_rules(monkeypatch):
    monkeypatch.setattr(server.mr2vh_model, "available", lambda: True)
    monkeypatch.setattr(server.mr2vh_model, "translate_text", lambda t, fallback: t.replace("शेतात", "वावरात"))
    assert server.to_varhadi("- तुम्ही शेतात जा") == "तुमी वावरात जा"   # list marker stripped, rules still applied


def test_to_varhadi_survives_model_failure(monkeypatch):
    monkeypatch.setattr(server.mr2vh_model, "available", lambda: True)

    def boom(t, fallback):
        raise RuntimeError("oom")
    monkeypatch.setattr(server.mr2vh_model, "translate_text", boom)
    assert server.to_varhadi("तुम्ही नाही") == "तुमी नाई"


def test_filter_off_keeps_unseen_word_changes(monkeypatch):
    from bot import mr2vh_model as m
    monkeypatch.setattr(m, "_allowed", lambda: {"pairs": [], "suffixes": []})
    gen = lambda sents: ["पानायवर डाग"]
    monkeypatch.setattr(m, "FILTER", "strict")
    assert m.translate_text("पानांवर डाग", fallback=lambda s: s, generate=gen) == "पानांवर डाग"   # unseen change reverted
    monkeypatch.setattr(m, "FILTER", "off")
    assert m.translate_text("पानांवर डाग", fallback=lambda s: s, generate=gen) == "पानायवर डाग"   # kept


def test_stray_quote_marks_from_the_model_are_removed(monkeypatch):
    from bot import mr2vh_model as m
    monkeypatch.setattr(m, "_allowed", lambda: {"pairs": [], "suffixes": []})
    gen = lambda sents: ["पाने गळतात.”"]
    assert m.translate_text("पाने गळतात.", fallback=lambda s: s, generate=gen) == "पाने गळतात."
    gen2 = lambda sents: ["तो म्हणाला “हो”"]
    assert m.translate_text("तो म्हणाला “हो”", fallback=lambda s: s, generate=gen2) == "तो म्हणाला “हो”"   # quotes in the source are kept
