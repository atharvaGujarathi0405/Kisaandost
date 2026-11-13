import glob
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from bot import mr2vh_model, server

VAHNT = Path(__file__).resolve().parents[1] / "data" / "transcripts" / "vahnt_text"
KB = json.loads((Path(server.__file__).with_name("farm_kb.json")).read_text(encoding="utf-8"))


def test_farm_rules_are_applied_but_hand_checked_rules_win():
    out = server.varhadify("आतापर्यंत पिकांना कृषी सहाय्यकाला विचारा, आणि नाही तर आहे.")
    assert "आतापरेंत" in out and "सहाय्यकाले" in out
    assert "अन्" in out and "नाई" in out and "हाय" in out               # the older rules still apply
    for k in KB["rules"]:                                               # a farm rule must never override a hand-checked one
        assert k not in server.STYLE["enforce"] and k not in server.STYLE["enforce_by_rule"], k


def test_farm_rules_never_touch_negations_or_digits():
    assert not set(KB["rules"]) & mr2vh_model.NEGATORS
    assert not any(re.search(r"\d", k + v) for k, v in KB["rules"].items())


def test_farm_rules_are_off_with_the_switch(monkeypatch):
    monkeypatch.setattr(server, "FARM_RULES", {})
    assert server.varhadify("आतापर्यंत") == "आतापर्यंत"


@pytest.mark.skipif(not VAHNT.exists(), reason="local VAHNT corpus (gitignored) not present")
def test_every_lexical_farm_rule_is_attested_in_the_real_text():
    c = Counter()
    for f in glob.glob(str(VAHNT / "*.json")):
        for u in json.load(open(f, encoding="utf-8"))["units"]:
            c.update(re.findall(r"[ऀ-ॿ]+", u["text"]))
    lexical = {k: v for k, v in KB["rules"].items() if isinstance(KB["evidence"][k], dict) and "dative_nouns_in_real_text" not in KB["evidence"][k]}  # dictionary swaps (string evidence) are not scripture-attested by design
    bad = [(k, v, c[v]) for k, v in lexical.items() if c[v] < 8]
    assert not bad, bad


def test_native_approved_swaps_extend_the_allowed_list(monkeypatch, tmp_path):
    (tmp_path / "allowed_subs.json").write_text(json.dumps({"pairs": [["अ", "ब"]], "suffixes": []}), encoding="utf-8")
    (tmp_path / "farm_kb.json").write_text(json.dumps({"allowed_extra": [["क", "ख"]]}), encoding="utf-8")
    monkeypatch.setattr(mr2vh_model, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(mr2vh_model, "__file__", str(tmp_path / "mr2vh_model.py"))   # the module reads farm_kb.json next to itself
    assert mr2vh_model._allowed()["pairs"] == [["अ", "ब"], ["क", "ख"]]
