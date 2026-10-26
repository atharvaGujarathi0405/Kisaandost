import json

from scripts import build_mr2vh_pairs as b


def row(m, v="तो घरी गेला होता आणि त्याले भेटला.", ch="MAT_001"):
    return {"ref": "X.1.1", "chapter": ch, "marathi": m, "varhadi": v, "prompt_version": b.PROMPT_VERSION}


def test_chapter_of():
    assert b.chapter_of("JHN.3.2") == "JHN_003" and b.chapter_of("MAT.28.20") == "MAT_028"


def test_keep_filters():
    ok = "तो घरी गेला होता आणि त्याला भेटला."
    assert b.keep(row(ok)) is None
    assert b.keep(row(ok + " KVK")) == "latin"
    assert b.keep(row("तो घरी गेला होता अन् त्याला भेटला.")) == "copied_varhadi"
    assert b.keep(row("तो")) == "length"
    v = "तो घरी गेला होता आणि भेटला."
    assert b.keep(row(v, v)) == "identical"


def test_splits_are_by_chapter_and_never_leak(tmp_path, monkeypatch):
    rows = [{**row("तो घरी गेला होता आणि त्याला भेटला."), "ref": f"{c[:3]}.{int(c[4:])}.1", "chapter": "JHN.1.VAHNT"}
            for c in ["MAT_001", "JHN_001", "JHN_002", "MAT_002"]]
    monkeypatch.setattr(b, "OUT", tmp_path)
    monkeypatch.setattr(b, "done", lambda: {r["ref"]: r for r in rows})
    b.build_splits()
    got = {s: [json.loads(l)["chapter"] for l in (tmp_path / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
           for s in ("train", "val", "test")}
    assert got == {"train": ["MAT_001", "MAT_002"], "val": ["JHN_002"], "test": ["JHN_001"]}


def test_length_guard_falls_back_on_runaway_output():
    from bot.mr2vh_model import guard
    src = "तो घरी गेला होता आणि त्याला भेटला"
    assert guard(src, src, "RULES") == src                      # same length: kept
    assert guard(src, src * 3, "RULES") == "RULES"              # repetition loop: too long
    assert guard(src, "तो", "RULES") == "RULES"                 # truncated: too short
