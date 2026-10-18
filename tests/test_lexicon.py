import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mine_lexicon as cli  # noqa: E402
from config import ROOT_DIR  # noqa: E402
from varhadi_data import gold, lexicon  # noqa: E402


def seg(sid, candidate, asr, chapter="JHN_001", triage="NEEDS_REVIEW"):
    r = gold.new_row(sid, "audio/x.mp3", "unknown", "0.000", "5.000", candidate=candidate, raw_asr=asr,
                     source_chapter=chapter)
    r["triage_status"] = triage
    return r


def test_align_tokens_returns_substitutions_and_counts_skips():
    pairs, skipped = lexicon.align_tokens("तो काई नाई बोलला।", "तो काही नाही बोलला")
    assert pairs == [("काई", "काही"), ("नाई", "नाही")]
    assert not skipped
    pairs, skipped = lexicon.align_tokens("अ ब क", "अ क ड")
    assert pairs == [] and skipped == {"delete": 1, "insert": 1}
    pairs, skipped = lexicon.align_tokens("अ बक", "अ ब क")  # merge/split is not paired 1:1
    assert pairs == [] and skipped == {"unequal_replace": 1}


def test_mine_aggregates_counts_segments_chapters_consistency():
    rows = [seg("s1", "तो काई म्हणे", "तो काही म्हणे"), seg("s2", "काई झालं", "काही झालं", chapter="LUK_001"),
            seg("s3", "काई झालं", "काय झालं", triage="HIGH_CONFIDENCE_CANDIDATE")]
    out, stats = lexicon.mine(rows)
    top = out[0]
    assert (top["varhadi_form"], top["asr_rendering"], top["count"]) == ("काई", "काही", 2)
    assert top["n_segments"] == 2 and top["n_chapters"] == 2 and top["consistency"] == 0.667
    assert top["example_segment_ids"] == "s1;s2" and top["triage"] == "NEEDS_REVIEW:2"
    assert out[1]["consistency"] == 0.333
    assert stats["segments"] == 3 and stats["substitutions"] == 3 and stats["unique_forms"] == 1


def test_inconsistent_form_ranks_below_consistent_one():
    rows = [seg("a1", "पयले", "पहिले"), seg("a2", "पयले", "पहिले"),  # one rendering, consistency 1.0
            seg("b1", "त्याले", "त्याला"), seg("b2", "त्याले", "त्याला"),  # same count but noisy
            seg("b3", "त्याले", "त्याने"), seg("b4", "त्याले", "त्यांना")]
    out, _ = lexicon.mine(rows)
    rank = {(r["varhadi_form"], r["asr_rendering"]): i for i, r in enumerate(out)}
    assert rank["पयले", "पहिले"] < rank["त्याले", "त्याला"] < rank["त्याले", "त्याने"]


def test_cli_output_never_fills_standard_marathi(tmp_path, monkeypatch):
    rows = [seg(f"s{i}", "काई नाई पयले", "काही नाही पहिले") for i in range(4)] + [seg("e", "", "काही")]
    monkeypatch.setattr(lexicon.gold, "read_gold", lambda data_dir=None: rows)
    out = tmp_path / "lex.tsv"
    assert cli.main(["--out", str(out)]) == 0
    written = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    assert len(written) == 3
    assert all(r["standard_marathi"] == "" for r in written)
    assert all(r["verified"] == "false" and r["source"] == lexicon.SOURCE for r in written)


def test_default_output_is_gitignored():
    import subprocess

    default = Path(cli.lexicon.DATA_DIR / "transcripts/lexicon_candidates.tsv").relative_to(ROOT_DIR)
    assert subprocess.run(["git", "check-ignore", "-q", str(default)], cwd=ROOT_DIR).returncode == 0
