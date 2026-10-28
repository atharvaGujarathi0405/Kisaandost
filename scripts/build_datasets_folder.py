"""Collect every dataset of the project as a CSV in datasets/ and write datasets/README.md (a catalogue).

  python -m scripts.build_datasets_folder

Nothing is invented here: each CSV is a flat export of a file the project already has (data/, bot/). Audio is NOT copied.
datasets/ is gitignored except README.md and the files in OPEN (see .gitignore): the other CSVs contain VAHNT scripture text
(unverified licence) or Wikipedia text and stay local, like their sources under data/.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D, OUT = ROOT / "data", ROOT / "datasets"
LATIN = re.compile(r"[A-Za-z0-9]")
REFERENCE = re.compile(r"Archived|बीज उपचारकी घरेलू")        # Wikipedia citation lines that slipped into the sentences
rows_written: dict[str, int] = {}


def write(name: str, fields: list[str], rows) -> None:
    rows = list(rows)
    try:
        with open(OUT / name, "w", encoding="utf-8-sig", newline="") as f:      # BOM so Excel shows Devanagari correctly
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    except PermissionError:                                                      # file open in Excel/IDE: keep the old copy
        print(f"WARNING: {name} is open in another program, left unchanged")
    rows_written[name] = len(rows)


def read_delim(path: Path, delim: str):
    return list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=delim))


def jsonl(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build() -> None:
    OUT.mkdir(exist_ok=True)

    # ---- created by the project -------------------------------------------------------------------------------------------
    src = read_delim(D / "agri_text" / "sources.tsv", "\t")
    write("agri_marathi_sentences.csv", ["id", "crop", "article", "marathi"], src)
    draft = read_delim(D / "agri_text" / "varhadi_draft.tsv", "\t")
    for r in draft:
        r["flag"] = ("wikipedia_reference_line" if REFERENCE.search(r["marathi"])
                     else "latin_or_digits" if LATIN.search(r["marathi"]) else "")
        r["identical_to_marathi"] = str(r["varhadi_draft"].strip() == r["marathi"].strip()).lower()
    write("agri_varhadi_drafts.csv", ["id", "crop", "article", "marathi", "varhadi_draft", "changed_words", "identical_to_marathi",
                                      "flag", "status", "reviewed_varhadi", "reviewer"], draft)
    farm = read_delim(D / "agri_text" / "mr2vh_farming_dataset.csv", ",")          # scripts/build_mr2vh_farming_dataset.py
    write("mr2vh_farming_dataset.csv", list(farm[0]), farm)
    write("recording_prompts.csv", ["prompt_id", "type", "domain", "topic", "crop", "text", "varhadi_text", "rewritten_by",
                                    "review_status", "notes"], read_delim(D / "prompts" / "recording_prompts.csv", ","))

    # ---- Varhadi adaptation layer (Marathi -> Varhadi) --------------------------------------------------------------------
    bt = jsonl(D / "translation" / "backtrans.jsonl")
    write("mr2vh_parallel_all.csv", ["ref", "chapter", "marathi", "varhadi"], bt)
    for split in ("train", "val", "test"):
        write(f"mr2vh_{split}.csv", ["ref", "chapter", "source_marathi", "target_varhadi"],
              ({"ref": r["ref"], "chapter": r["chapter"], "source_marathi": r["source"], "target_varhadi": r["target"]}
               for r in jsonl(D / "translation" / f"{split}.jsonl")))
    write("mr2vh_test_predictions.csv", ["ref", "chapter", "source_marathi", "reference_varhadi", "model_output"],
          ({"ref": r["ref"], "chapter": r["chapter"], "source_marathi": r["source"], "reference_varhadi": r["target"],
            "model_output": r.get("pred") or r.get("prediction") or r.get("output", "")}
           for r in jsonl(D / "translation" / "test_pred_v3.jsonl")))
    subs = json.load(open(D / "translation" / "allowed_subs_v3.json", encoding="utf-8"))["pairs"]
    write("mr2vh_allowed_substitutions.csv", ["marathi_word", "varhadi_word"], ({"marathi_word": a, "varhadi_word": b} for a, b in subs))
    style = json.load(open(ROOT / "bot" / "varhadi_style.json", encoding="utf-8"))
    write("varhadi_word_rules.csv", ["marathi_word", "varhadi_word"],
          ({"marathi_word": a, "varhadi_word": b} for a, b in style["enforce"].items()))
    lex = read_delim(D / "transcripts" / "lexicon_candidates.tsv", "\t")
    write("varhadi_lexicon_candidates.csv", list(lex[0]), lex)
    hc = read_delim(D / "transcripts" / "lexicon_candidates_hc.tsv", "\t")
    write("varhadi_lexicon_high_confidence.csv", list(hc[0]), hc)

    # ---- Varhadi speech (VAHNT audio Bible) -------------------------------------------------------------------------------
    gold = read_delim(D / "transcripts" / "gold_transcripts.csv", ",")
    write("vahnt_aligned_segments.csv", list(gold[0]), gold)
    tri = read_delim(D / "evaluation" / "triage_report.csv", ",")
    write("vahnt_confidence_triage.csv", list(tri[0]), tri)
    man = read_delim(D / "private" / "metadata" / "manifest.csv", ",")
    write("vahnt_audio_manifest.csv", list(man[0]), man)
    ev = read_delim(D / "evaluation" / "gold_eval.csv", ",")
    write("native_verified_gold_eval.csv", ["audio_path", "reference_text", "speaker_id", "segment_id", "source_chapter", "duration_sec"], ev)
    tts = [{"file_name": r["file_name"], "text": r["text"], "split": "train"}
           for r in read_delim(D / "tts_train" / "luk" / "metadata.csv", ",")]
    held = json.load(open(D / "tts_train" / "luk_heldout" / "texts.json", encoding="utf-8"))
    tts += [{"file_name": f"{h['id']}.ogg", "text": h["text"], "duration_sec": h["dur"], "split": "heldout"} for h in held]
    write("tts_luke_clips.csv", ["file_name", "text", "duration_sec", "split"], tts)

    # ---- open data used by the bot ----------------------------------------------------------------------------------------
    places = json.load(open(ROOT / "bot" / "places.json", encoding="utf-8"))["districts"]
    talukas, villages = [], []
    for dist, dv in places.items():
        for t in dv["talukas"]:
            talukas.append({"district_en": dist, "district_mr": dv["mr"], "taluka_en": t["en"], "taluka_mr": t["mr"],
                            "lgd_code": t["lgd"], "lat": t["lat"], "lon": t["lon"], "villages": len(t["villages"])})
            for en, mr, lat, lon in t["villages"]:
                villages.append({"district_en": dist, "taluka_en": t["en"], "village": en, "village_alt": mr if mr != en else "",
                                 "lat": lat, "lon": lon})
    write("vidarbha_talukas.csv", ["district_en", "district_mr", "taluka_en", "taluka_mr", "lgd_code", "lat", "lon", "villages"], talukas)
    write("vidarbha_villages.csv", ["district_en", "taluka_en", "village", "village_alt", "lat", "lon"], villages)

    g = json.load(open(ROOT / "bot" / "disease_guidance.json", encoding="utf-8"))
    write("disease_guidance.csv", ["crop", "class_label", "name_mr", "guidance_mr"],
          ({"crop": crop, "class_label": label, "name_mr": e["name_mr"], "guidance_mr": " ".join(e["mr"])}
           for crop, cls in g["crops"].items() for label, e in cls.items()))
    qa = json.load(open(D / "disease_eval" / "qa_soy_citrus.json", encoding="utf-8"))
    write("disease_qa.csv", list(qa[0]), qa)

    readme()


# file -> (what it is, where it comes from, licence, status, open?)
CATALOGUE = [
    ("CREATED BY THIS PROJECT", None),
    ("agri_marathi_sentences.csv", "Marathi farming sentences (cotton, soybean, orange, tur, general) split from Marathi Wikipedia articles", "mr.wikipedia.org, revisions in data/agri_text/provenance.json", "CC BY-SA 4.0 (attribution + share-alike)", "Raw text, includes a few citation lines", False),
    ("agri_varhadi_drafts.csv", "The same sentences converted to Varhadi by the adaptation layer (ByT5 model + word rules)", "scripts/draft_agri_varhadi.py", "CC BY-SA 4.0 (derived from Wikipedia)", "MACHINE DRAFT, not checked by a native speaker; `flag` marks 3 Wikipedia reference lines and 13 rows with Latin letters or digits", False),
    ("mr2vh_farming_dataset.csv", "MAIN OUTPUT: Marathi farming sentences (Wikipedia, bot's disease advice, recording prompts) translated to Varhadi by the adaptation layer; separate columns for the model stage and the word-rule stage", "scripts/build_mr2vh_farming_dataset.py", "Wikipedia rows CC BY-SA 4.0, other rows project's own", "MACHINE DRAFT, not checked by a native speaker; model trained on scripture, not farming text", False),
    ("recording_prompts.csv", "Prompts for collecting spoken Varhadi farming queries (elicitation + question prompts)", "written for this project", "project's own", "Draft, needs native review", True),
    ("native_verified_gold_eval.csv", "Native-speaker-verified test set for measuring the adaptation layer", "to be collected", "-", "EMPTY (header only): no native review has happened yet", False),
    ("VARHADI ADAPTATION LAYER (Marathi -> Varhadi)", None),
    ("mr2vh_parallel_all.csv", "All Marathi/Varhadi verse pairs; Varhadi is the VAHNT New Testament, Marathi side is machine-written back-translation", "data/translation/backtrans.jsonl", "VAHNT licence unverified, research use only", "Marathi side is machine output, no native verification", False),
    ("mr2vh_train.csv", "Training split of the pairs (by chapter)", "same", "same", "same", False),
    ("mr2vh_val.csv", "Validation split", "same", "same", "same", False),
    ("mr2vh_test.csv", "Test split", "same", "same", "same", False),
    ("mr2vh_test_predictions.csv", "Model output for the test split next to the reference", "data/translation/test_pred_v3.jsonl", "same", "Bible domain only; not evaluated on farming language", False),
    ("mr2vh_allowed_substitutions.csv", "Word pairs the model is allowed to change (guard against invented words)", "mined from the pairs", "same", "Unverified", False),
    ("varhadi_word_rules.csv", "Marathi -> Varhadi word rules applied after the model (नाही->नाई, आहे->हाय, ...)", "bot/varhadi_style.json, counted in VAHNT text", "project's own list of word forms", "Unverified by a native speaker", True),
    ("varhadi_lexicon_candidates.csv", "Varhadi word forms mined from the aligned audio-Bible text, with counts", "scripts/mine_lexicon.py", "VAHNT licence unverified", "Candidates only (verified = false)", False),
    ("varhadi_lexicon_high_confidence.csv", "The high-confidence subset of the candidates", "same", "same", "same", False),
    ("VARHADI SPEECH (VAHNT audio Bible)", None),
    ("vahnt_aligned_segments.csv", "Audio segments (start/end) aligned to VAHNT text with ASR text and review fields", "bible.com VAHNT audio Bible, version 3451", "Unverified licence, local research only", "Machine-aligned, review_status pending", False),
    ("vahnt_confidence_triage.csv", "Per-segment quality scores (CER vs ASR, silence, clipping, duration) and a triage label", "scripts in scripts/", "same", "Automatic scores", False),
    ("vahnt_audio_manifest.csv", "One row per chapter audio file (89 chapters) with licence note", "data/private/metadata/manifest.csv", "same", "Index only, audio not included", False),
    ("tts_luke_clips.csv", "Transcripts of the 349 training + 43 held-out narrator clips (Luke) used for the voice fine-tune; audio not included", "VAHNT audio Bible, Luke", "Unverified, private demo only, never publish", "Machine-aligned transcripts", False),
    ("OPEN DATA USED BY THE BOT", None),
    ("vidarbha_talukas.csv", "11 districts, 120 talukas with Marathi names and centre coordinates", "OpenStreetMap (Overpass API)", "ODbL, credit: OpenStreetMap contributors", "Names come from OSM, not checked against the LGD list", True),
    ("vidarbha_villages.csv", "13,000+ villages, towns and hamlets with coordinates per taluka (names mostly in Roman script)", "OpenStreetMap (Overpass API)", "ODbL", "Incomplete in places (OSM coverage)", True),
    ("disease_guidance.csv", "Marathi advice per crop and photo-disease class (soybean, orange); no products or doses", "written for this project", "project's own", "UNVERIFIED by an agronomist", True),
    ("disease_qa.csv", "Question/answer pairs for the disease feature", "data/disease_eval/qa_soy_citrus.json", "project's own", "Unverified", False),
]
OPEN = [c[0] for c in CATALOGUE if c[1] is not None and c[5]]


def readme() -> None:
    lines = ["# Datasets of KisaanDost", "",
             "Every dataset of the project as CSV (UTF-8 with BOM, opens in Excel). Regenerate with `python -m scripts.build_datasets_folder`.",
             "Audio files are not included. **Nothing here has been verified by a native Varhadi speaker yet**: the Varhadi in the "
             "drafts is machine output from a model trained on scripture text.", "",
             "Only the files marked *tracked* are meant for the public repo; the rest contain VAHNT scripture text (licence "
             "unverified) or Wikipedia text and are git-ignored.", ""]
    for entry in CATALOGUE:
        if entry[1] is None:
            lines += ["", f"## {entry[0].title()}", "", "| File | Rows | What it is | Source | Licence | Status | Git |", "|---|---|---|---|---|---|---|"]
            continue
        name, what, source, lic, status, tracked = entry
        lines.append(f"| `{name}` | {rows_written[name]:,} | {what} | {source} | {lic} | {status} | {'tracked' if tracked else 'local only'} |")
    lines += ["", "## Counts at a glance", "",
              f"- Farming sentences drafted into Varhadi: {rows_written['agri_varhadi_drafts.csv']}",
              f"- Verse pairs for the adaptation layer: {rows_written['mr2vh_parallel_all.csv']:,}",
              f"- Native-verified evaluation rows: {rows_written['native_verified_gold_eval.csv']}", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
    for n, c in rows_written.items():
        print(f"{c:>7,}  {n}")
