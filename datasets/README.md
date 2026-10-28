# Datasets of KisaanDost

Every dataset of the project as CSV (UTF-8 with BOM, opens in Excel). Regenerate with `python -m scripts.build_datasets_folder`.
Audio files are not included. **Nothing here has been verified by a native Varhadi speaker yet**: the Varhadi in the drafts is machine output from a model trained on scripture text.

Only the files marked *tracked* are meant for the public repo; the rest contain VAHNT scripture text (licence unverified) or Wikipedia text and are git-ignored.


## Created By This Project

| File | Rows | What it is | Source | Licence | Status | Git |
|---|---|---|---|---|---|---|
| `agri_marathi_sentences.csv` | 491 | Marathi farming sentences (cotton, soybean, orange, tur, general) split from Marathi Wikipedia articles | mr.wikipedia.org, revisions in data/agri_text/provenance.json | CC BY-SA 4.0 (attribution + share-alike) | Raw text, includes a few citation lines | local only |
| `agri_varhadi_drafts.csv` | 491 | The same sentences converted to Varhadi by the adaptation layer (ByT5 model + word rules) | scripts/draft_agri_varhadi.py | CC BY-SA 4.0 (derived from Wikipedia) | MACHINE DRAFT, not checked by a native speaker; `flag` marks 3 Wikipedia reference lines and 13 rows with Latin letters or digits | local only |
| `mr2vh_farming_dataset.csv` | 573 | MAIN OUTPUT: Marathi farming sentences (Wikipedia, bot's disease advice, recording prompts) translated to Varhadi by the adaptation layer; separate columns for the model stage and the word-rule stage | scripts/build_mr2vh_farming_dataset.py | Wikipedia rows CC BY-SA 4.0, other rows project's own | MACHINE DRAFT, not checked by a native speaker; model trained on scripture, not farming text | local only |
| `recording_prompts.csv` | 27 | Prompts for collecting spoken Varhadi farming queries (elicitation + question prompts) | written for this project | project's own | Draft, needs native review | tracked |
| `native_verified_gold_eval.csv` | 0 | Native-speaker-verified test set for measuring the adaptation layer | to be collected | - | EMPTY (header only): no native review has happened yet | local only |

## Varhadi Adaptation Layer (Marathi -> Varhadi)

| File | Rows | What it is | Source | Licence | Status | Git |
|---|---|---|---|---|---|---|
| `mr2vh_parallel_all.csv` | 3,695 | All Marathi/Varhadi verse pairs; Varhadi is the VAHNT New Testament, Marathi side is machine-written back-translation | data/translation/backtrans.jsonl | VAHNT licence unverified, research use only | Marathi side is machine output, no native verification | local only |
| `mr2vh_train.csv` | 3,222 | Training split of the pairs (by chapter) | same | same | same | local only |
| `mr2vh_val.csv` | 127 | Validation split | same | same | same | local only |
| `mr2vh_test.csv` | 206 | Test split | same | same | same | local only |
| `mr2vh_test_predictions.csv` | 206 | Model output for the test split next to the reference | data/translation/test_pred_v3.jsonl | same | Bible domain only; not evaluated on farming language | local only |
| `mr2vh_allowed_substitutions.csv` | 2,022 | Word pairs the model is allowed to change (guard against invented words) | mined from the pairs | same | Unverified | local only |
| `varhadi_word_rules.csv` | 92 | Marathi -> Varhadi word rules applied after the model (नाही->नाई, आहे->हाय, ...) | bot/varhadi_style.json, counted in VAHNT text | project's own list of word forms | Unverified by a native speaker | tracked |
| `varhadi_lexicon_candidates.csv` | 345 | Varhadi word forms mined from the aligned audio-Bible text, with counts | scripts/mine_lexicon.py | VAHNT licence unverified | Candidates only (verified = false) | local only |
| `varhadi_lexicon_high_confidence.csv` | 205 | The high-confidence subset of the candidates | same | same | same | local only |

## Varhadi Speech (Vahnt Audio Bible)

| File | Rows | What it is | Source | Licence | Status | Git |
|---|---|---|---|---|---|---|
| `vahnt_aligned_segments.csv` | 1,296 | Audio segments (start/end) aligned to VAHNT text with ASR text and review fields | bible.com VAHNT audio Bible, version 3451 | Unverified licence, local research only | Machine-aligned, review_status pending | local only |
| `vahnt_confidence_triage.csv` | 1,296 | Per-segment quality scores (CER vs ASR, silence, clipping, duration) and a triage label | scripts in scripts/ | same | Automatic scores | local only |
| `vahnt_audio_manifest.csv` | 89 | One row per chapter audio file (89 chapters) with licence note | data/private/metadata/manifest.csv | same | Index only, audio not included | local only |
| `tts_luke_clips.csv` | 392 | Transcripts of the 349 training + 43 held-out narrator clips (Luke) used for the voice fine-tune; audio not included | VAHNT audio Bible, Luke | Unverified, private demo only, never publish | Machine-aligned transcripts | local only |

## Open Data Used By The Bot

| File | Rows | What it is | Source | Licence | Status | Git |
|---|---|---|---|---|---|---|
| `vidarbha_talukas.csv` | 120 | 11 districts, 120 talukas with Marathi names and centre coordinates | OpenStreetMap (Overpass API) | ODbL, credit: OpenStreetMap contributors | Names come from OSM, not checked against the LGD list | tracked |
| `vidarbha_villages.csv` | 13,202 | 13,000+ villages, towns and hamlets with coordinates per taluka (names mostly in Roman script) | OpenStreetMap (Overpass API) | ODbL | Incomplete in places (OSM coverage) | tracked |
| `disease_guidance.csv` | 14 | Marathi advice per crop and photo-disease class (soybean, orange); no products or doses | written for this project | project's own | UNVERIFIED by an agronomist | tracked |
| `disease_qa.csv` | 16 | Question/answer pairs for the disease feature | data/disease_eval/qa_soy_citrus.json | project's own | Unverified | local only |

## Counts at a glance

- Farming sentences drafted into Varhadi: 491
- Verse pairs for the adaptation layer: 3,695
- Native-verified evaluation rows: 0
