# Experiment Log

Every entry records only measured results. Unmeasured = PENDING, unavailable = NOT AVAILABLE.

---

## Exp 0.1 — Pretrained Marathi ASR, single 30 s sample

- **Date:** 2026-09-23
- **Objective:** Prove the raw baseline pipeline end to end on Varhadi speech.
- **Dataset:** `asr_audio/JHN_001_sample30s.wav`, the first 30 s of VAHNT John 1. General Varhadi scripture speech, not agricultural.
- **Model:** `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, CTC decoder, AI4Bharat NeMo 1.23.0rc0, torch 2.14.0+cpu.
- **Configuration:** no correction, no normalization, no fine-tuning.
- **Result:** transcribed in 3.75 s. The raw output is in `asr_outputs/predictions.csv`. WER/CER: NOT AVAILABLE (no reference).
- **Limitations:** a full 473 s chapter in one pass was OOM-killed at about 6 GB RSS.
- **Next step:** chunked inference.

## Exp 0.2 — Chunked inference on a full chapter

- **Date:** 2026-10-04
- **Objective:** Remove the memory ceiling so whole chapters can be transcribed. Same raw baseline, no correction.
- **Dataset:** `audio/JHN_001.mp3` (473.6 s), plus the 30 s sample for the comparison.
- **Model:** same as Exp 0.1.
- **Configuration:** chunks cut at the lowest-energy point, target 20 s, max 30 s, overlap 0 s (see `docs/asr_baseline.md`).
- **Result:** 19 chunks, 53.9 s inference (~8.8x realtime), peak RSS 1.74 GB. The stitched raw output is in `asr_outputs/predictions.csv` and the per-chunk outputs in `asr_outputs/chunks/JHN_001.csv`. WER/CER: NOT AVAILABLE.
- **QUALITATIVE OBSERVATION:** with 0 overlap at silence cuts, the 30 s sample's chunked output was nearly identical to the single pass (a few boundary spellings differ). Exact-match overlap stitching failed on a real seam, so it was replaced with a longest-common-run stitcher.
- **Limitations:** no references, so neither the effect of chunking nor the baseline accuracy can be scored.
- **Next step:** batch over all 89 downloaded chapters (approval needed), then P1 reference alignment.

## P1b pilot: VAHNT text forced alignment (JHN_001)

- **Date:** 2026-10-04
- **Objective:** Test whether the published VAHNT text can be aligned to the chapter audio to produce candidate paired (audio, text) segments for Varhadi adaptation (option b).
- **Dataset:** `audio/JHN_001.mp3` (473.57 s) and the published VAHNT text for John 1 (51 verses, 4 headings), fetched from bible.com.
- **Model:** the baseline CTC head (Marathi softmax), used only for alignment.
- **Configuration:** numpy CTC Viterbi with unaligned audio allowed at the start and end; segments of 20 s or less; log-probs computed with the inference chunk cuts (20/30/0).
- **Result:** 31 segments covering 6.32–471.56 s, computed in 87 s. In all 31 segments, the baseline decode of each span starts and ends on the same words as the aligned text (automatic check). `align_score` median -0.77.
- **QUALITATIVE OBSERVATION:** the text/ASR differences are mostly systematic Varhadi-vs-Marathi forms (काई/काही, नाई/नाही, त्याले/त्याला, पयले/पहिले).
- **Limitations:** no human listening check yet. The published text is unverified against the narration. The CER against the published text is NOT a verified baseline metric. Only one chapter has been tried.
- **Next step:** human spot-check a sample of segments, then scale to the downloaded chapters and import them as pending_review gold rows.

## Exp 0.3: Raw baseline over all downloaded VAHNT chapters

- **Date:** 2026-10-04 (started 2026-10-03, stopped at 41 files and resumed the next morning)
- **Objective:** Raw pretrained-Marathi transcriptions of every downloaded Varhadi chapter, for error analysis and as the baseline side of later comparisons. No correction, normalisation or fine-tuning.
- **Dataset:** 89 VAHNT chapters (705.9 min, 208–880 s each), speaker unknown, general scripture speech (not agricultural).
- **Model:** same as Exp 0.1. Chunked 20/30/0 (Exp 0.2).
- **Result:** 89/89 chapters transcribed, 0 failed, 0 empty outputs, 1,771 chunks. Throughput, measured on the 87 chapters that did not reuse earlier chunks: 689.0 min of audio in 72.6 min of inference (~9.5x realtime overall, per-chapter median 10.3x, range 4.4–11.5x). The slowest chapters overlapped with other jobs running on the same CPU. Outputs: `asr_outputs/predictions.csv`, `asr_outputs/chunks/`, `asr_outputs/baseline_report.md`.
- **WER/CER:** NOT AVAILABLE. No verified references yet; `metrics.json` records 0 evaluated samples.
- **Pipeline fix found here:** file-level resume compared raw path strings, so `JHN_001` had been recorded twice (once from a single-file run, once from the batch), which inflated the report to 91 files and 714.3 min. Paths are now stored repo-relative and normalised on resume. The duplicate row had identical text and was removed, keeping the original row with its real 53.9 s timing.
- **Limitations:** transcription only; scripture domain; a single unknown narrator (or narrators), so it says nothing about speaker variation.
- **Next step:** align the published text (P1b) to obtain candidate references, then human review.

## P1c: Gold review workflow and evaluation pipeline (tooling, no new measurements)

- **Date:** 2026-10-04
- **Objective:** Make human verification practical and safe, and have the first real WER/CER computable as soon as verified references exist.
- **Changes:**
  - **Gold schema:** `candidate_text` is now immutable (the old schema overwrote the candidate on review). Decisions are correct, needs_correction, rejected or needs_realignment.
  - **Review tool:** `scripts/review_server.py` is a local listening tool that only accepts a verifying decision after the clip has been played through.
  - **Scoring:** `scripts/run_gold_eval.py` refuses to score without verified segments and labels results PILOT by default. WER/CER plus substitutions, deletions and insertions, with punctuation-only normalisation.
  - **Error analysis:** word pairs with a string-similarity heuristic tag; categories are left for manual labelling.
  - **P2 preparation:** `compare.py`, the manifest builder, the fine-tune config and the Colab notebook.
- **Result:** the review queue holds 369 pending segments (12 chapters at import time). Verified segments: 0. **WER/CER: NOT AVAILABLE.**
- **Next step:** finish alignment, then a human reviews the JHN_001 segments, then run `run_gold_eval.py` for the first PILOT WER/CER.

## P1d: Confidence-based triage for gold review (prompt03)

- **Date:** 2026-10-04
- **Objective:** Cut manual review effort without treating automatic transcripts as ground truth.
- **Implementation:** `varhadi_data/triage.py` with `triage_config.json` (triage-v1) and `scripts/triage_gold_candidates.py`. The upgraded `scripts/review_server.py` orders the queue by triage and shows the scores and flags, with keyboard review and auto-advance. The gold schema gained triage and notes columns, and reviewer IDs are validated. `build_manifests.py` keeps gold-only by default and writes PSEUDO_LABELED data to separate files. Documented in `docs/gold_review.md`.
- **Formula:** confidence = 0.35·alignment + 0.30·ASR agreement + 0.15·audio quality + 0.10·duration + 0.10·speaking rate − artifact penalties. Thresholds: high-confidence ≥ 0.90 (final), reject < 0.50 (base, before penalties). Parentheses segments are never auto-rejected.
- **Calibration on the real distribution:** the first version would have auto-rejected 9 segments whose only problem was a translator note in parentheses (good audio). The policy was changed so correctable text artifacts only block high-confidence status. The thresholds are PROVISIONAL until compared with human decisions.
- **Result, all 1,296 candidates (40 chapters):** HIGH_CONFIDENCE_CANDIDATE 718 (unverified), NEEDS_REVIEW 573, REJECT_CANDIDATE 5. Confidence median 0.909 (p10 0.783). Flags: contains_heading 222, parentheses 22, speaking_rate_out_of_range 4, duration_out_of_range 1.
- **Result, JHN_001 pilot (31 segments):** NEEDS_REVIEW 11, HIGH_CONFIDENCE_CANDIDATE 20, REJECT_CANDIDATE 0. Confidence 0.821–1.0. JHN_001 is the TEST chapter, so all 31 still need human review; triage only orders them.
- **Human-verified segments: 0.** WER/CER NOT AVAILABLE. `run_gold_eval.py` exits safely ("No verified segments").
- **Limitations:** heuristic, uncalibrated weights; the agreement signal uses the baseline model (circular, so it can't select test data); ASR decoding stability is not available.
- **Next step:** human review of JHN_001 in the review server, then the first PILOT WER/CER, then compare triage against the human decisions and recalibrate.

## P1e: Candidate lexicon mining from aligned data (no model run)

- **Date:** 2026-10-12
- **Objective:** List candidate (published Varhadi form, Marathi-ASR rendering) pairs as a worklist for a native speaker. These are error observations, not a Varhadi→Marathi mapping: `standard_marathi` is always empty and `verified` always false.
- **Dataset:** `data/transcripts/gold_transcripts.csv`, 1,296 candidate segments (40 chapters), `candidate_text` vs `raw_asr` of the same row. Human-verified segments: 0.
- **Method:** `varhadi_data/lexicon.py` and `scripts/mine_lexicon.py`. Punctuation stripping only (`strip_punctuation`), whitespace tokens, `difflib.SequenceMatcher` (autojunk off). Only equal-length `replace` blocks are paired 1:1. Inserts, deletes and unequal-length replaces (word merges/splits) are counted and skipped. consistency = pair count / all pairs for that form. Ranking: count desc, then consistency desc. Output: `data/transcripts/lexicon_candidates.tsv` (gitignored).
- **Result, all segments:** 12,992 substitutions; skipped 1,098 unequal replaces, 51 deletes, 26 inserts; 2,435 unique forms, 3,545 unique pairs. count ≥ 3: 528; consistency ≥ 0.5: 2,509; both (rows written): 345.
- **Result, HIGH_CONFIDENCE_CANDIDATE only:** 718 segments, 6,642 substitutions (skipped 493/14/8), 1,603 forms, 2,171 pairs; count ≥ 3: 297; consistency ≥ 0.5: 1,656; both: 205.
- **QUALITATIVE OBSERVATION:** the top rows include the forms already noted in P1b (नाई/नाही, त्याले/त्याला, काई/काही) and more (तवा/तेव्हा, हाय/आहे, तुमी/तुम्ही, पायजे/पाहिजे). Several high-count rows are spelling or orthography differences rather than dialect (कि/की, अन्/अन, येशूनं/येशून). Some look like ASR misrecognitions (संग/संघ, माह्या/माया).
- **Precision of the list as dialect evidence:** PENDING (no native-speaker review).
- **Limitations:** the published text is unverified against the narration. Every pair depends on the baseline model. Merges/splits are not mined. A single narrator in a single domain.
- **Next step:** a native speaker fills `standard_marathi` and marks rows as dialect, orthography or ASR error. Measure precision on the top N.
