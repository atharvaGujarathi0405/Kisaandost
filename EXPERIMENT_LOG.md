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

## P3a: Marathi -> Varhadi pairs by back-translation, and ByT5 fine-tune prepared (not trained)

- **Date:** 2026-10-26
- **Objective:** Use the local VAHNT Varhadi text to train the Marathi -> Varhadi layer of the bot instead of hand-written swap rules only.
- **Method:** `scripts/build_mr2vh_pairs.py` sends VAHNT verses to Groq (`openai/gpt-oss-120b`, temperature 0, prompt `bt-v1`) and asks for plain standard Marathi. Pair = (Groq Marathi -> real VAHNT Varhadi). Splits by chapter: test JHN_001, JHN_005, LUK_003, MAT_005, MRK_002; val JHN_002, LUK_004, MAT_006, MRK_003. Filters: Latin letters, length ratio outside 0.6-1.6, Varhadi marker words surviving on the Marathi side (copied, not translated), identical pair.
- **Data approval:** the user approved sending VAHNT text to Groq on 2026-10-26. VAHNT licence is still unverified (research use only); outputs are gitignored in `data/translation/`.
- **Model choice check:** on 8 verses `gpt-oss-20b` and `qwen3.8-27b` changed meaning (wrong subject, dropped phrases, invented words) while `gpt-oss-120b` stayed faithful, so only 120b is used. Groq free tier caps each model per day, so 120b throttled after 1,240 verses (about 55 verses/hour afterwards).
- **Result so far (1,240 of 3,695 verses, JHN and part of LUK):** dropped 33 (copied_varhadi 30, latin 1, length 1, identical 1). train 1,010, val 62, test 135 verses (3 test chapters, 2 val chapters).
- **Baselines on the 135 held-out verses (WER / CER vs real VAHNT):** Marathi unchanged 0.533 / 0.244; rules only (`varhadify`) 0.406 / 0.190. **Fine-tuned ByT5: PENDING** (not trained).
- **Trainer:** `experiments/varhadi_adaptation/mr2vh.py` (ByT5-small, `train` / `predict` / `score`). Checked on CPU with a tiny random T5 (train, predict, score all run); the real model, GPU memory and step time are UNMEASURED.
- **Limitations:** the Marathi side is machine-written, so pairs can carry Groq's paraphrases; the target side is scripture register from one translation team; 3 test chapters, no significance test; no native-speaker review.
- **Fine-tune run (Colab T4, 2026-10-26):** google/byt5-small, 1,008 train pairs (2 over 1,024 bytes dropped), 62 val, batch 2 x accum 8, lr 1e-4, 10 epochs, seed 13. First attempt at batch 4 ran out of GPU memory (14.5 GB) at step 5. Wall time 72.2 min (about 4.3 s per optimizer step). Validation loss 0.259 (epoch 1) -> 0.1742 (epoch 9, best, kept) -> 0.1745 (epoch 10). Predictions: beam 4, 135 test verses, no length limit beyond 1,024 bytes.
- **Test result, 135 held-out verses (JHN_001, JHN_005, LUK_003), WER / CER vs real VAHNT, PILOT, no significance test:**
  - Marathi unchanged 0.533 / 0.244
  - rules only (varhadify) 0.406 / 0.190
  - fine-tuned ByT5, raw 0.593 / 0.443 (WORSE than unchanged)
  - fine-tuned, length-guarded 0.333 / 0.177
- **Why raw is worse:** 21 outputs fall into repetition loops (the genealogy verses in LUK_003 repeat "X's son was Y"); 19 verses have WER above 1.0. Per-verse median WER is 0.360 for the model against 0.409 for the rules, and the model beats the rules on 86 of 135 verses.
- **Length guard:** fall back to `varhadify(source)` when output/source character ratio is outside 0.77-1.52 (the min and max of the training pairs). It fired on 22/135 verses (JHN_001 4/51, JHN_005 1/47, LUK_003 17/37). On the other 113 verses: unchanged 0.559 / 0.252, rules 0.425 / 0.194, model 0.336 / 0.179. CAVEAT: the guard was added after seeing the loops, and tested on the same 135 verses, so the guarded number is somewhat optimistic; it needs fresh held-out chapters.
- **Observed error type:** invented spellings (one output wrote `प्रगाश` where the text has `ऊजीळ`). Not measured systematically.
- **Not done:** deployment in the bot (model is only in the Colab runtime, ~1.2 GB, CPU speed unmeasured); native-speaker review; farm-domain sentences (data/agri_text has no Varhadi side yet); remaining 2,455 verses (the background back-translation is throttled by Groq, so train was only 1,008 pairs from JHN and part of LUK).
- **Next step:** save the model (Drive or download), back-translate more verses (MAT, MRK) and retrain; test on fresh chapters; have a native speaker rate model vs rules on farm sentences before it replaces the rules in the bot.

## P3b: Fine-tuned model put in the bot (CPU deployment checks)

- **Date:** 2026-10-26
- **What:** `bot/mr2vh_model.py` loads the fine-tuned ByT5 from `bot/models/mr2vh/` (gitignored; copy also on the user's Google Drive, My Drive/KisaanDost/mr2vh_model). `to_varhadi()` in `bot/server.py`: LLM Marathi -> per sentence ByT5 -> length guard -> attested-substitution filter -> negation/digit check -> word-swap rules. Any failure falls back to the rules. `KD_MR2VH=0` switches the model off.
- **Fidelity:** on CPU the model reproduced the Colab GPU output exactly on 11 of 12 verses (beam 4).
- **Why a word-level filter:** model word changes that appeared in the training pairs matched the real Varhadi 71% of the time (575 changes); changes not in the training pairs matched 21% (152 changes), several of them meaning errors (`माझ्याकडे` -> `माह्यापासून`, `त्याबद्दल` -> `त्यायले`). A test showed one meaning change by the unfiltered model on a bot-style sentence (`हवे असेल` -> `हाय`). The filter keeps the source word instead. Effect on the 135 test verses (beam-4 Colab predictions, verse level, production functions with the length guard, the filter and the negation check): WER 0.339 / CER 0.180, against 0.333 / 0.177 with the length guard alone. The length guard fell back to rules on 22 verses, the negation check on 6 more.
- **Speed, this PC (8-core CPU, fp32, 133-character sentences):** beam 4 21.5 s per sentence, beam 1 10.7 s; WER 0.338 vs 0.343 on 24 verses, so beam 1 is the default. int8 dynamic quantisation: no speed-up (12.7 s vs 12.5 s) and different outputs, dropped. Model load 8-54 s (disk cache dependent).
- **End to end (Groq gpt-oss-20b + model, beam 1):** 29 s and 16 s for two farm questions (3-6 sentences each). Too slow for field use on this machine; a GPU host or a faster model is needed.
- **Quota conflict found:** the Groq free-tier daily cap is per model; the background back-translation job and the bot's default chat model (gpt-oss-120b) share it, so the bot returned 429 until the job was stopped. The job is paused at 1,392 of 3,695 verses (resumable).
- **Tests:** 104 passed across the whole repository.
- **Not measured:** quality on farm sentences (no Varhadi farm text exists), native-speaker rating, behaviour on long replies.
