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

## P3c: Why the bot sounded like Marathi, and the data state (2026-10-26)

- **Symptom:** the user said the bot was not speaking Varhadi.
- **Diagnosis (stage by stage on two farm questions):** the raw ByT5 output did contain Varhadi forms (`हायत`, `ओयखा`, `पानायवर`, `पिकाले`, `अन्`, `असतीन`, `रोगायन`), but the attested-substitution filter (P3b) reverted most of them, because farm vocabulary never occurs in the scripture training pairs. The Groq model also writes long, formal Marathi, and the browser voice is a Marathi voice.
- **Looser filter tested:** accepting unseen word changes that follow edit patterns seen in training (k >= 3, 5, 10, 20) matched the real Varhadi only 19-20% of the time (91-108 changes), no better than the 21% for all unseen changes. No safe middle setting exists with this data.
- **Change:** `KD_MR2VH_FILTER=strict` (default) or `off` in `bot/mr2vh_model.py`. `off` keeps all model word changes (length and negation guards stay on). In `off` mode a test reply gave `... अन् ... हाय ... काई ... देसत हाय ... मालूम दे ...`. Some changes there are probably wrong (`-आय-` on non-human nouns such as `लक्षणायबद्दल`; the training data shows it on people only). Not native-speaker checked.
- **Data state:** back-translation 2,800 of 3,695 verses (JHN 873, LUK 1,117, MAT 749, MRK 61); a second Groq organization's key was used after the first hit its 200,000-token daily cap (the cap refills continuously, about 8,300 tokens an hour). Splits: train 2,382 / val 127 / test 206 (fresh test chapters MAT_005 and MRK_002). A v2 retrain on Colab was started and aborted; nothing from it was saved. The old-model check on the fresh chapters stopped at 30 of 71 verses.
- **Bot default chat model:** now `openai/gpt-oss-20b`, so the data job can use `gpt-oss-120b` quota.
- **Next step:** finish the remaining 895 verses, retrain v2, check the length guard and filter on MAT_005/MRK_002, and get farm-domain Varhadi (`data/agri_text/sources.tsv`, 491 sentences) translated by a native speaker.

## P3d: v3 fine-tune on all back-translated data (2026-10-27)

- **Data:** all 3,695 verses back-translated (Groq gpt-oss-120b; 140 pairs dropped by filters, 137 of them `copied_varhadi`). Splits by chapter: train 3,222 (3,216 under 1,024 bytes), val 127 (JHN_002, LUK_004, MAT_006, MRK_003), test 206 (JHN_001, JHN_005, LUK_003, MAT_005, MRK_002). MAT_005 and MRK_002 (71 verses) were never seen by v1 either.
- **Training (Colab T4):** ByT5-small, batch 2 x accum 8, lr 1e-4, 5 epochs, 1,005 optimizer steps, 95.5 min (about 5.7 s/step by the end). Validation loss 0.2130, 0.1897, 0.1785, 0.1732, 0.1719 (epochs 1-5). No GPU memory error. Outputs saved to Google Drive: My Drive/KisaanDost/mr2vh_model_v3 (model.safetensors 1,198,571,496 bytes, configs, tokenizer files, test_pred.jsonl).
- **Result, 206 held-out verses, WER / CER vs real VAHNT (PILOT, no significance test):**
  - Marathi unchanged 0.539 / 0.248; word rules only 0.405 / 0.188; v3 raw 0.396 / 0.260.
  - v3 + length and negation guards, filter off 0.324 / 0.168; v3 + all safeguards (strict filter, bot default) **0.324 / 0.170**.
  - Fresh chapters (71 verses): rules 0.402 / 0.186; v1 with all safeguards 0.350 / 0.182; v3 with all safeguards 0.335 / 0.177.
  - Old test chapters (135 verses): v1 with all safeguards 0.339 / 0.180; v3 0.318 / 0.166.
  - The length guard fired on 15 of 206 verses (v1: 22 of 135).
- **Check of the guard on fresh chapters (the P3a caveat):** v1 on the 71 unseen Matthew/Mark verses: raw 0.393 / 0.240, guards 0.348 / 0.181, all safeguards 0.350 / 0.182, rules 0.402 / 0.186. The guard did not break down on fresh text (the negation check fell back on 10 of 71 verses).
- **Not done:** v3 is NOT yet installed in the bot (the bot still runs v1 from bot/models/mr2vh/; the 1.2 GB v3 file must be copied from Drive, together with `data/translation/allowed_subs_v3.json` renamed to `allowed_subs.json`); native-speaker review; farm-domain Varhadi; CPU latency unchanged (about 16-29 s per reply).

## P3e: v3 installed in the bot, filter tightened (2026-10-27)

- **Install:** v3 downloaded from Google Drive (zip 1,110,913,217 bytes, model.safetensors 1,198,571,496 bytes, zip integrity OK), placed in `bot/models/mr2vh/`; v1 kept in `bot/models/mr2vh_v1/` (both gitignored). CPU output matched the Colab GPU output exactly on 6 of 6 test verses (beam 4).
- **Filter list (`allowed_subs.json`, rebuilt from the 3,222 training pairs):** the larger set made the allowed list noisy (2,022 pairs, including alignment artifacts and context-dependent pairs such as का -> कावून, which turned "whether" into "because" in a live reply). Added `min_share` = 0.2 (the change must occur in at least 20% of the word's occurrences): 1,234 pairs, WER 0.324 -> 0.328 on the 206 held-out verses (every bad pair found had a share of 0.00-0.17).
- **Filter loophole closed:** the strict filter used to keep any word the model inserted, deleted, merged or split (this let `पिकायबाबत`, a -आय- form on a non-human noun, through). Now inserts are dropped and deletes/merges/splits revert to the source words: WER 0.328 / CER 0.165 (CER 0.172 before). With this the strict output can only contain source words, attested changes, and the word-swap rules.
- **Chat model:** back to `openai/gpt-oss-120b` (the data job that shared its quota is finished); prompt now says never to reply with only a question. A sample of 5 clear questions gave guidance in 5 of 5 replies; two earlier live replies had been question-only (prompt variance, not measured further). The farm advice itself is unverified (for example it suggested washing leaves or pruning for cotton pests).
- **End to end on this PC:** 16-23 s per reply (Groq plus the CPU model, 1 beam).
- **Tests:** 111 passed.
- **Not done:** native-speaker review; farm-domain Varhadi; commit and push of the day's work.

## P4: Crop photo -> possible disease (soybean and orange), independent test and guardrails (2026-10-27)

- **What was built:** `bot/disease.py`, `POST /api/diagnose`, a camera button and result card in `bot/web/index.html`, advice in `bot/disease_guidance.json` (precomputed into Varhadi by `scripts/build_disease_guidance.py`). Flow: the farmer picks the crop, takes a photo (resized to 1,024 px in the browser), the server checks photo quality, runs the crop's classifier, and returns a possible disease, the model's score, safe steps and a crop-specific warning. The photo is only held in memory; nothing is stored or logged.
- **Models (not trained by us):** `ixrbhii/soybean-disease-convnext` (6 classes) and `ixrbhii/citrus-disease-convnext` (8 classes), ConvNeXt-Small, CC BY 4.0 (credit "Multi-crop disease models, IXR (agritechixr)" and the LeafNet dataset). Model cards report 91.7% / 99.5% on their own held-out photos and warn that real field photos score lower.
- **Independent test (public photos the models were not tested on):** soybean = Project-AgML India soybean dataset (294 of 1,176 photos, every 4th); orange = Project-AgML citrus fruit-and-leaf dataset (253 of 759, every 3rd). Only classes that exist in both label sets can be scored as right or wrong.
  - Soybean: Septoria brown spot 65/71 correct (92%). Healthy leaves: only 38/72 called Healthy, and 29/72 called a disease at >= 45% confidence (mostly Downy Mildew). Bacterial blight, vein necrosis and dry leaf are not in the model: 97 of 151 such photos were still answered at >= 45% confidence.
  - Orange: canker 77/80 (96%), black spot 56/64 (88%), healthy 16/26 (10/26 called a disease). **Citrus greening (HLB, not in the model's classes): 67 of 73 photos called "Black Spot" with confidence >= 45%** (all 73 above 45%).
  - Precision on known diseases when the model answers: soybean 73% at 0.45, 76% at 0.60, 78% at 0.75; orange 88% / 91% / 93%.
- **Guardrails chosen from these numbers:** (1) confidence bar raised from 0.45 to 0.60; (2) a "Healthy" prediction is shown as "could not recognise a disease; this does not mean the crop is healthy" because healthy calls were wrong about half the time; (3) every orange result carries a standing warning that greening (yellow mottled leaves, small lopsided fruit) is not recognised; (4) every soybean result lists the six diseases it knows; (5) the score is labelled "the model's score, can be wrong"; (6) photo checks: too small, too dark or bright, blurry.
- **Blur check calibration:** 128 public leaf photos: sharp photos had Laplacian variance of at least 16.7 (5th percentile 33.6, median 128.7); threshold 15 flags 0 of 128 sharp photos. Gaussian blur of radius 4 mostly falls below it (98 of 128 at a threshold of 12, 109 at 20); radius 2 is mostly not caught. These are tidy dataset photos, not phone photos from a field.
- **Not supported:** tur (no model, no public disease dataset found) and cotton (no ready model; public cotton data exists, `Project-AgML/cotton_leaf_disease_classification`, 1,373 raw photos, 5 classes, CC BY-NC 4.0, so fine for this academic project, not for commercial use). Both show an honest "not available yet" message.
- **Guidance text:** the `ixrbhii/crop-disease-qa` dataset has no real content for soybean or citrus (placeholder rows only), so the advice was written from general crop-protection practice. It names no products and gives no doses (a test checks this). Not checked by an agronomist.
- **Voice:** the read-aloud button uses the browser's Marathi voice. There is no Varhadi voice.
- **Limits of this evaluation:** one sample of public photos, no field photos from Vidarbha, no significance test; the models' own training data may overlap with those public sets.
- **Next step:** train a cotton classifier on Colab; get field photos of the four crops from farmers (with consent) to measure real accuracy; add a way to say "none of these" (needs unknown-disease training photos, for example greening and dieback for orange); agronomist review of `bot/disease_guidance.json`.
- **Bug found while reviewing the generated advice:** the Marathi->Varhadi model sometimes appends a stray closing quote mark (it learned from verses full of quotes; 4 of 51 advice lines, and one chat reply earlier). `translate_text` now removes quote marks the source did not have (test added), and the 4 lines in `bot/disease_guidance.json` were cleaned. Reading the 51 converted lines also showed `बाजूला` -> `बाजुले` (the dative ending on a place word, probably wrong); left as is for the native-speaker review.

## P5: Microphone input redesigned (2026-10-27)

- **Problem reported:** speech was cut off at the first pause and half-finished words were sent. Cause: one `SpeechRecognition` session, `continuous = false`, and the text was sent as soon as the browser ended the session.
- **Change (`bot/web/index.html`):** `continuous = true`; when the browser ends a session on its own the page starts a new one and keeps the text; the live text is shown in the box; the question is sent only when the farmer taps the stop button; an X button and Esc cancel; a timer shows the recording is alive; 5-minute safety stop; a mic-blocked error stops cleanly and 3 network errors in a row stop it; results that only extend the previous result are merged (an Android Chrome quirk repeats words otherwise).
- **Checked with a simulated speech engine in Chrome (5 scenarios, all as intended):** text kept across a restarted session and sent once on stop; cancel sends nothing and restores the box; cumulative results are not repeated; "not allowed" stops without restarting; 3 network errors stop it.
- **NOT checked:** a real microphone and real speech on a phone, and the recognition accuracy for Varhadi words (the browser's Marathi model, run by the browser vendor, needs internet). If accuracy is poor, the alternative is to record the whole clip and transcribe it with Groq `whisper-large-v3`, which supports Marathi (no live text, uses Groq quota); not built.

## P6: Server-side voice for the bot (Marathi voice; Varhadi voice NOT trained) (2026-10-27)

- **Why:** the bot read answers with the browser's own Marathi voice, which differs by device and is missing on many desktops. Asked for a "Varhadi TTS model".
- **What was built (`bot/tts.py`, `POST /api/tts`, `GET /api/tts/info`, page changes in `bot/web/index.html`):** server voice `facebook/mms-tts-mar` (VITS, 36.3M parameters, CC BY-NC 4.0, 16 kHz). It reads Devanagari directly but has no punctuation or digits in its vocabulary, so the text is split into sentences and clauses and pauses are inserted (380 ms at sentence ends, 170 ms at commas); characters outside the vocabulary are dropped; the same text gives the same audio (fixed seed); a 64-entry cache. The page requests one sentence at a time and prefetches the next while one plays, and falls back to the browser's Marathi voice if the server voice fails. `KD_TTS=0` turns it off; a voice placed in `bot/models/tts/` (or `KD_TTS_DIR`) is used automatically.
- **MEASURED on this PC (CPU):** model load about 42 s the first time (preloaded in the background at start-up); 3.4 s of speech in 3.1 s (first call), then about 2x faster than real time; first sentence of a reply heard after about 5 s; repeated text is instant from the cache. In the browser a 9.8 s reply played from the server voice.
- **What this is NOT:** it is a MARATHI voice reading Varhadi spellings, not a Varhadi voice. I cannot judge pronunciation by ear: a native speaker has to listen to `data/tts_eval/mms_*.wav` and rate it. Varhadi spellings such as नाई, हाय, अन्, पानायवर are read the way Devanagari is normally read.
- **Data available for a real Varhadi voice (narrator recordings of the published text):** 718 high-confidence aligned segments = 3.02 h (median length about 15 s); only 304 segments (0.98 h) are 2-15 s with no brackets, digits or headings; plus 167 (0.55 h) from the needs-review group. The speaker field is "unknown", so the recordings may contain more than one narrator, which would make a single consistent voice impossible.
- **Open decision (not started):** fine-tune `facebook/mms-tts-mar` on the narrator clips on Colab. Concerns: (1) it copies the voice of a real narrator whose recordings have an unverified licence (private research use only, never publish the model); (2) 1-3 h of possibly mixed-speaker audio with unverified text may give poor or unstable speech; (3) quality can only be judged by a listener. Alternatives checked: `ai4bharat/indic-parler-tts` (Apache 2.0, 0.9B parameters, gated, needs a GPU), `ai4bharat/IndicF5` (MIT, gated), `SYSPIN/Marathi_Male_TTS`.
- **Tests:** 7 new (pauses, long clauses, vocabulary cleaning, WAV format and duration, cache, empty text, endpoint errors). 132 tests pass in total.

## P7: Varhadi voice fine-tune (MMS-TTS Marathi on narrator clips) (2026-10-28)

- **What:** fine-tuned `facebook/mms-tts-mar` on 349 clips (about 1.4 h) of the VAHNT narrator reading Luke, with the `ylacombe/finetune-hf-vits` recipe pinned to commit 6f3f51f (`experiments/varhadi_tts/tts_finetune_kaggle.ipynb`; Colab version beside it). 100 epochs = 4,400 steps, batch 8, lr 2e-5, fp16, seed 456; Kaggle T4, 1 h 16 min. Generator-only export, 145 MB, in `bot/models/tts/` (gitignored: unverified licence, a real narrator's voice, **private demo only, never publish**).
- **What went wrong on the way:** the first Colab run finished but the weights were lost when the VM was recycled before download; Colab's free GPU quota then ran out. The first Kaggle run (T4 x2) crashed at about step 2200 (`ChildFailedError`: accelerate started 2 processes); the real error was hidden by my log filter. Fix: one GPU (`CUDA_VISIBLE_DEVICES=0`, `--num_processes 1`) and the full log kept in `train_full.log`. Second run completed. Step loss stayed at about 22-28 throughout (it is a noisy sum of terms, not a quality score).
- **Check (`scripts/eval_tts.py`, 20 held-out Luke sentences, speak, then Whisper-large-v3 (Marathi) transcribes, CER against the Varhadi text):** real narrator **0.263**, generic MMS Marathi voice **0.260**, fine-tuned voice **0.360**.
- **How to read this (honest):** by this proxy the fine-tuned voice is LESS recognisable to a Marathi recogniser than the generic voice (0.36 vs 0.26). That does not mean it is worse: Whisper is a Marathi model that corrects Varhadi forms towards standard Marathi, CER says nothing about accent or naturalness, and n = 20. It does mean there is no evidence yet that the voice is better. What is needed: a native Varhadi speaker listening to `data/tts_eval/voices/hub_*.wav` (generic) against `tts_*.wav` (fine-tuned) and rating pronunciation and naturalness.
- **Known limits:** 1.4 h of one narrator, scripture register; one training run, no hyper-parameter search; the voice is that narrator's voice, not a farmer's; the server loads it automatically when `bot/models/tts/model.safetensors` exists (`bot/tts.py`).
- **Next step:** native listening test; if it is not clearly better, keep the generic voice as default (delete `bot/models/tts/` or set `KD_TTS_DIR` elsewhere).
