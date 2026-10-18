# KisaanDost: Technical Workflow (from the farmer's spoken question to the final answer)

Last updated: 2026-10-12

This document explains how KisaanDost works stage by stage, which tools and libraries each stage uses, and
how far each stage has been built. It is written for the project mentor.

Each stage is labelled with one of three statuses:

| Label | Meaning |
|---|---|
| **BUILT** | The code exists in this repo, has tests, and has been run on real data. |
| **PREPARED** | The code and config exist, but the step has not been run yet (for example, training). |
| **PLANNED** | Designed in `prompt.md` but not written yet. The tools listed are the planned choices. |

Short version: the project's research core is a speech recogniser that understands Varhadi. That part is
built: the baseline ASR, the dataset, alignment, review and evaluation tooling. Fine-tuning is prepared.
Everything after the transcript (normalisation, intent, RAG, LLM, TTS, UI) is planned.

---

## 1. What the system is

KisaanDost is a voice assistant for farmers in Vidarbha, Maharashtra, who speak **Varhadi**, a dialect of
Marathi. A farmer asks a farming question by voice (later also by text, photo or phone call) and gets a
short, safe answer in Marathi/Varhadi, both spoken and on screen.

**Main research contribution:** Indian ASR models are trained on standard Marathi. They make systematic
mistakes on Varhadi word forms, for example `नाई` (Varhadi) → `नाही` (Marathi), `त्याले` → `त्याला`,
`काई` → `काही`. We take a pretrained Marathi ASR model, **adapt (fine-tune) it on Varhadi speech**, and
measure both models on the same unseen test set with **WER and CER**:

```
Experiment A: pretrained Marathi ASR   vs   Experiment B: Varhadi-adapted ASR
              (same held-out test set, same metrics)
```

We are not training an ASR model from scratch, and we are not creating a new language.

---

## 2. End-to-end flow at a glance

```
 FARMER speaks: "माझ्या कपाशीला किडे लागले" ("Maza kapusala kide laglet")
    │
    ▼
 [1] Capture: React web app (mic) / IVR phone call       PLANNED
    │  audio (webm/m4a/8 kHz phone audio)
    ▼
 [2] Input validation + security (FastAPI)              PLANNED
    │  size/MIME check, auth, rate limit
    ▼
 [3] Audio preprocessing → 16 kHz mono WAV              BUILT   (librosa, soundfile)
    ▼
 [4] Chunking long audio at silences                    BUILT   (numpy, difflib)
    ▼
 [5] Speech-to-text: IndicConformer (Marathi)           BUILT   (AI4Bharat NeMo, PyTorch)
     → Varhadi-adapted checkpoint                       PREPARED (Colab GPU fine-tune)
    │  raw transcript (Devanagari text)
    ▼
 [6] ASR confidence check → "please repeat?"            PLANNED
    ▼
 [7] Varhadi normalisation (rules + lexicon)            PARTLY BUILT (lexicon mining only)
    │  canonical Marathi text
    ▼
 [8] Intent + entity extraction                         PLANNED
    │  {intent: pest_problem, crop: cotton, ...}
    ▼
 [9] RAG: retrieve verified Vidarbha agri documents     PLANNED (ChromaDB/Qdrant)
    ▼
 [10] LLM: grounded answer from retrieved text only     PLANNED
    ▼
 [11] Safety check (pesticides, low evidence, escalate) PLANNED
    ▼
 [12] Text-to-speech in Marathi                         PLANNED
    ▼
 FARMER hears the answer + sees it on screen (React)    PLANNED
```

Security, privacy, logging and evaluation apply to every stage (see sections 6 and 7).

---

## 3. Stage by stage

### Stage 1: Capture the farmer's voice (PLANNED)

- **What happens:** the farmer taps one large mic button in a mobile-first web app and speaks. Later, the
  farmer can also dial a phone number (IVR) with no smartphone needed.
- **Tech:** React + TailwindCSS (Material UI where useful), using the browser `MediaRecorder` API to record
  audio. IVR will be an adapter: the telephony provider sends audio to a FastAPI webhook or websocket, and
  the rest of the pipeline is the same.
- **Design rule:** the core AI logic is not tied to the web UI, so the same pipeline serves web and phone.
- **Output:** an audio file (for example webm/opus) sent to the backend.

### Stage 2: Backend entry, validation and security (PLANNED)

- **What happens:** the FastAPI endpoint (planned `backend/app/api/speech.py`) receives the audio. It checks
  the login token, role (FARMER / EXPERT / ADMIN), file size and MIME type, and the rate limit. Then it
  passes the audio to the ASR service.
- **Tech:** FastAPI + Pydantic schemas, PostgreSQL for users, conversations and audit logs (with
  migrations), and file storage for audio (not stored inside the database).
- **Data stored:** pseudonymous IDs only. No unnecessary phone numbers or names.

### Stage 3: Audio preprocessing (BUILT)

- **What happens:** any input format (MP3, m4a, wav) is decoded, resampled to **16 kHz**, mixed down to
  **mono**, and cached as a WAV file. The model was trained on 16 kHz mono audio, so this must match. The
  original file is never modified.
- **Tech:** `librosa` (decode, resample, mono), `soundfile` (write WAV).
- **Code:** `asr_baseline/audio_utils.py`: `validate_audio()` rejects missing, undecodable or zero-length
  files. `ensure_wav_copy()` converts once and reuses the cached copy.

### Stage 4: Chunking long audio (BUILT)

- **Problem found:** transcribing a full 8-minute chapter in one pass was **killed for running out of
  memory (about 6 GB)** on our CPU machine. The Conformer's self-attention memory grows with the square of
  the input length.
- **Solution:** split the audio into roughly 20–30 s chunks. Each cut is placed at the **quietest 25 ms
  frame** between 20 s and 30 s, so it usually falls in a pause between words. Each chunk is transcribed
  separately, and the texts are joined in order.
- **Stitching:** with overlap 0 (the default), chunks are joined with a space and no words are duplicated.
  With overlap > 0, we find the **longest common run of at least 2 words** between neighbouring chunks
  (Python `difflib.SequenceMatcher`). We first tried exact suffix/prefix matching, but it failed on a real
  seam, so we replaced it.
- **Robustness:** each chunk's result is saved to `asr_outputs/chunks/<file>.csv`. After a crash, only
  the unfinished chunks are redone.
- **Tech:** `numpy` (RMS energy per frame), `difflib` (stdlib).
- **Code:** `asr_baseline/chunking.py` (`plan_chunks`, `stitch`, `transcribe_chunked`).
- **Measured:** JHN_001 (473.6 s) → 19 chunks, 53.9 s of inference, **peak memory 1.74 GB** (down from
  being killed at about 6 GB).

### Stage 5: Speech-to-text (ASR): the research core

**5a. Baseline model (BUILT)**

| Item | Value |
|---|---|
| Model | `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large` (AI4Bharat, Hugging Face, gated) |
| Architecture | Conformer encoder: 17 layers, d_model 512, 8 attention heads, 4× subsampling |
| Decoders | Hybrid: RNN-T plus an auxiliary **CTC** head. We decode with CTC. |
| Tokenizer | SentencePiece BPE: 22 languages × 256 tokens, with a separate softmax per language. Marathi = `mr`. |
| Features | 80-dim log-mel spectrogram, 25 ms window, 10 ms stride |
| Framework | **AI4Bharat fork of NVIDIA NeMo** (1.23.0rc0), **PyTorch** (CPU build) |
| Runs in | WSL2 Ubuntu. NeMo's `pynini` dependency has no Windows build. |

How the model processes one chunk:
1. The 16 kHz waveform becomes a log-mel spectrogram: a picture of which frequencies are present every 10 ms.
2. The Conformer encoder (convolution for local sound patterns, self-attention for long-range context)
   turns this into one vector per 40 ms frame.
3. The CTC head gives, for each frame, a probability for every Marathi subword token plus a "blank".
4. Greedy CTC decoding picks the best token per frame, merges repeats, drops blanks, and joins the subwords
   into Devanagari words.

**Code:** `asr_baseline/model.py`, with fixes we had to make:
- `ASRModel.from_pretrained()` guesses the wrong checkpoint filename for this repo. We use
  `huggingface_hub.hf_hub_download()` + `ASRModel.restore_from()` instead.
- NumPy 2 removed `np.sctypes`, which NeMo still reads. We added a small compatibility shim instead of
  downgrading NumPy.
- In CTC mode, `transcribe()` returns a nested list, so we unwrap it.
- The `HF_TOKEN` is read from `.env`, which is never committed.

**Measured baseline run (Experiment 0):** all 89 downloaded chapters (705.9 min of audio) were transcribed
in 1,771 chunks with 0 failures, at about **9.5× faster than real time on CPU**. Output:
`asr_outputs/predictions.csv`. The output is kept **raw** (no correction) so the comparison with the
adapted model is fair.

Example raw output (John 1, first 30 s): `योहानाची सुवारता अध्याय एक प्रारंभी शब्द होता …`, recognisable,
with visible errors such as `स््रूप`.

**5b. Varhadi-adapted model (PREPARED, not trained)**
- **Method:** transfer learning. Start from the same checkpoint and keep the tokenizer unchanged. Fine-tune
  on Varhadi (audio, transcript) pairs with NeMo's `speech_to_text_finetune.py`.
- **Config:** `experiments/varhadi_adaptation/finetune_varhadi.yaml`. Every value is tagged with its source
  (the checkpoint or the fork's reference config): AdamW lr 1e-4, cosine annealing, batch 8, SpecAugment,
  max 30 s per segment, best checkpoint chosen by `val_wer`.
- **Where:** Google Colab GPU (`colab_finetune.ipynb`). The local machine is CPU-only and is used for data
  preparation, alignment and evaluation.
- **Training data format:** NeMo JSONL manifests built by `build_manifests.py`. Splits are by chapter: the
  test and validation chapters never appear in training. **The test split only contains human-verified
  segments.**
- **Why it is not trained yet:** see section 4 and `P2_READY.md`. There are 0 verified test segments so far.

### Stage 6: ASR confidence check (PLANNED)

If the transcript looks unreliable (low decoder confidence, or too short), the system asks the farmer to
repeat or confirm instead of guessing: "तुम्ही असं म्हणालात का: …?" ("Did you say …?")

### Stage 7: Varhadi normalisation (PARTLY BUILT)

- **What happens:** convert Varhadi word forms in the transcript to a canonical form, so that later stages
  (and the knowledge base, written in standard Marathi) can match them. Example: `कपाशीले` → `कपाशीला`,
  `नाई` → `नाही`. This is a separate layer after ASR, not part of the ASR model.
- **Planned order:** deterministic rules → versioned lexicon → agricultural terminology map → LLM fallback
  only when needed (not an LLM call for every word).
- **Built so far (P1e):** `varhadi_data/lexicon.py` + `scripts/mine_lexicon.py` compare the published
  Varhadi text with what the Marathi model heard (`difflib` word alignment) and list frequent, consistent
  word pairs. This gave 345 candidate rows, such as `तवा/तेव्हा`, `हाय/आहे`, `तुमी/तुम्ही` and
  `पायजे/पाहिजे`. These are **candidates only**. A native speaker must confirm each row and classify it as
  dialect, spelling difference or ASR error before it is used.

### Stage 8: Intent and entity extraction (PLANNED)

- **Output:** structured JSON, for example

```json
{ "intent": "pest_problem",
  "entities": { "crop": "cotton", "issue": "pest", "growth_stage": null },
  "location": "Vidarbha" }
```

- **Intents:** crop_problem, disease_identification, pest_problem, fertilizer_query, irrigation_query,
  weather_query, crop_practice, government_scheme, market_information, general_agriculture, unknown.
- **Method:** keyword and lexicon rules first (`कपाशी` → cotton, `किडे`/`अळी` → pest), then structured LLM
  extraction only for unclear cases. This module is kept independent from RAG.
- **Crop scope (proposed, to be confirmed):** cotton, soybean, orange, tur (`data/prompts/crop_scope.json`).

### Stage 9: Retrieval (RAG) over verified agricultural knowledge (PLANNED)

- **Knowledge base:** a small, trusted set of sources only: ICAR, PDKV Akola (the agricultural university
  for Vidarbha), state agriculture department documents and government scheme pages. Each document records
  source, date, region, crop, topic, licence and verification status.
- **Pipeline:** clean → chunk → embed with a multilingual embedding model → store in **ChromaDB or Qdrant**
  → retrieve the top-k chunks, filtered by the extracted crop and topic → optional reranking.
- **Low-evidence rule:** if retrieval scores are low, the system does not answer from guesswork. It asks a
  clarifying question, says verified information is not available, or escalates to an expert.
- **LangChain** will be used only where it saves real work (for example document loaders and splitters).
  It is not a hard dependency.

### Stage 10: LLM answer generation (PLANNED)

- **Input:** normalised question + intent/entities + retrieved chunks + vision result (if a photo was sent).
- **Rules:** answer only from the retrieved text, briefly and in farmer-friendly Marathi, and name the
  source. System instructions are kept separate from user text and retrieved documents. Both of those are
  treated as untrusted data, which protects against prompt injection.
- **Model:** not chosen yet. It will be called from the backend only. No API keys in the frontend.

### Stage 11: Safety layer (PLANNED)

| Situation | Action |
|---|---|
| ASR unsure | Ask the farmer to repeat or confirm |
| No supporting evidence found | Do not make up an answer; say so or escalate |
| Pesticide or chemical dose advice | Only from a verified source, with safety information; otherwise refer to an expert |
| Vision model unsure | Ask for a better photo or expert verification |
| Ambiguous question | Ask a clarifying question |
| Critical case | Flag it for the EXPERT review dashboard |

### Stage 12: Text-to-speech and reply (PLANNED)

- A replaceable TTS service, starting with an existing Marathi/Indic TTS model. We will **not** call it
  "Varhadi TTS" until it has been adapted and checked by native speakers.
- The farmer hears the answer and sees it as text with its source. Intermediate steps (raw transcript,
  intent, retrieved chunks) are shown only in developer/debug mode.

### Side branch: photo of a crop (PLANNED)

Image upload → size/type validation → image quality check → a replaceable plant-disease model gives a label
and confidence → RAG guidance. The system never claims a definite diagnosis and never prescribes pesticides
on its own.

---

## 4. The data pipeline behind the ASR research (BUILT)

There was **no Varhadi speech with matching transcripts** at the start. Without it we cannot fine-tune or
measure WER/CER. Most of the work so far is about building such data honestly.

```
 bible.com VAHNT audio (Varhadi-Nagpuri New Testament)
    │  downloader.py + browser.py   (Playwright headless Chromium + requests)
    ▼
 89 chapter MP3s, ~706 min                                         audio/
    │  scripts/run_asr_baseline.py   (raw baseline, Experiment 0)
    ▼
 raw transcripts                                                   asr_outputs/
    │
    │  published chapter text fetched with Playwright              varhadi_data/vahnt_text.py
    ▼
 CTC forced alignment: text ↔ audio timestamps                     varhadi_data/align.py
    │  (numpy Viterbi over the model's CTC log-probs)
    ▼
 1,296 candidate segments (≤ ~20 s each, 5.54 h, 40/89 chapters)   status: pending_review
    │  confidence triage (orders review, never verifies)           varhadi_data/triage.py
    ▼
 human listening review in a local web tool                        scripts/review_server.py
    │  correct / needs_correction (fixes the text) / rejected
    ▼
 VERIFIED gold segments  ──►  test set (held-out chapters)  ──►  WER/CER  (scripts/run_gold_eval.py)
                         └─►  train/val manifests            ──►  fine-tuning on Colab
```

### 4.1 Audio source and download
- **Source:** the Varhadi-Nagpuri New Testament audio Bible on bible.com: one narrator reading scripture.
  This is **general Varhadi speech, not farming speech**, so it is used for feasibility, baseline and
  adaptation experiments, not as farmer data.
- **Tech:** `playwright` (headless Chromium) opens the page and watches which audio file the page actually
  loads. `requests` then downloads it with the same cookies. Each MP3 is validated (size and header), and
  the download can resume from `progress.json`.

### 4.2 Forced alignment (how we get audio–text pairs without typing transcripts)
- bible.com also publishes the **text** of each chapter. But the text is per chapter, and training needs
  short clips with exact text. Forced alignment finds **where each verse is spoken** in the audio.
- **How:** run the baseline model only up to the CTC log-probabilities (per 40 ms frame, per token).
  Tokenize the published text. Then a **Viterbi dynamic-programming search** finds the most likely path
  through the frames that emits exactly those tokens in order. The audio before the first word (the
  "John chapter 1" announcement) and after the last word may stay unaligned.
- Verses are grouped into segments of at most about 20 s, cut between verses.
- **Tech:** pure `numpy` (no extra alignment library), in `varhadi_data/align.py`.
- **Important:** the aligned text is a **candidate**, not ground truth. The published text may differ from
  what the narrator actually said (for example, translator notes in brackets that are not read aloud).

### 4.3 Confidence triage (saves reviewer time, not a replacement for review)
- Score = 0.35·alignment + 0.30·agreement with baseline ASR + 0.15·audio quality + 0.10·duration
  + 0.10·speaking rate, minus penalties for text artifacts (brackets, digits, repeated words).
- Labels: HIGH_CONFIDENCE (≥ 0.90), NEEDS_REVIEW, REJECT_CANDIDATE. Result on 1,296: 718 / 573 / 5.
- All weights and thresholds live in `varhadi_data/triage_config.json` (version `triage-v1`).
- **Known bias:** the agreement signal uses the baseline model itself, so triage may only **order** the
  review. It must never choose the test set. Otherwise we would keep only the clips the baseline already
  gets right, and the comparison would be unfair.

### 4.4 Human review (gold data)
- `scripts/review_server.py` is a small local web page (Python stdlib `http.server`, only on 127.0.0.1).
  The reviewer plays each clip, sees the candidate text and the baseline output, and marks it
  **correct / needs_correction (types the right text) / rejected**, with keyboard shortcuts.
- A verifying decision is only accepted after the clip has played to the end, and a reviewer ID is required.
- The original candidate text is never overwritten. Corrections are stored in a separate column
  (`data/transcripts/gold_transcripts.csv`).

### 4.5 Evaluation (WER / CER)
- **WER** = (substitutions + deletions + insertions) / number of reference words. **CER** is the same at
  character level, which is more forgiving for Devanagari spelling variants.
- **Tech:** `jiwer` (`process_words`, `process_characters`). Both sides only have punctuation removed, so
  the scoring is identical for model A and model B.
- `scripts/run_gold_eval.py` refuses to compute anything when there are 0 verified segments.
  `experiments/evaluation/compare.py` produces `comparison.csv/json/md` for A vs B on the same segment IDs.
- Error analysis lists the most frequent word substitutions (for example `नाई → नाही`), which shows *what*
  the adaptation fixes.

### 4.6 Native farmer recordings (tooling BUILT, recordings not yet collected)
- Prompt sheets for agricultural questions (`data/prompts/recording_prompts.csv`). The prompts are drafted
  in standard Marathi and must be rewritten into Varhadi by a native speaker.
- Intake with a **consent gate**. Speaker IDs are pseudonymous: `VH_S001`, generated with HMAC-SHA256 from
  a secret salt, so names and phone numbers are never stored. Recordings get automatic quality flags
  (clipping, loudness), and `withdraw-speaker` deletes all of a speaker's data.
- **Train/val/test split by speaker**, with a leakage check, so the same voice never appears in both
  train and test.

---

## 5. Worked example: one question through the full system (target design)

| Step | Data at this point |
|---|---|
| Farmer says | "माझ्या कपाशीले किडे लागले, काय करू?" (audio, about 4 s) |
| Stage 3 | 16 kHz mono WAV, 64,000 samples |
| Stage 4 | Under 30 s, so 1 chunk |
| Stage 5 | Raw ASR: `माझ्या कपाशीले किडे लागले काय करू` |
| Stage 6 | Confidence OK, continue |
| Stage 7 | Normalised: `माझ्या कपाशीला किडे लागले काय करू` |
| Stage 8 | `{intent: pest_problem, crop: cotton, issue: pest}` |
| Stage 9 | Top-k chunks from verified cotton pest-management documents (for example ICAR/PDKV), filtered to crop = cotton |
| Stage 10 | Short Marathi answer based only on those chunks: what to check (which pest, crop stage) and safe first steps, with the source named |
| Stage 11 | Any chemical advice must come from a verified source and include safety notes; otherwise refer the farmer to the local agriculture officer |
| Stage 12 | Marathi speech + on-screen text |

Today, only stages 3–5 run, using the baseline model.

---

## 6. Tech stack summary

| Layer | Technology | Status |
|---|---|---|
| Language | Python 3.12 | BUILT |
| Audio download | Playwright (Chromium), requests | BUILT |
| Audio processing | librosa, soundfile, numpy | BUILT |
| ASR model | AI4Bharat IndicConformer (Marathi), Hugging Face Hub | BUILT |
| ASR framework | AI4Bharat NeMo fork 1.23.0rc0, PyTorch (CPU locally, GPU on Colab), PyTorch Lightning 2.2.1 | BUILT / PREPARED |
| Forced alignment | numpy CTC Viterbi (own code) | BUILT |
| Metrics | jiwer (WER, CER, S/D/I) | BUILT |
| Text diffing (stitching, lexicon) | Python `difflib` | BUILT |
| Review tool | Python `http.server` + a plain HTML/JS page | BUILT |
| Tests | pytest (offline, mocked audio; the model is never downloaded in tests) | BUILT |
| Fine-tuning | NeMo `speech_to_text_finetune.py`, Google Colab GPU | PREPARED |
| Backend API | FastAPI + Pydantic | PLANNED |
| Database | PostgreSQL (+ migrations) | PLANNED |
| Vector store | ChromaDB or Qdrant | PLANNED |
| RAG glue | LangChain, only where it adds value | PLANNED |
| LLM | to be chosen; backend-only, grounded prompts | PLANNED |
| TTS | existing Marathi/Indic TTS, replaceable | PLANNED |
| Vision | replaceable plant-disease model | PLANNED |
| Frontend | React + TailwindCSS (+ Material UI) | PLANNED |
| Telephony | IVR adapter via webhook/websocket | PLANNED |
| Environments | Windows venv (downloader, tests) + WSL2 Ubuntu venv (NeMo) | BUILT |

---

## 7. Cross-cutting rules

- **Research integrity:** no made-up WER/CER, transcripts or results. Anything not measured is written as
  PENDING. Automatic text is never treated as ground truth. Baseline and adapted results are kept separate.
- **Privacy:** pseudonymous speaker IDs, consent before recording, a deletion command, and no PII in logs.
  The repo is public, so audio, transcripts and evaluation data are gitignored.
- **Secrets:** `HF_TOKEN` and future keys live in `.env` only (`.env.example` documents them).
- **Security (planned):** authentication with roles (FARMER / EXPERT / ADMIN), rate limits, file
  size/type checks, CORS limits, parameterised SQL, audit logs.
- **Reproducibility:** every run is logged in `EXPERIMENT_LOG.md`, every config value records its source,
  and runs can resume (`predictions.csv`, per-chunk CSVs).

---

## 8. Where we are and what comes next

| Item | Status (2026-10-12) |
|---|---|
| Baseline transcription of 89 chapters | Done, 0 failures |
| Text alignment | 40 of 89 chapters, 1,296 candidate segments (5.54 h) |
| Triage | Done on all 1,296 |
| Human-verified segments | **0**, the main blocker |
| WER/CER of the baseline | **PENDING** (needs verified segments) |
| Fine-tuning | Prepared, not run (`P2_READY.md`) |
| Lexicon candidates | 345 rows, waiting for native-speaker review |
| Native farmer recordings | 0 (tooling ready) |

**Next steps, in order:**
1. A native speaker reviews the test chapter JHN_001 (31 clips, about 8 min) → first real baseline WER/CER.
2. Finish aligning the remaining 49 chapters, and review more segments.
3. Fine-tune on a Colab GPU → evaluate on the same test set → baseline vs adapted comparison.
4. Then normalisation → intent → RAG → LLM → safety → TTS → FastAPI + React prototype.

---

## 9. Questions a mentor is likely to ask

- **Why not train an ASR model from scratch?** We have hours of data, not thousands. Transfer learning from
  a strong Marathi model is the standard approach for a low-resource dialect.
- **Bible audio is not farming speech. Is that a problem?** Yes, and we say so. It measures dialect
  adaptation (Varhadi vs Marathi word forms). Domain coverage comes from the native agricultural
  recordings, for which the tooling is ready.
- **How do you know the transcripts are correct?** Not from alignment alone. Every test segment is checked
  by a human listener, and only "verified" rows are scored.
- **Isn't using the baseline to build the data biased?** The baseline only helps order the review. The test
  set is chosen by held-out chapter, never filtered on baseline accuracy or alignment score.
- **Why chunk the audio?** A full chapter needed more than 6 GB of RAM on CPU. Silence-based 20–30 s chunks
  use 1.74 GB, and the output was nearly the same on a 30 s check.
- **Why CTC decoding and not RNN-T?** CTC gives per-frame probabilities, which we also need for forced
  alignment. Using the same head keeps transcription and alignment consistent.
- **What if the LLM makes something up?** It may answer only from retrieved, verified documents. Low
  evidence leads to "I don't know" or escalation, and pesticide advice needs a verified source and a
  safety note.
