# VAHNT Audio Downloader

Downloads the 260 New Testament chapter MP3 files for the Varhadi-Nagpuri New Testament (`VAHNT`, Bible version `3451`). This milestone collects audio only; transcript extraction, TTS, alignment, speech-to-text, and dataset processing are intentionally not included.

## Installation

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Commands

```powershell
python downloader.py
python downloader.py --book MAT
python downloader.py --book MAT --headless
python downloader.py --chapter MAT.1.VAHNT
python downloader.py --retry-failed
python downloader.py --status
```

Chromium is visible by default. Add `--headless` for unattended runs. Downloads are sequential and wait two seconds between chapters by default; change that with `--delay SECONDS`.

## Output and resume behavior

Audio is saved as `audio/{BOOK}_{CHAPTER:03d}.mp3`. `progress.json` records successful chapter IDs, `failed.json` records chapters that exhausted three attempts, and `chapter_metadata.json` records the source page and local file for every success. A file is skipped only after it passes the minimum-size and MP3-header validation, even if it is already listed in progress.

Logs are written to `logs/downloader.log`. Downloads use Playwright to observe the page's actual audio response, then reuse the browser user agent and cookies for a direct `requests` download. No CDN URL is assumed or constructed.

## Tests

```powershell
python -m pytest
```

The tests are offline and do not download Bible.com content.

## Current limitations

- Only direct MP3 resources are supported. A streaming manifest is reported as a failure rather than being converted.
- Validation is intentionally basic: file size and MP3 header checks are used; no transcript or audio dataset processing is performed.
- The downloader intentionally uses one browser session per chapter and no parallelism.

## Dataset provenance

Audio in `audio/` is the VAHNT (Varhadi-Nagpuri) New Testament audio Bible, scraped from
`bible.com` (version 3451). It is downloaded for research use in a final-year academic
project (KisaanDost). Downloading this audio does not by itself grant a license to
redistribute it or to train and publish a model on it — check bible.com's terms before
using it beyond local research experiments. This is scripture audio, not agricultural or
farmer speech; treat it only as general Varhadi speech exposure, not as a proxy for the
agricultural domain the wider project targets.

## ASR baseline (Experiment 0)

`asr_baseline/` is a separate, isolated pipeline that transcribes the downloaded audio with
a pretrained Marathi ASR model (`ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, via
the AI4Bharat NeMo fork) and, where a ground-truth transcript is available, scores WER/CER
with `jiwer`. It does not modify `downloader.py`, `audio/`, `progress.json`, or any other
downloader state, and has its own resume mechanism based on `asr_outputs/predictions.csv`.

**No Varhadi correction of any kind is applied** — predictions are the raw model output, so
that a later Varhadi-adapted model can be compared against a clean baseline.

### Environment

NeMo's `pynini` dependency has no Windows wheel, so install and run the ASR pipeline in
WSL2/Linux (see `requirements-asr.txt` for exact steps), not the Windows venv used for the
downloader:

```bash
wsl
cd /mnt/e/KisaanDost
python3 -m venv .venv-asr-wsl && source .venv-asr-wsl/bin/activate
pip install -r requirements-asr.txt
# then install the AI4Bharat NeMo fork per requirements-asr.txt's comments
```

### Commands

```bash
python scripts/validate_asr_dataset.py                 # inspect audio + reference availability
python scripts/run_asr_baseline.py --audio audio/example.mp3   # single file
python scripts/run_asr_baseline.py                      # batch, resumable via predictions.csv
python scripts/run_asr_baseline.py --force               # re-run everything (also clears chunk resume state)
python scripts/run_asr_baseline.py --chunk-sec 20 --max-chunk-sec 30 --overlap-sec 0  # chunking knobs
python -m asr_baseline.evaluate --predictions asr_outputs/predictions.csv  # WER/CER (only for rows with a reference)
python -m asr_baseline.error_analysis
python scripts/generate_asr_report.py
```

### Chunked inference (long audio)

A full ~8-minute chapter OOM-kills the encoder on CPU in one pass. Audio longer than
`--max-chunk-sec` (default 30 s) is split at the quietest point between `--chunk-sec` and
`--max-chunk-sec`, each chunk is transcribed with the unchanged model, and the raw outputs are
stitched in order. Per-chunk metadata and raw outputs go to `asr_outputs/chunks/<stem>.csv`,
which is also the per-chunk resume state: a crash or a failed chunk is retried on the next run
without redoing finished chunks. Settings can also come from `ASR_CHUNK_SEC`,
`ASR_CHUNK_MAX_SEC`, `ASR_CHUNK_OVERLAP_SEC`. Measured on `JHN_001` (473.6 s): 19 chunks,
53.9 s inference, peak RSS 1.74 GB. See `docs/asr_baseline.md` for the stitching strategy.

### Dataset tooling (P1)

`scripts/dataset_tool.py` manages `data/`: the manifest, native-recording intake with consent
and pseudonymous speaker IDs, speaker-level splits, and the human gold-transcript workflow.
See `docs/dataset.md`.

### Ground truth

`asr_metadata.csv` (`audio_path,reference_text,speaker_id`) is currently a header-only
template: no sentence-level Varhadi transcripts exist for this audio yet, so WER/CER cannot
be computed against it. Add rows with a manually transcribed `reference_text` to evaluate a
sample; audio without a row is still transcribed (transcription-only mode), just never scored.

### Tests

ASR unit tests (`tests/test_asr_*.py`) run with `requirements-asr.txt` installed, use mocked
audio/CSV fixtures, and never download the model. Run them the same way as the downloader's
tests: `python -m pytest`.
## Chat bot (bot/)

Mobile-first web chat. Pipeline: a Groq LLM answers in simple standard Marathi; then, if the fine-tuned
ByT5 model is installed in `bot/models/mr2vh/` (gitignored, see `docs`/EXPERIMENT_LOG P3a), each sentence is
converted to Varhadi by it, with a length guard that falls back to the rules; finally `varhadify()` swaps
whole words for Varhadi forms that occur in the local VAHNT corpus (`bot/varhadi_style.json`).
Groq models invent Konkani/Malvani endings when asked for Varhadi directly, so they are not asked to.
The result has not been checked by a native speaker. `KD_MR2VH=0` turns the model off; `KD_MR2VH_FILTER=off` keeps all of its word changes (more dialect, but unseen changes are often wrong); `KD_BT_KEY_VAR` picks the `.env` key the data job uses.

```powershell
python -m pip install -r requirements-bot.txt      # Windows venv
# GROQ_API_KEY in .env, then:
python -m uvicorn bot.server:app --reload           # open http://localhost:8000
python -m pytest tests/test_bot.py
```

Voice in/out uses the browser's Marathi speech engines (Chrome/Android; needs `localhost` or HTTPS,
so a phone on plain-http LAN will not get the mic). It does not use the project's IndicConformer ASR yet.
The microphone works like a recorder: tap to start, it keeps listening through pauses (the browser ends a
recognition session at the first pause, so the page restarts it and joins the text), your words appear live, and
only a tap on the red stop button sends the question; the X (or Esc) cancels. It stops by itself after 5 minutes.

### Crop photo diagnosis

The camera button sends a leaf photo to `POST /api/diagnose` with the crop (cotton, soybean, orange, tur). Soybean and orange use
CC BY 4.0 ConvNeXt classifiers by IXR (`ixrbhii/soybean-disease-convnext`, `ixrbhii/citrus-disease-convnext`, downloaded from
Hugging Face on first use; credit "Multi-crop disease models, IXR (agritechixr)" and the LeafNet dataset). Cotton and tur have no model
yet and say so. The result is always a *possible* disease with the advice in `bot/disease_guidance.json` (no product names or doses,
not agronomist-checked). Measured accuracy on independent public photos, and why the confidence bar is 0.60, is in `EXPERIMENT_LOG.md` (P4).
Run `python -m scripts.build_disease_guidance` after editing the guidance to refresh its Varhadi text.

### Farmer profile, local weather and region

On first launch the page asks for district, taluka, village (optional name and crops). There is **no password or account**: the details are
kept in the browser (`localStorage`) and sent with each question as `profile`. The server (`bot/places.py`) checks district/taluka against
`bot/places.json` (11 Vidarbha districts, 120 talukas, about 13,000 places), looks up the forecast on Open-Meteo (`bot/weather.py`, free,
no key, cached 30 min; village coordinates when OpenStreetMap has them, else the taluka centre) and adds location, crops, weather and the
PDKV Akola agro-climatic zone (Western/Central/Eastern Vidarbha) to the model's prompt. Soil is stated only in general terms and the bot
sends the farmer to the Soil Health Card. `GET /api/weather` feeds the weather card; `GET /api/places` and `/api/places/villages` feed the form.

Rebuild the place list with `python -m scripts.build_vidarbha_places` (about 11 Overpass requests, with back-off; it retries because the
public server is often busy). Villages come from OpenStreetMap contributors (ODbL, credit shown in the page) and are **incomplete** and
mostly spelled in Roman letters; a village missing from the list can be typed. Weather: Open-Meteo.com.

### Voice output

Replies are read by a server voice, `facebook/mms-tts-mar` (CC BY-NC 4.0; downloaded from Hugging Face on first use), through
`POST /api/tts` (`bot/tts.py`), sentence by sentence with pauses; the page falls back to the browser's Marathi voice. It is a
**Marathi** voice by default. A Varhadi-tuned voice (MMS fine-tuned on 1.4 h of one narrator, `EXPERIMENT_LOG.md` P7) is used automatically if
`bot/models/tts/` exists. It is local only and **never to be published** (narrator's recordings have an unverified licence). By the Whisper
round-trip check it is not more recognisable than the generic voice (CER 0.36 vs 0.26, n = 20); only a native listening test can say whether
it sounds better. `KD_TTS=0` turns the server voice off.
