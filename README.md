# Multimodal Depression Detection Platform

End-to-end pipeline for multimodal (text + audio + metadata) depression
detection and severity assessment, built against the DAIC-WOZ / E-DAIC-WOZ
dataset family.

This repo is being built **before E-DAIC-WOZ access is granted**, so the
project is ready to run on real data the moment access clears. Everything
here runs end-to-end today against synthetic data with the same file
structure and field names expected from the real dataset.

## Locked architecture decisions

- **Text backbone:** frozen BERT (sentence/segment-level mean-pooled embeddings)
- **Audio backbone:** frozen Wav2Vec2 (raw 16kHz waveform in, no hand-engineered features)
- **Metadata:** structured vector, one per participant
- **Segment length:** fixed 30 seconds across all modalities
- **Segment count N:** variable per participant — no padding/truncation
- **Aggregation:** mean-pool embeddings within each modality first, per participant
- **Fusion:** concatenate [text | audio | metadata] → single lightweight
  fusion head (small MLP) → one prediction per participant
- **No ensemble voting** — concatenation + single shared head, fewer moving parts
- **Output:** continuous severity score (PHQ-8-aligned), not just binary detection
- **Infra (target):** Google Colab Pro (training only) + **Firebase** (Cloud
  Run for the API, Firestore + Storage for data, Firebase Auth, Firebase
  Hosting for the web app) — supersedes the earlier Railway/Supabase line.
  Not deployed yet; see the roadmap below.

## Status

**The score model is code-complete — only real data + training remain.**
`src/pipelines/` → `src/fusion/` → `src/safety/` (sync, text/audio/metadata
encoders, fusion head, XAI, and the PHQ-9-item-9/HAM-D-suicide-domain
referral flag) all match the real data-collection instrument
(`docs/data-collection-tool-migration.md`). The model backbones
(`BertTextEncoder`, `Wav2Vec2AudioEncoder`) currently use lightweight
deterministic mock embeddings, not the real pretrained weights, when
torch/transformers aren't installed — this avoids multi-GB downloads / GPU
requirements while you wait for data access, and proves out 100% of the
surrounding plumbing ahead of time. Every encoder class has a clearly marked
real-model-with-mock-fallback path; the interface never changes, so nothing
downstream needs to be touched once real weights are trained.

**The rest of the system (backend API, web app, live audio, data platform,
deploy) is being built out beyond the score model itself.** A local FastAPI
backend (`api/`) and a minimal static demo page (`web/index.html`) already
exercise the full loop — participant intake → PHQ-9/HAM-D item responses
(with the referral flag firing immediately, independent of the model) →
audio upload → ASR (mock fallback, same convention) → scoring → the score
modal, with a GenUI narrative (template fallback without an API key). See
the phased roadmap for what's next (live in-browser recording, the real
Next.js app, DSM-5-grounded RAG, and the Firebase production deploy).

## Directory structure

```
depression-detection/
├── data/
│   ├── synthetic/
│   │   ├── generate_synthetic_data.py   # builds fake DAIC-WOZ-shaped sessions
│   │   └── sessions/                    # generated output (gitignored)
│   └── live/                            # API-staged session data + local JSON store (gitignored)
├── src/
│   ├── pipelines/
│   │   ├── sync.py                      # interviewer-turn stripping + 30s slicing
│   │   ├── text_pipeline.py
│   │   ├── audio_pipeline.py
│   │   ├── asr_pipeline.py              # Whisper transcription (mock fallback)
│   │   └── metadata_pipeline.py
│   ├── fusion/
│   │   ├── aggregate.py                 # mean-pool segments -> participant vector
│   │   ├── model.py                     # fusion head (MLP)
│   │   ├── run_pipeline.py              # orchestrates one participant end-to-end
│   │   └── train.py                     # trains the head on Colab, exports numpy weights
│   ├── eval/
│   │   ├── metrics.py                   # F1 / RMSE / MAE harness
│   │   └── robustness.py                # noise-robustness eval gate
│   ├── safety/
│   │   └── risk_flag.py                 # PHQ-9 item 9 / HAM-D suicide-domain referral flag
│   └── xai/
│       └── attribution.py               # per-modality attribution prototype
├── api/                                 # FastAPI backend (Phase 1 of the roadmap)
│   ├── main.py                          # HTTP endpoints, wraps the pipeline unchanged
│   ├── storage.py                       # local JSON store (Firestore-shaped; swaps in Phase 4)
│   └── genui.py                         # score-modal narrative (Claude, template fallback)
├── web/
│   └── index.html                       # minimal static score-modal demo (calls api/)
├── tests/
│   ├── test_end_to_end.py               # pipeline smoke tests on synthetic data
│   └── test_api.py                      # full local API flow
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running the end-to-end smoke test (pipeline only)

```bash
pip install -r requirements.txt
python data/synthetic/generate_synthetic_data.py --n_participants 5
python src/fusion/run_pipeline.py
python -m pytest tests/
```

## Running the local API + demo score modal

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
# then open web/index.html directly in a browser (CORS is open for local dev)
```

Or via Docker (`docker compose up` from `docker/`, brings up just the API by
default; `docker compose --profile batch up inference` runs the one-off batch
scoring job instead). Optional env vars: `ANTHROPIC_API_KEY` (real GenUI
narrative instead of the template fallback), `DEP_TEXT_MODEL` /
`DEP_AUDIO_MODEL` / `DEP_ASR_MODEL` (real backbone overrides).

This exercises the same vertical slice as the smoke test above, but through
HTTP: participant intake → PHQ-9/HAM-D scale responses (referral flag fires
immediately) → audio upload → scoring → the score modal. `web/index.html` is
a deliberately minimal stand-in for the real web app (Next.js, live in-browser
recording, auth, full WCAG coverage) — see the roadmap.

## When E-DAIC-WOZ access clears

1. Drop the real session folders into `data/real/` matching the same
   structure as `data/synthetic/sessions/` (see `generate_synthetic_data.py`
   for the exact expected schema).
2. Update `DATA_ROOT` in `src/pipelines/sync.py` to point at `data/real/`.
3. Swap each encoder's mock `encode()` body for real model loading (see
   `# SWAP:` comments in each `*_pipeline.py` file).
4. Re-run `src/fusion/run_pipeline.py` — no other code changes needed.

## Team split (suggested mapping onto this repo)

- **Person A:** `src/pipelines/text_pipeline.py`, `src/pipelines/audio_pipeline.py`,
  data access / E-DAIC-WOZ liaison
- **Person B:** `src/fusion/`, `src/eval/`, `src/xai/`,
  `docker/`, deployment

## System roadmap

The score model itself is done (only real data + training remain). What's
left — live audio capture, the real web app, Firebase production deploy,
DSM-5-grounded RAG for the GenUI narrative, and the rest of
`methodology.txt`'s objectives — is tracked in
[`docs/system-roadmap.md`](docs/system-roadmap.md).

## Open decisions to pin down before Phase 1 (of the ML pipeline) closes

- Is E-DAIC-WOZ the train+test source, or test-only with a separate primary
  training set? (See `data/synthetic/generate_synthetic_data.py` — it
  currently assumes a single source split into train/dev/test, matching
  E-DAIC-WOZ's official split convention.)
