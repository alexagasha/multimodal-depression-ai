# Multimodal Depression Detection Platform

End-to-end pipeline for multimodal (text + audio + video + metadata) depression
detection and severity assessment, built against the DAIC-WOZ / E-DAIC-WOZ
dataset family.

This repo is being built **before E-DAIC-WOZ access is granted**, so the
project is ready to run on real data the moment access clears. Everything
here runs end-to-end today against synthetic data with the same file
structure and field names expected from the real dataset.

## Locked architecture decisions

- **Text backbone:** frozen BERT (sentence/segment-level mean-pooled embeddings)
- **Audio backbone:** frozen Wav2Vec2 (raw 16kHz waveform in, no hand-engineered features)
- **Video backbone:** frozen MobileNetV3 (per-frame embeddings, 1 fps sampling)
- **Metadata:** structured vector, one per participant
- **Segment length:** fixed 30 seconds across all modalities
- **Segment count N:** variable per participant — no padding/truncation
- **Aggregation:** mean-pool embeddings within each modality first, per participant
- **Fusion:** concatenate [text | audio | video | metadata] → single lightweight
  fusion head (small MLP) → one prediction per participant
- **No ensemble voting** — concatenation + single shared head, fewer moving parts
- **Output:** continuous severity score (PHQ-8-aligned), not just binary detection
- **Infra:** Google Colab Pro (training), Railway (deploy), Supabase (data/metadata store)

## Status

**This is a stub system.** The model backbones (`BertTextEncoder`,
`Wav2Vec2AudioEncoder`, `MobileNetV3VideoEncoder`) currently use lightweight
deterministic mock embeddings, not the real pretrained weights — this avoids
multi-GB downloads / GPU requirements while you wait for data access, and
proves out 100% of the surrounding plumbing (segmenting, sync, pooling,
fusion, training loop, eval, XAI, Docker) ahead of time.

Every encoder class has a clearly marked `# SWAP: load real model here`
comment showing exactly what to replace once you have GPU access and want
real embeddings. The interface (`encode(self, segment) -> np.ndarray`)
will not change — nothing downstream needs to be touched.

## Directory structure

```
depression-detection/
├── data/
│   └── synthetic/
│       ├── generate_synthetic_data.py   # builds fake DAIC-WOZ-shaped sessions
│       └── sessions/                    # generated output (gitignored)
├── src/
│   ├── pipelines/
│   │   ├── sync.py                      # interviewer-turn stripping + 30s slicing
│   │   ├── text_pipeline.py
│   │   ├── audio_pipeline.py
│   │   ├── video_pipeline.py
│   │   └── metadata_pipeline.py
│   ├── fusion/
│   │   ├── aggregate.py                 # mean-pool segments -> participant vector
│   │   ├── model.py                     # fusion head (MLP)
│   │   └── run_pipeline.py              # orchestrates one participant end-to-end
│   ├── eval/
│   │   └── metrics.py                   # F1 / RMSE / MAE harness
│   └── xai/
│       └── attribution.py               # per-modality attribution prototype
├── backend/                             # FastAPI layer over src/ — see below
│   ├── app/
│   │   ├── main.py                      # routes + CORS
│   │   ├── services.py                  # wraps src/ pipeline, no changes to src/
│   │   └── schemas.py                   # request/response models
│   └── requirements.txt
├── frontend/                            # operator dashboard (Vite + React + TS)
│   └── src/
├── tests/
│   └── test_end_to_end.py               # smoke test on synthetic data
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running the end-to-end smoke test

```bash
pip install -r requirements.txt
python data/synthetic/generate_synthetic_data.py --n_participants 5
python src/fusion/run_pipeline.py
python -m pytest tests/
```

## Dashboard & API

A FastAPI backend (`backend/`) and a React + TypeScript dashboard (`frontend/`)
sit on top of the pipeline above. `backend/` only *imports* `src/` — nothing
in `src/` was changed, so the CLI commands above, the Docker image, and the
pytest suite all still work exactly as before.

**Run locally** (two terminals):

```bash
# terminal 1 — API on :8000
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# terminal 2 — dashboard on :5173, proxies /api to :8000 (see vite.config.ts)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. From there: generate a synthetic dataset, run
the pipeline, click into a participant for the per-modality XAI breakdown,
or run `tests/` directly from the sidebar. Interactive API docs (Swagger)
are at http://localhost:8000/docs.

**Run with Docker** (adds `api` + `dashboard` services to the existing
`inference` one):

```bash
cd docker
docker compose up --build api dashboard
```

Dashboard: http://localhost:8080 · API: http://localhost:8000

**Deploying separately** (e.g. Vercel for the dashboard, Render for the API —
the same split as LASA-VOTING-SYSTEM): set `VITE_API_BASE_URL` at frontend
build time to the deployed API URL, and add the frontend's origin to the
API's `CORS_EXTRA_ORIGINS` env var. See `frontend/.env.example`.

### API surface

| Method | Path                              | Purpose                                    |
|--------|------------------------------------|---------------------------------------------|
| GET    | `/api/health`                     | dataset status, model seed, last run        |
| POST   | `/api/dataset/generate`           | regenerate synthetic sessions               |
| POST   | `/api/pipeline/run`               | run the fusion pipeline, write predictions  |
| GET    | `/api/predictions`                | latest predictions.csv as JSON              |
| GET    | `/api/metrics`                    | F1 / RMSE / MAE / confusion matrix          |
| GET    | `/api/participants/{pid}`         | metadata, segments, prediction, attribution |
| GET    | `/api/participants/{pid}/attribution` | XAI attribution for one participant     |
| GET    | `/api/attributions`               | XAI attribution for every participant (batch, avoids N+1 calls from the table view) |
| POST   | `/api/tests/run`                  | runs `pytest tests/`, returns pass/fail     |


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
- **Person B:** `src/pipelines/video_pipeline.py`, `src/fusion/`, `src/eval/`,
  `docker/`, deployment

