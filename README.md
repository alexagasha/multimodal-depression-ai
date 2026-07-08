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
- **Infra:** Google Colab Pro (training), Railway (deploy), Supabase (data/metadata store)

## Status

**This is a stub system.** The model backbones (`BertTextEncoder`,
`Wav2Vec2AudioEncoder`) currently use lightweight
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
│   │   └── metadata_pipeline.py
│   ├── fusion/
│   │   ├── aggregate.py                 # mean-pool segments -> participant vector
│   │   ├── model.py                     # fusion head (MLP)
│   │   └── run_pipeline.py              # orchestrates one participant end-to-end
│   ├── eval/
│   │   └── metrics.py                   # F1 / RMSE / MAE harness
│   └── xai/
│       └── attribution.py               # per-modality attribution prototype
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

## Open decisions to pin down before Phase 1 closes

- Is E-DAIC-WOZ the train+test source, or test-only with a separate primary
  training set? (See `data/synthetic/generate_synthetic_data.py` — it
  currently assumes a single source split into train/dev/test, matching
  E-DAIC-WOZ's official split convention.)
