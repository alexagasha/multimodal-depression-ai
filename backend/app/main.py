"""
FastAPI server exposing the multimodal depression-severity pipeline over HTTP.

Run (from backend/):
    uvicorn app.main:app --reload --port 8000

This never touches src/ — it only imports and orchestrates it (see
services.py). The CLI entry point (python -m src.fusion.run_pipeline),
Docker image, and pytest suite all keep working exactly as before.
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import services
from .schemas import (
    DatasetInfo,
    GenerateDatasetRequest,
    HealthResponse,
    ParticipantDetail,
    PipelineRunResponse,
    RunPipelineRequest,
    TestRunResponse,
)

app = FastAPI(
    title="Depression Detection API",
    description=(
        "REST layer over the multimodal (text/audio/metadata -> PHQ-8 severity) "
        "fusion pipeline. Currently backed by deterministic MOCK encoders — see "
        "README.md — so predictions are plumbing proof, not clinical output."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS: allow the local Vite dev server by default. When you deploy the
# frontend (e.g. Vercel, same pattern as LASA-VOTING-SYSTEM), add that origin
# via the CORS_EXTRA_ORIGINS env var (comma-separated) rather than editing
# this file.
# ---------------------------------------------------------------------------
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
extra = os.environ.get("CORS_EXTRA_ORIGINS", "")
allowed_origins = DEFAULT_ORIGINS + [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    return services.health()


@app.post("/api/dataset/generate", response_model=DatasetInfo)
def generate_dataset(body: GenerateDatasetRequest):
    return services.generate_dataset(body.n_participants)


@app.get("/api/dataset", response_model=DatasetInfo)
def get_dataset():
    return services.dataset_info()


@app.post("/api/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(body: RunPipelineRequest):
    try:
        return services.run_pipeline(seed=body.seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/predictions")
def get_predictions():
    preds = services.load_predictions()
    if preds is None:
        raise HTTPException(
            status_code=404,
            detail="No predictions yet. Run the pipeline first (POST /api/pipeline/run).",
        )
    return preds


@app.get("/api/metrics")
def get_metrics():
    metrics = services.get_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="No predictions yet. Run the pipeline first (POST /api/pipeline/run).",
        )
    return metrics


@app.get("/api/participants/{pid}", response_model=ParticipantDetail)
def get_participant(pid: int):
    try:
        return services.get_participant_detail(pid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Participant {pid} not found on disk.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/attributions")
def get_all_attributions():
    return services.get_all_attributions()


@app.get("/api/participants/{pid}/attribution")
def get_participant_attribution(pid: int):
    try:
        return services.get_attribution(pid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Participant {pid} not found on disk.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/tests/run", response_model=TestRunResponse)
def run_tests():
    return services.run_tests()
