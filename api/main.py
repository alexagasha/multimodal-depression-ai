"""
FastAPI backend — Phase 1 of the system roadmap (see the plan / migration doc).

Wraps the existing scoring pipeline behind HTTP endpoints without touching any
of its code: uploads get staged into the same on-disk session-folder shape
(`{id}_TRANSCRIPT.csv`, `{id}_AUDIO.*`, `{id}_METADATA.json`) that
sync.py/text_pipeline.py/audio_pipeline.py/metadata_pipeline.py already read,
under data/live/sessions/ instead of data/synthetic/sessions/.

Local dev:  uvicorn api.main:app --reload
Docker:     see docker/docker-compose.yml (api service)

api/storage.py is a local JSON stand-in for Firestore (Phase 4 swaps the
implementation only, not the call sites). api/genui.py generates the score
modal's narrative (Claude if ANTHROPIC_API_KEY is set, template otherwise).
"""
import io
import os
import sys
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipelines.sync import build_segments
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline, _load_wav
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.pipelines.asr_pipeline import WhisperASR, run_asr_pipeline
from src.fusion.aggregate import build_participant_vector
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD
from src.xai.attribution import explain_participant
from src.safety.risk_flag import flag_risk
from api.storage import store
from api.genui import generate_narrative

LIVE_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "live", "sessions")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "weights", "fusion_head.npz")

app = FastAPI(title="Depression Detection API", version="0.1.0")

# Local dev only: the static score-modal demo page (web/index.html) is opened
# directly from disk or a plain static server, so it needs CORS to call this
# API. Phase 3's real Next.js app replaces this with a same-origin/proxy
# setup; tighten this before any real deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frozen encoders + fusion head, loaded once. Mock fallback if torch/transformers
# aren't installed (see each module's docstring) — safe for local dev.
_text_enc = BertTextEncoder()
_audio_enc = Wav2Vec2AudioEncoder()
_asr = WhisperASR()
_fusion_head = (
    FusionHead.load(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else FusionHead()
)


class ParticipantIn(BaseModel):
    age_band: str
    sex: str
    marital_status: str
    ethnicity: str
    residence: str
    education_level: str
    employment_status: str
    smartphone: str
    site: str


class SessionIn(BaseModel):
    participant_id: str


class ScaleResponsesIn(BaseModel):
    phq9_total: int
    hamd_total: int
    phq9_item9: int
    hamd_suicide_item: int


def _session_dir(session_id: str) -> str:
    return os.path.join(LIVE_DATA_ROOT, session_id)


def _require_session(session_id: str) -> dict:
    session = store.get("sessions", session_id)
    if session is None:
        raise HTTPException(404, f"session {session_id} not found")
    return session


@app.post("/participants")
def create_participant(body: ParticipantIn):
    participant_id = uuid.uuid4().hex[:10]
    record = {"participant_id": participant_id, **body.model_dump()}
    store.set("participants", participant_id, record)
    return {"participant_id": participant_id}


@app.post("/sessions")
def create_session(body: SessionIn):
    participant = store.get("participants", body.participant_id)
    if participant is None:
        raise HTTPException(404, f"participant {body.participant_id} not found")

    session_id = uuid.uuid4().hex[:10]
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    metadata = {k: v for k, v in participant.items() if k != "participant_id"}
    metadata["participant_id"] = session_id  # pid used by metadata_pipeline is the session id
    import json
    with open(os.path.join(session_dir, f"{session_id}_METADATA.json"), "w") as f:
        json.dump(metadata, f)

    record = {
        "session_id": session_id,
        "participant_id": body.participant_id,
        "status": "created",
        "risk_flag": None,
    }
    store.set("sessions", session_id, record)
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/scale-responses")
def submit_scale_responses(session_id: str, body: ScaleResponsesIn):
    _require_session(session_id)
    risk_flag = flag_risk(body.phq9_item9, body.hamd_suicide_item)
    store.set("scale_responses", session_id, {"session_id": session_id, **body.model_dump()})
    store.update("sessions", session_id, {"status": "scale_responses_recorded", "risk_flag": risk_flag})
    return {"risk_flag": risk_flag}


@app.post("/sessions/{session_id}/audio")
async def upload_audio(session_id: str, file: UploadFile):
    _require_session(session_id)
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    raw = await file.read()
    wav_path = os.path.join(session_dir, f"{session_id}_AUDIO.wav")
    with open(wav_path, "wb") as f:
        f.write(raw)

    waveform, sr = _load_wav(wav_path)
    duration_sec = len(waveform) / float(sr)
    import json
    with open(os.path.join(session_dir, f"{session_id}_AUDIO.meta.json"), "w") as f:
        json.dump({"sample_rate": sr, "duration_sec": duration_sec}, f)

    transcript_rows = run_asr_pipeline(waveform, sr, _asr)
    import pandas as pd
    pd.DataFrame(transcript_rows).to_csv(
        os.path.join(session_dir, f"{session_id}_TRANSCRIPT.csv"), index=False
    )

    store.update("sessions", session_id, {"status": "audio_uploaded"})
    return {"duration_sec": duration_sec, "n_transcript_rows": len(transcript_rows)}


@app.post("/sessions/{session_id}/score")
def score_session(session_id: str):
    session = _require_session(session_id)
    scale = store.get("scale_responses", session_id)
    if scale is None:
        raise HTTPException(400, "scale responses not submitted yet")

    segments = build_segments(session_id, data_root=LIVE_DATA_ROOT)
    if not segments:
        raise HTTPException(400, "no transcript segments — upload audio first")

    text_embs = run_text_pipeline(segments, _text_enc)
    audio_embs = run_audio_pipeline(session_id, segments, _audio_enc, data_root=LIVE_DATA_ROOT)
    metadata_vec = run_metadata_pipeline(session_id, data_root=LIVE_DATA_ROOT)
    fusion_vec = build_participant_vector(text_embs, audio_embs, metadata_vec)

    scores = _fusion_head.forward(fusion_vec)
    binary_pred = int(scores["hamd"] >= HAMD_CASENESS_THRESHOLD)
    xai = explain_participant(fusion_vec, _fusion_head)
    risk_flag = flag_risk(scale["phq9_item9"], scale["hamd_suicide_item"])

    narrative = generate_narrative(
        scores["phq9"], scores["hamd"], binary_pred,
        xai["modality_attributions"], risk_flag, HAMD_CASENESS_THRESHOLD,
    )

    result = {
        "session_id": session_id,
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        "binary_pred": binary_pred,
        "risk_flag": risk_flag,
        "modality_attributions": xai["modality_attributions"],
        "narrative": narrative,
    }
    store.set("predictions", session_id, result)
    store.update("sessions", session_id, {"status": "scored"})
    return result


@app.get("/sessions/{session_id}/results")
def get_results(session_id: str):
    _require_session(session_id)
    result = store.get("predictions", session_id)
    if result is None:
        raise HTTPException(404, "not scored yet — POST /sessions/{id}/score first")
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
