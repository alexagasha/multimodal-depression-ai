"""
Service layer: wraps the existing src/ pipeline for the API.

Deliberately does NOT modify anything under src/ — the CLI entry point
(src/fusion/run_pipeline.py), Docker image, and pytest suite all keep working
exactly as before. This module composes the same building blocks
(sync -> per-modality encode -> aggregate -> fuse) so the API can additionally:

  - cache fusion vectors per participant (so XAI attribution is instant
    instead of re-encoding on every request)
  - accept a configurable model seed per run (the CLI script always uses
    the FusionHead default of seed=0)
  - keep in-memory state about the last run so /health and the XAI endpoint
    stay consistent with whatever is currently on screen
"""
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Path setup — PROJECT_ROOT is the depression-detection/ folder (two levels
# up from backend/app/). Inserting it on sys.path lets us import `src.*`
# exactly like the existing scripts do, without copying or moving any code.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.sync import build_segments  # noqa: E402
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline  # noqa: E402
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline  # noqa: E402
from src.pipelines.metadata_pipeline import run_metadata_pipeline, load_metadata  # noqa: E402
from src.fusion.aggregate import build_participant_vector  # noqa: E402
from src.fusion.model import FusionHead, DEPRESSION_THRESHOLD  # noqa: E402
from src.xai.attribution import explain_participant  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "synthetic", "sessions")
LABELS_PATH = os.path.join(DATA_ROOT, "LABELS.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
PREDICTIONS_CSV = os.path.join(OUTPUTS_DIR, "predictions.csv")

MAX_PARTICIPANTS = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PipelineState:
    """
    In-memory singleton holding the currently "active" model + cached fusion
    vectors. A single process-wide instance is fine here: this is a research
    dashboard for one operator, not a multi-tenant service.
    """

    def __init__(self):
        self.lock = Lock()
        self.text_enc: Optional[BertTextEncoder] = None
        self.audio_enc: Optional[Wav2Vec2AudioEncoder] = None
        self.fusion_head: Optional[FusionHead] = None
        self.model_seed: Optional[int] = None
        self.last_run_at: Optional[str] = None
        self.last_n_participants: Optional[int] = None
        self.fusion_vec_cache = {}  # participant_id (int) -> np.ndarray

    def ensure_model(self, seed: int = 0):
        """(Re)builds the encoder/model trio only if the seed actually changed."""
        if self.fusion_head is None or self.model_seed != seed:
            self.text_enc = BertTextEncoder()
            self.audio_enc = Wav2Vec2AudioEncoder()
            self.fusion_head = FusionHead(seed=seed)
            self.model_seed = seed
            self.fusion_vec_cache = {}


STATE = PipelineState()


def dataset_info() -> dict:
    if not os.path.exists(LABELS_PATH):
        return {"exists": False, "n_participants": 0, "splits": {}, "participant_ids": []}
    df = pd.read_csv(LABELS_PATH)
    return {
        "exists": True,
        "n_participants": int(len(df)),
        "splits": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "participant_ids": sorted(int(x) for x in df["participant_id"].tolist()),
    }


def health() -> dict:
    return {
        "status": "ok",
        "mock_encoders": True,
        "model_seed": STATE.model_seed,
        "last_run_at": STATE.last_run_at,
        "last_n_participants": STATE.last_n_participants,
        "predictions_available": os.path.exists(PREDICTIONS_CSV),
        "dataset": dataset_info(),
    }


def generate_dataset(n_participants: int) -> dict:
    import shutil

    from data.synthetic.generate_synthetic_data import generate  # local import: has its own argparse at module scope guarded by __main__

    n_participants = max(1, min(int(n_participants), MAX_PARTICIPANTS))

    with STATE.lock:
        if os.path.exists(DATA_ROOT):
            shutil.rmtree(DATA_ROOT)
        generate(n_participants, DATA_ROOT)
        STATE.fusion_vec_cache = {}
        if os.path.exists(PREDICTIONS_CSV):
            os.remove(PREDICTIONS_CSV)

    return dataset_info()


def _fusion_vector_for(pid: int) -> np.ndarray:
    """Computes (or returns cached) fusion vector for one participant."""
    if pid in STATE.fusion_vec_cache:
        return STATE.fusion_vec_cache[pid]

    segments = build_segments(pid, data_root=DATA_ROOT)
    if not segments:
        raise ValueError(f"No segments found for participant {pid}")

    text_embs = run_text_pipeline(segments, STATE.text_enc)
    audio_embs = run_audio_pipeline(pid, segments, STATE.audio_enc, data_root=DATA_ROOT)
    metadata_vec = run_metadata_pipeline(pid, data_root=DATA_ROOT)

    vec = build_participant_vector(text_embs, audio_embs, metadata_vec)
    STATE.fusion_vec_cache[pid] = vec
    return vec


def run_pipeline(seed: int = 0) -> dict:
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "No synthetic dataset found. Generate one first (POST /api/dataset/generate)."
        )

    with STATE.lock:
        STATE.ensure_model(seed=seed)
        labels_df = pd.read_csv(LABELS_PATH)

        results = []
        for raw_pid in labels_df["participant_id"]:
            pid = int(raw_pid)
            segments = build_segments(pid, data_root=DATA_ROOT)
            if not segments:
                continue

            fusion_vec = _fusion_vector_for(pid)
            phq8_pred = STATE.fusion_head.forward(fusion_vec)
            binary_pred = int(phq8_pred >= DEPRESSION_THRESHOLD)

            row = labels_df[labels_df["participant_id"] == raw_pid].iloc[0]
            phq8_true = int(row["phq8_score"])
            results.append({
                "participant_id": pid,
                "n_segments": len(segments),
                "phq8_pred": round(float(phq8_pred), 3),
                "binary_pred": binary_pred,
                "phq8_true": phq8_true,
                "binary_true": int(phq8_true >= DEPRESSION_THRESHOLD),
                "split": str(row["split"]),
            })

        df = pd.DataFrame(results)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        df.to_csv(PREDICTIONS_CSV, index=False)

        STATE.last_run_at = _now_iso()
        STATE.last_n_participants = len(results)

    metrics = compute_metrics(df, verbose=False) if not df.empty else None
    return {
        "results": results,
        "metrics": metrics,
        "model_seed": seed,
        "ran_at": STATE.last_run_at,
    }


def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV):
        return None
    df = pd.read_csv(PREDICTIONS_CSV)
    return df.to_dict(orient="records")


def get_metrics() -> Optional[dict]:
    if not os.path.exists(PREDICTIONS_CSV):
        return None
    df = pd.read_csv(PREDICTIONS_CSV)
    if df.empty:
        return None

    metrics = compute_metrics(df, verbose=False)
    cm = confusion_matrix(df["binary_true"], df["binary_pred"], labels=[0, 1]).tolist()
    metrics["confusion_matrix"] = {"labels": ["non_depressed", "depressed"], "matrix": cm}
    return metrics


def get_attribution(pid: int) -> dict:
    pid = int(pid)
    with STATE.lock:
        STATE.ensure_model(seed=STATE.model_seed if STATE.model_seed is not None else 0)
        fusion_vec = _fusion_vector_for(pid)
        return explain_participant(fusion_vec, STATE.fusion_head)


def get_all_attributions() -> list:
    """
    Batch XAI attribution for every participant currently in predictions.csv.

    Exists so the table view can show a per-row fusion-strip glyph with a
    single request instead of the client firing one call per participant
    (which would turn an N-row table into an N+1 network problem as the
    dataset grows).
    """
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    df = pd.read_csv(PREDICTIONS_CSV)

    with STATE.lock:
        STATE.ensure_model(seed=STATE.model_seed if STATE.model_seed is not None else 0)
        out = []
        for raw_pid in df["participant_id"]:
            pid = int(raw_pid)
            try:
                vec = _fusion_vector_for(pid)
                exp = explain_participant(vec, STATE.fusion_head)
                exp["participant_id"] = pid
                out.append(exp)
            except Exception:
                continue
        return out


def get_participant_detail(pid: int) -> dict:
    pid = int(pid)
    meta = load_metadata(pid, data_root=DATA_ROOT)
    segments = build_segments(pid, data_root=DATA_ROOT)

    pred_row = None
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
        match = df[df["participant_id"] == pid]
        if not match.empty:
            pred_row = match.iloc[0].to_dict()

    try:
        attribution = get_attribution(pid)
    except Exception:
        attribution = None

    return {
        "participant_id": pid,
        "metadata": meta,
        "n_segments": len(segments),
        "segments_preview": [
            {"segment_idx": s["segment_idx"], "start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ],
        "prediction": pred_row,
        "attribution": attribution,
    }


def run_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    passed = failed = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))

    return {
        "passed": passed,
        "failed": failed,
        "success": proc.returncode == 0,
        "output": output.strip(),
    }
