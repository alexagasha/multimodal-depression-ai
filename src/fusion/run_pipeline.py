"""
Orchestrates the full pipeline for each participant:
  sync -> text -> audio -> metadata -> aggregate -> fusion head -> prediction

This is the script that proves end-to-end plumbing works, and will become
the inference entry point once real data and trained weights are available.

Predicts both severity scores (PHQ-9 and HAM-D); binary caseness (the primary
classification label) is derived from HAM-D >= threshold (clinician gold standard).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.sync import build_segments, DATA_ROOT
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.fusion.aggregate import build_participant_vector
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD


def run_participant(pid, text_enc, audio_enc, fusion_head, data_root=DATA_ROOT):
    """Full pipeline for a single participant. Returns result dict."""
    segments = build_segments(pid, data_root=data_root)
    if not segments:
        print(f"[WARN] pid={pid}: no participant segments found, skipping.")
        return None

    text_embs = run_text_pipeline(segments, text_enc)
    audio_embs = run_audio_pipeline(pid, segments, audio_enc, data_root=data_root)
    metadata_vec = run_metadata_pipeline(pid, data_root=data_root)

    fusion_vec = build_participant_vector(text_embs, audio_embs, metadata_vec)
    scores = fusion_head.forward(fusion_vec)
    binary_pred = int(scores["hamd"] >= HAMD_CASENESS_THRESHOLD)

    return {
        "participant_id": pid,
        "n_segments": len(segments),
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        "binary_pred": binary_pred,
    }


def run_all(data_root=DATA_ROOT, labels_path=None, split_filter=None):
    """Run pipeline on all participants found in data_root."""
    if labels_path is None:
        labels_path = os.path.join(data_root, "..", "LABELS.csv")

    labels_df = pd.read_csv(labels_path)
    if split_filter:
        labels_df = labels_df[labels_df["split"] == split_filter]

    text_enc = BertTextEncoder()
    audio_enc = Wav2Vec2AudioEncoder()
    model = FusionHead()

    results = []
    for pid in labels_df["participant_id"]:
        r = run_participant(int(pid), text_enc, audio_enc, model, data_root)
        if r:
            row = labels_df[labels_df["participant_id"] == pid].iloc[0]
            r["phq9_true"] = row["phq9_score"]
            r["hamd_true"] = row["hamd_score"]
            r["binary_true"] = int(row["hamd_score"] >= HAMD_CASENESS_THRESHOLD)
            r["split"] = row["split"]
            results.append(r)
            print(f"pid={pid} | true PHQ-9={r['phq9_true']} HAM-D={r['hamd_true']} | "
                  f"pred PHQ-9={r['phq9_pred']} HAM-D={r['hamd_pred']} | binary={r['binary_pred']}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions"
    )
    labels_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions", "LABELS.csv"
    )
    df = run_all(data_root=data_root, labels_path=labels_path)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "predictions"), exist_ok=True)
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "predictions", "predictions.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")
    print(df.to_string())
