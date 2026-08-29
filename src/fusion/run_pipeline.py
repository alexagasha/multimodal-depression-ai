"""
Orchestrates the full pipeline for each participant:
  sync -> text -> audio -> metadata -> aggregate -> fusion head -> prediction

This is the script that proves end-to-end plumbing works, and will become
the inference entry point once real data and trained weights are available.

Predicts both severity scores (PHQ-9 and HAM-D); binary caseness (the primary
classification label) is derived from HAM-D >= threshold (clinician gold standard).

Also surfaces risk_flag (see src/safety/risk_flag.py) when item-level PHQ-9
item 9 / HAM-D suicide-domain scores are supplied — a rule-based safety check,
independent of and never gated by the ML prediction.
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
from src.fusion.aggregate import build_participant_vector, run_acoustic_pipeline
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD
from src.safety.risk_flag import flag_risk


def run_participant(pid, text_enc, audio_enc, fusion_head, data_root=DATA_ROOT,
                     phq9_item9=None, hamd_suicide_item=None):
    """
    Full pipeline for a single participant. Returns result dict.

    phq9_item9 / hamd_suicide_item: optional item-level risk scores (see
    src/safety/risk_flag.py). When provided, risk_flag is computed
    independently of the ML prediction — a rule-based safety check that
    stands even if the fusion head is untrained.
    """
    segments = build_segments(pid, data_root=data_root)
    if not segments:
        print(f"[WARN] pid={pid}: no participant segments found, skipping.")
        return None

    text_embs = run_text_pipeline(segments, text_enc)
    acoustic = run_acoustic_pipeline(pid, segments, audio_enc, data_root=data_root)
    metadata_vec = run_metadata_pipeline(pid, data_root=data_root)

    fusion_vec = build_participant_vector(text_embs, acoustic, metadata_vec)
    scores = fusion_head.forward(fusion_vec)
    # calibrated cut point when the head carries one; see model.LinearHead
    binary_pred = int(scores["hamd"] >= fusion_head.caseness_threshold)

    result = {
        "participant_id": pid,
        "n_segments": len(segments),
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        "binary_pred": binary_pred,
    }
    if phq9_item9 is not None or hamd_suicide_item is not None:
        result["risk_flag"] = flag_risk(phq9_item9, hamd_suicide_item)
    return result


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
        row = labels_df[labels_df["participant_id"] == pid].iloc[0]
        phq9_item9 = row.get("phq9_item9_score")
        hamd_suicide_item = row.get("hamd_suicide_item_score")
        r = run_participant(int(pid), text_enc, audio_enc, model, data_root,
                             phq9_item9=phq9_item9, hamd_suicide_item=hamd_suicide_item)
        if r:
            r["phq9_true"] = row["phq9_score"]
            r["hamd_true"] = row["hamd_score"]
            r["binary_true"] = int(row["hamd_score"] >= HAMD_CASENESS_THRESHOLD)
            r["split"] = row["split"]
            results.append(r)
            risk_note = f" | RISK_FLAG={r['risk_flag']}" if r.get("risk_flag") else ""
            print(f"pid={pid} | true PHQ-9={r['phq9_true']} HAM-D={r['hamd_true']} | "
                  f"pred PHQ-9={r['phq9_pred']} HAM-D={r['hamd_pred']} | binary={r['binary_pred']}{risk_note}")

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
