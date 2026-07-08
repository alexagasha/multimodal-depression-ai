"""
XAI attribution prototype: per-modality contribution to a participant's prediction.

Phase 1 approach: permutation-based feature importance.
  For each modality block in the fusion vector, zero it out and measure the
  drop in the predicted PHQ-9 score. Larger drop = larger modality contribution.
  (PHQ-9 is used as the primary target for attribution; HAM-D baseline is also
  reported for reference.)

This is a simple, model-agnostic approach that works on the numpy fusion head
with no extra dependencies (no SHAP/LIME needed in Phase 1).

SWAP for Phase 2: replace with SHAP KernelExplainer or GradientExplainer
once the PyTorch fusion head (FusionHeadTorch) is in use. The interface
here (explain_participant) won't change, only the internal impl.
"""
import numpy as np
from typing import Dict

from src.fusion.model import FusionHead
from src.fusion.aggregate import TEXT_DIM, AUDIO_DIM, METADATA_DIM, FUSION_INPUT_DIM

# Modality slices within the fusion vector [text | audio | metadata]
MODALITY_SLICES = {
    "text":     (0, TEXT_DIM),
    "audio":    (TEXT_DIM, TEXT_DIM + AUDIO_DIM),
    "metadata": (TEXT_DIM + AUDIO_DIM, TEXT_DIM + AUDIO_DIM + METADATA_DIM),
}


def explain_participant(
    fusion_vec: np.ndarray,
    model: FusionHead,
) -> Dict[str, float]:
    """
    Returns per-modality attribution scores for one participant.

    Attribution = drop in predicted PHQ-9 when that modality's block in the
    fusion vector is zeroed out. Normalised to sum to 1.0.

    Positive score = modality pushed score UP (toward depression).
    """
    baseline = model.forward(fusion_vec)
    baseline_pred = baseline["phq9"]
    attributions = {}

    for modality, (start, end) in MODALITY_SLICES.items():
        ablated = fusion_vec.copy()
        ablated[start:end] = 0.0
        ablated_pred = model.forward(ablated)["phq9"]
        attributions[modality] = float(baseline_pred - ablated_pred)

    # Normalise by total absolute attribution
    total = sum(abs(v) for v in attributions.values()) or 1.0
    normalised = {k: round(v / total, 4) for k, v in attributions.items()}

    return {
        "baseline_phq9_pred": round(float(baseline_pred), 3),
        "baseline_hamd_pred": round(float(baseline["hamd"]), 3),
        "modality_attributions": normalised,
    }


def explain_batch(predictions_df, fusion_vecs: dict, model: FusionHead):
    """
    Runs explain_participant for every participant in predictions_df.

    fusion_vecs: dict mapping participant_id (int) -> np.ndarray
    Returns list of attribution dicts with participant_id added.
    """
    results = []
    for _, row in predictions_df.iterrows():
        pid = int(row["participant_id"])
        if pid not in fusion_vecs:
            continue
        exp = explain_participant(fusion_vecs[pid], model)
        exp["participant_id"] = pid
        exp["phq9_true"] = row.get("phq9_true", None)
        results.append(exp)
    return results


if __name__ == "__main__":
    model = FusionHead()
    dummy_vec = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    result = explain_participant(dummy_vec, model)
    print("XAI attribution result:")
    print(f"  Baseline PHQ-9 prediction : {result['baseline_phq9_pred']}")
    print(f"  Baseline HAM-D prediction : {result['baseline_hamd_pred']}")
    for mod, score in result["modality_attributions"].items():
        direction = "up depression" if score > 0 else "down depression"
        print(f"  {mod:<12}: {score:+.4f}  ({direction})")
