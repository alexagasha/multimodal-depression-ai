"""
XAI attribution prototype: per-modality contribution to a participant's prediction.

Phase 1 approach: permutation-based feature importance.
  For each modality block in the fusion vector, zero it out and measure
  the drop in predicted PHQ-8 score. Larger drop = larger modality contribution.

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
    # TODO: VIDEO — add when video is active:
    # "video": (TEXT_DIM + AUDIO_DIM, TEXT_DIM + AUDIO_DIM + VIDEO_DIM)
    # and update metadata slice start accordingly
}


def explain_participant(
    fusion_vec: np.ndarray,
    model: FusionHead,
) -> Dict[str, float]:
    """
    Returns per-modality attribution scores for one participant.

    Attribution = drop in predicted PHQ-8 when that modality's block
    in the fusion vector is zeroed out. Normalised to sum to 1.0.

    Positive score = modality pushed score UP (toward depression).
    """
    baseline_pred = model.forward(fusion_vec)
    attributions = {}

    for modality, (start, end) in MODALITY_SLICES.items():
        ablated = fusion_vec.copy()
        ablated[start:end] = 0.0
        ablated_pred = model.forward(ablated)
        attributions[modality] = float(baseline_pred - ablated_pred)

    # Normalise by total absolute attribution
    total = sum(abs(v) for v in attributions.values()) or 1.0
    normalised = {k: round(v / total, 4) for k, v in attributions.items()}

    return {
        "baseline_phq8_pred": round(float(baseline_pred), 3),
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
        exp["phq8_true"] = row.get("phq8_true", None)
        results.append(exp)
    return results


if __name__ == "__main__":
    model = FusionHead()
    dummy_vec = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    result = explain_participant(dummy_vec, model)
    print("XAI attribution result:")
    print(f"  Baseline PHQ-8 prediction : {result['baseline_phq8_pred']}")
    for mod, score in result["modality_attributions"].items():
        direction = "↑ depression" if score > 0 else "↓ depression"
        print(f"  {mod:<12}: {score:+.4f}  ({direction})")
