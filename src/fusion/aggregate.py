"""
Aggregation: per-segment embeddings -> single participant-level vector per modality.

Strategy: mean-pool across N segments within each modality independently,
then concatenate the pooled vectors + metadata into one fusion input.

N is variable per participant (ragged sessions) — mean pooling absorbs
this naturally without padding, truncation, or special batching logic.

Fusion vector layout:
  [text_pooled (768) | audio_pooled (768) | metadata (16)]
  Total input dim = 1552
"""
import os

import numpy as np

TEXT_DIM = 768
AUDIO_DIM = 768        # wav2vec2-base-960h, per 30 s segment
PROSODY_DIM = 24       # keep in sync with prosody_pipeline.EMBED_DIM
METADATA_DIM = 16      # keep in sync with metadata_pipeline.EMBED_DIM

# Which acoustic representation is served.
#
#   prosody   24 interpretable measures. AUC 0.826, permutation p = 0.016 —
#             the best single modality evaluated, and nameable to a clinician.
#   wav2vec2  768-d self-supervised embedding. AUC 0.605, p = 0.082: it did not
#             clear its own permutation null, and lost to three measures of
#             speech quantity. Retained for the thesis comparison only.
#
# Default is still wav2vec2 while api/main.py assembles that vector; Day 3 of
# docs/mvp-plan.md flips it to prosody together with the scoring path, so that
# the served features and the served weights change in one step.
ACOUSTIC = os.environ.get("DEP_ACOUSTIC", "wav2vec2").strip().lower()
if ACOUSTIC not in ("wav2vec2", "prosody"):
    raise ValueError(f"DEP_ACOUSTIC must be 'wav2vec2' or 'prosody', got {ACOUSTIC!r}")
ACOUSTIC_DIM = AUDIO_DIM if ACOUSTIC == "wav2vec2" else PROSODY_DIM

FUSION_INPUT_DIM = TEXT_DIM + ACOUSTIC_DIM + METADATA_DIM   # 1552 or 808


def mean_pool(embeddings: list) -> np.ndarray:
    """Mean-pool a list of np.ndarray segment embeddings into one vector."""
    if not embeddings:
        raise ValueError("Cannot pool empty embedding list — participant has no valid segments.")
    return np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)


def build_participant_vector(text_embeddings, acoustic, metadata_vec):
    """
    Aggregates modality representations into one participant-level vector.

    Args:
        text_embeddings: list[np.ndarray], each shape (TEXT_DIM,), one per segment
        acoustic:        either a list of per-segment embeddings (wav2vec2, which
                         is mean-pooled here) or an already participant-level
                         vector (prosody, which is aggregated across turns by
                         prosody_pipeline and must not be pooled again)
        metadata_vec:    np.ndarray, shape (METADATA_DIM,)

    Returns:
        np.ndarray of shape (FUSION_INPUT_DIM,)

    The concatenation order — text, acoustic, metadata — is the order the
    shipped weights were fitted in. Changing it silently changes what every
    coefficient multiplies.
    """
    text_pooled = mean_pool(text_embeddings)
    acoustic_pooled = (mean_pool(acoustic) if isinstance(acoustic, (list, tuple))
                       else np.asarray(acoustic, dtype=np.float32).ravel())

    if acoustic_pooled.shape[0] != ACOUSTIC_DIM:
        raise ValueError(
            f"acoustic representation has {acoustic_pooled.shape[0]} dimensions but "
            f"DEP_ACOUSTIC={ACOUSTIC!r} expects {ACOUSTIC_DIM}. The served features "
            f"and the served weights must describe the same thing.")

    fusion_vec = np.concatenate([text_pooled, acoustic_pooled, metadata_vec])
    assert fusion_vec.shape == (FUSION_INPUT_DIM,), (
        f"Fusion vector dim mismatch: got {fusion_vec.shape}, expected ({FUSION_INPUT_DIM},)"
    )
    return fusion_vec


if __name__ == "__main__":
    dummy_text = [np.random.randn(TEXT_DIM).astype(np.float32) for _ in range(5)]
    dummy_audio = [np.random.randn(AUDIO_DIM).astype(np.float32) for _ in range(5)]
    dummy_meta = np.random.randn(METADATA_DIM).astype(np.float32)
    vec = build_participant_vector(dummy_text, dummy_audio, dummy_meta)
    print(f"Participant fusion vector shape: {vec.shape}")  # expect (1552,)
