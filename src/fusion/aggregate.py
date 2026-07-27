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
import numpy as np

TEXT_DIM = 768
AUDIO_DIM = 768
METADATA_DIM = 16  # keep in sync with metadata_pipeline.EMBED_DIM

# text + audio + metadata
FUSION_INPUT_DIM = TEXT_DIM + AUDIO_DIM + METADATA_DIM  # 1552


def mean_pool(embeddings: list) -> np.ndarray:
    """Mean-pool a list of np.ndarray segment embeddings into one vector."""
    if not embeddings:
        raise ValueError("Cannot pool empty embedding list — participant has no valid segments.")
    return np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)


def build_participant_vector(text_embeddings, audio_embeddings, metadata_vec):
    """
    Aggregates per-segment modality embeddings into one fixed-length
    participant-level fusion input vector.

    Args:
        text_embeddings:  list[np.ndarray], each shape (TEXT_DIM,)
        audio_embeddings: list[np.ndarray], each shape (AUDIO_DIM,)
        metadata_vec:     np.ndarray, shape (METADATA_DIM,)

    Returns:
        np.ndarray of shape (FUSION_INPUT_DIM,)
    """
    text_pooled = mean_pool(text_embeddings)
    audio_pooled = mean_pool(audio_embeddings)

    fusion_vec = np.concatenate([text_pooled, audio_pooled, metadata_vec])
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
