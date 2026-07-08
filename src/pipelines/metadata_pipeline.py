"""
Metadata pipeline: sociodemographic + clinical fields -> fixed-length vector.

Simplest pipeline of the four:
  - one vector per participant (not per segment)
  - the same vector is attached to every segment for that participant
  - no time-alignment or segmenting needed

Fields expected (matching synthetic generator and anticipated E-DAIC-WOZ metadata):
  age              int
  gender           str  ("male" / "female")
  education_years  int

When E-DAIC-WOZ access clears, extend FIELD_DEFAULTS with any additional
demographic/clinical fields present in the real metadata files.
"""
import json
import os

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")

GENDER_MAP = {"male": 0.0, "female": 1.0, "other": 0.5}

# Approximate normalisation constants (update from real dataset statistics)
AGE_MEAN, AGE_STD = 35.0, 12.0
EDU_MEAN, EDU_STD = 14.0, 3.0

EMBED_DIM = 3  # [age_norm, gender_enc, education_norm]


def load_metadata(pid, data_root=DATA_ROOT):
    path = os.path.join(data_root, str(pid), f"{pid}_METADATA.json")
    with open(path) as f:
        return json.load(f)


def encode_metadata(meta: dict) -> np.ndarray:
    age_norm = (float(meta.get("age", AGE_MEAN)) - AGE_MEAN) / AGE_STD
    gender_enc = GENDER_MAP.get(str(meta.get("gender", "other")).lower(), 0.5)
    edu_norm = (float(meta.get("education_years", EDU_MEAN)) - EDU_MEAN) / EDU_STD

    # SWAP: add more fields here as they become available from E-DAIC-WOZ metadata.
    # Extend EMBED_DIM accordingly and update the fusion head input_dim in model.py.
    return np.array([age_norm, gender_enc, edu_norm], dtype=np.float32)


def run_metadata_pipeline(pid, data_root=DATA_ROOT) -> np.ndarray:
    """Returns a single fixed-length vector for the participant."""
    meta = load_metadata(pid, data_root)
    return encode_metadata(meta)


if __name__ == "__main__":
    vec = run_metadata_pipeline(300)
    print(f"Metadata vector (pid=300): {vec}, dim={vec.shape}")
