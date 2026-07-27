"""
Metadata pipeline: Section A sociodemographic fields -> fixed-length numeric
vector.

Simplest pipeline of the four:
  - one vector per participant (not per segment)
  - the same vector is attached to every segment for that participant
  - no time-alignment or segmenting needed

Encoding (matches the real data-collection tool, Section A):
  age_band          ordinal  (1)   18-25 .. 56-65
  sex               binary   (1)
  marital_status    one-hot  (4)   married / divorced / widowed / single
  ethnicity         one-hot  (3)   bantu / luo / other
  residence         ordinal  (1)   urban / semi-urban / rural
  education_level   ordinal  (1)   none / primary / secondary / tertiary_university
  employment_status one-hot  (4)   employed / unemployed / self-employed / student
  smartphone        binary   (1)
                    -----------------------------------------------------------
                    EMBED_DIM = 16

NOTE: `site` (Butabika/Mulago/Other) is intentionally EXCLUDED from the feature
vector — it is recruitment provenance and would let the model shortcut on
"which hospital" instead of symptoms. It stays in the metadata JSON for
stratified analysis only.

The study is English-only, so there is no `language` field.
"""
import json
import os

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")

# Ordinal scales (encoded to [0, 1]; unknown -> 0.5)
AGE_BAND_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-65"]
RESIDENCE_ORDER = ["urban", "semi-urban", "rural"]
EDUCATION_ORDER = ["none", "primary", "secondary", "tertiary_university"]

# Nominal categories (one-hot; unknown -> all zeros)
MARITAL_CATS = ["married", "divorced", "widowed", "single"]
ETHNICITY_CATS = ["bantu", "luo", "other"]
EMPLOYMENT_CATS = ["employed", "unemployed", "self-employed", "student"]

# 1 + 1 + 4 + 3 + 1 + 1 + 4 + 1
EMBED_DIM = 16


def _ordinal(value, order):
    v = str(value).strip().lower()
    if v in order:
        return order.index(v) / (len(order) - 1)
    return 0.5  # unknown / missing


def _one_hot(value, cats):
    vec = np.zeros(len(cats), dtype=np.float32)
    v = str(value).strip().lower()
    if v in cats:
        vec[cats.index(v)] = 1.0
    return vec  # unknown -> all zeros


def _binary(value, true_token):
    v = str(value).strip().lower()
    if v == true_token:
        return 1.0
    if v in ("", "none", "nan", "unknown"):
        return 0.5
    return 0.0


def load_metadata(pid, data_root=DATA_ROOT):
    path = os.path.join(data_root, str(pid), f"{pid}_METADATA.json")
    with open(path) as f:
        return json.load(f)


def encode_metadata(meta: dict) -> np.ndarray:
    parts = [
        np.array([_ordinal(meta.get("age_band"), AGE_BAND_ORDER)], dtype=np.float32),
        np.array([_binary(meta.get("sex"), "female")], dtype=np.float32),
        _one_hot(meta.get("marital_status"), MARITAL_CATS),
        _one_hot(meta.get("ethnicity"), ETHNICITY_CATS),
        np.array([_ordinal(meta.get("residence"), RESIDENCE_ORDER)], dtype=np.float32),
        np.array([_ordinal(meta.get("education_level"), EDUCATION_ORDER)], dtype=np.float32),
        _one_hot(meta.get("employment_status"), EMPLOYMENT_CATS),
        np.array([_binary(meta.get("smartphone"), "yes")], dtype=np.float32),
    ]
    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape == (EMBED_DIM,), f"metadata dim mismatch: {vec.shape} != ({EMBED_DIM},)"
    return vec


def run_metadata_pipeline(pid, data_root=DATA_ROOT) -> np.ndarray:
    """Returns a single fixed-length vector for the participant."""
    meta = load_metadata(pid, data_root)
    return encode_metadata(meta)


if __name__ == "__main__":
    vec = run_metadata_pipeline(300)
    print(f"Metadata vector (pid=300): dim={vec.shape}\n{vec}")
