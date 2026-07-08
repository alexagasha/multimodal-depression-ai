"""
Lightweight fusion head: concatenated participant vector -> PHQ-8 severity score.

Phase 1 architecture (numpy-only, no torch dependency):
  Linear(1539 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 64) -> ReLU -> Linear(64 -> 1)

This is a deterministic forward-pass stub with random weights that proves
the shape plumbing is correct end-to-end. Weights are not trained here —
training happens in train.py once real (or larger synthetic) data is ready.

SWAP instructions for real PyTorch model:
  See the FusionHeadTorch class at the bottom of this file.
  Replace FusionHead with FusionHeadTorch in run_pipeline.py and train.py.

Output: continuous float in [0, 24] (PHQ-8 range), clamped at inference.
Threshold for binary detection: PHQ-8 >= 10 -> depressed.
"""
import numpy as np

from .aggregate import FUSION_INPUT_DIM

HIDDEN1 = 256
HIDDEN2 = 64
OUTPUT_DIM = 1
PHQ8_MIN, PHQ8_MAX = 0.0, 24.0
DEPRESSION_THRESHOLD = 10.0


def relu(x):
    return np.maximum(0, x)


class FusionHead:
    """
    Numpy-only fusion MLP. Weights are random at init (untrained).
    Replace with FusionHeadTorch when ready to train for real.
    """

    def __init__(self, input_dim=FUSION_INPUT_DIM, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / HIDDEN1)
        scale3 = np.sqrt(2.0 / HIDDEN2)
        self.W1 = rng.normal(0, scale1, (HIDDEN1, input_dim)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (HIDDEN2, HIDDEN1)).astype(np.float32)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = rng.normal(0, scale3, (OUTPUT_DIM, HIDDEN2)).astype(np.float32)
        self.b3 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    def forward(self, x: np.ndarray) -> float:
        """x: shape (FUSION_INPUT_DIM,) -> scalar PHQ-8 prediction."""
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        out = (self.W3 @ h2 + self.b3).flatten()[0]
        return float(np.clip(out, PHQ8_MIN, PHQ8_MAX))

    def predict_label(self, x: np.ndarray) -> int:
        """Binary: 1 if predicted PHQ-8 >= threshold, else 0."""
        return int(self.forward(x) >= DEPRESSION_THRESHOLD)


# ---------------------------------------------------------------------------
# SWAP: PyTorch version — replace FusionHead with this once training is ready
# ---------------------------------------------------------------------------
# import torch
# import torch.nn as nn
#
# class FusionHeadTorch(nn.Module):
#     def __init__(self, input_dim=FUSION_INPUT_DIM):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, HIDDEN1), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(HIDDEN1, HIDDEN2), nn.ReLU(),
#             nn.Linear(HIDDEN2, OUTPUT_DIM),
#         )
#     def forward(self, x):          # x: (batch, input_dim)
#         return self.net(x).squeeze(-1)   # (batch,)


if __name__ == "__main__":
    model = FusionHead()
    dummy = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    score = model.forward(dummy)
    label = model.predict_label(dummy)
    print(f"PHQ-8 prediction: {score:.2f}, binary label: {label}")
