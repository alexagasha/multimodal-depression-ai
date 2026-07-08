"""
Train the fusion head on labeled sessions (PyTorch; runs on Colab GPU).

Two phases:
  1) ENCODE (once): push every labeled participant through the FROZEN encoders
     -> one fusion vector per participant. Cached to disk so retraining the head
     does not re-encode audio/text.
  2) TRAIN: fit a small MLP (identical architecture to the numpy FusionHead) to
     predict [phq9, hamd], then export the weights to .npz so the numpy inference
     path (run_pipeline / FusionHead.load) uses them with NO torch dependency.

Only the head trains — BERT/Wav2Vec2 stay frozen. That is why you never label
audio or text: the single supervision signal is the participant's PHQ-9/HAM-D
score (attached to the whole session), and the head learns
[text | audio | metadata embedding] -> [phq9, hamd] across participants.

Usage (from the repo root):
  python src/fusion/train.py --data_root data/edaic/sessions --epochs 300
Then in inference:
  from src.fusion.model import FusionHead
  model = FusionHead.load("outputs/weights/fusion_head.npz")
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.sync import build_segments
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.fusion.aggregate import build_participant_vector, FUSION_INPUT_DIM
from src.fusion.model import (
    FusionHead, HIDDEN1, HIDDEN2, OUTPUT_DIM, PHQ9_RANGE, HAMD_RANGE, HAMD_CASENESS_THRESHOLD,
)
from src.eval.metrics import compute_metrics


def build_dataset(data_root, labels_path, cache_path=None, recache=False):
    """Encode every labeled participant -> (X, Y, splits, pids). Cached to .npz."""
    if cache_path and os.path.exists(cache_path) and not recache:
        d = np.load(cache_path, allow_pickle=True)
        print(f"[cache] loaded {cache_path}  X={d['X'].shape}")
        return d["X"], d["Y"], d["splits"], d["pids"]

    labels = pd.read_csv(labels_path)
    text_enc = BertTextEncoder()
    audio_enc = Wav2Vec2AudioEncoder()

    X, Y, splits, pids = [], [], [], []
    for _, row in labels.iterrows():
        pid = int(row["participant_id"])
        if pd.isna(row.get("phq9_score")) or pd.isna(row.get("hamd_score")):
            print(f"[skip] pid={pid}: missing phq9/hamd label")
            continue
        segs = build_segments(pid, data_root=data_root)
        if not segs:
            print(f"[skip] pid={pid}: no segments")
            continue
        t = run_text_pipeline(segs, text_enc)
        a = run_audio_pipeline(pid, segs, audio_enc, data_root=data_root)
        m = run_metadata_pipeline(pid, data_root=data_root)
        X.append(build_participant_vector(t, a, m))
        Y.append([float(row["phq9_score"]), float(row["hamd_score"])])
        splits.append(str(row.get("split", "train")))
        pids.append(pid)
        print(f"[enc] pid={pid}  segments={len(segs)}")

    if not X:
        raise RuntimeError("No labeled participants encoded — check LABELS.csv / data_root.")

    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    splits = np.array(splits)
    pids = np.array(pids)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, X=X, Y=Y, splits=splits, pids=pids)
        print(f"[cache] saved {cache_path}")
    return X, Y, splits, pids


def _torch_head(input_dim, dropout=0.3):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(HIDDEN1, HIDDEN2), nn.ReLU(),
        nn.Linear(HIDDEN2, OUTPUT_DIM),   # -> [phq9, hamd]
    )


def train(X, Y, splits, epochs=300, lr=1e-3, weight_decay=1e-4, seed=0, verbose=True):
    """Fit the MLP head. Loss is normalized per target so PHQ-9 and HAM-D weigh equally."""
    import torch

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = torch.tensor([PHQ9_RANGE[1], HAMD_RANGE[1]], dtype=torch.float32, device=device)

    tr = splits == "train"
    dv = splits == "dev"
    if tr.sum() == 0:
        tr = np.ones(len(splits), dtype=bool)   # no explicit train split -> use all
    if dv.sum() == 0:
        dv = tr                                  # no dev split -> monitor on train

    Xtr = torch.tensor(X[tr], device=device)
    Ytr = torch.tensor(Y[tr], device=device)
    Xdv = torch.tensor(X[dv], device=device)
    Ydv = torch.tensor(Y[dv], device=device)

    net = _torch_head(X.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    mse = torch.nn.MSELoss()

    best = {"dev": float("inf"), "state": None}
    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = mse(net(Xtr) / scale, Ytr / scale)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            dev_loss = mse(net(Xdv) / scale, Ydv / scale).item()
        if dev_loss < best["dev"]:
            best = {"dev": dev_loss,
                    "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"epoch {ep:4d}  train_loss={loss.item():.4f}  dev_loss={dev_loss:.4f}")

    net.load_state_dict(best["state"])
    if verbose:
        print(f"[best] dev_loss={best['dev']:.4f}")
    return net


def export_numpy(net, out_path):
    """Copy the trained torch Linear weights into a numpy FusionHead .npz."""
    import torch

    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3, f"expected 3 Linear layers, got {len(linears)}"
    w = {"W1": linears[0].weight.detach().cpu().numpy(), "b1": linears[0].bias.detach().cpu().numpy(),
         "W2": linears[1].weight.detach().cpu().numpy(), "b2": linears[1].bias.detach().cpu().numpy(),
         "W3": linears[2].weight.detach().cpu().numpy(), "b3": linears[2].bias.detach().cpu().numpy()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **w)
    print(f"[save] trained head -> {out_path}")
    return out_path


def evaluate(weights_path, X, Y, splits, pids, which):
    """Run the numpy FusionHead (loaded from trained weights) over one split + metrics."""
    head = FusionHead.load(weights_path)
    idx = np.where(splits == which)[0]
    if len(idx) == 0:
        return None
    rows = []
    for i in idx:
        s = head.forward(X[i])
        rows.append({
            "participant_id": int(pids[i]),
            "phq9_pred": s["phq9"], "hamd_pred": s["hamd"],
            "phq9_true": float(Y[i, 0]), "hamd_true": float(Y[i, 1]),
            "binary_pred": int(s["hamd"] >= HAMD_CASENESS_THRESHOLD),
            "binary_true": int(Y[i, 1] >= HAMD_CASENESS_THRESHOLD),
        })
    print(f"\n===== {which.upper()} =====")
    return compute_metrics(pd.DataFrame(rows), verbose=True)


def main():
    ap = argparse.ArgumentParser(description="Train the fusion head.")
    ap.add_argument("--data_root", default=os.path.join("data", "synthetic", "sessions"))
    ap.add_argument("--labels", default=None, help="defaults to <data_root>/LABELS.csv")
    ap.add_argument("--cache", default=os.path.join("outputs", "cache", "fusion_vectors.npz"))
    ap.add_argument("--weights", default=os.path.join("outputs", "weights", "fusion_head.npz"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--recache", action="store_true", help="re-encode even if a cache exists")
    args = ap.parse_args()

    labels_path = args.labels or os.path.join(args.data_root, "LABELS.csv")
    X, Y, splits, pids = build_dataset(args.data_root, labels_path, args.cache, args.recache)
    counts = dict(zip(*np.unique(splits, return_counts=True)))
    print(f"dataset: X={X.shape}  Y={Y.shape}  splits={counts}")

    net = train(X, Y, splits, epochs=args.epochs, lr=args.lr)
    export_numpy(net, args.weights)

    for split in ("dev", "test"):
        evaluate(args.weights, X, Y, splits, pids, split)

    print(f"\nDone. Load the trained head in inference with:"
          f"\n  FusionHead.load('{args.weights}')")


if __name__ == "__main__":
    main()
