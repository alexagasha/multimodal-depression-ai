"""
Baselines and cross-validated evaluation for the fusion head.

WHY THIS EXISTS
    With 135 participants and an 82% caseness base rate, two things make a
    single train/dev/test result untrustworthy:

      1. A constant "everyone is a case" predictor scores 82% accuracy. Any
         accuracy figure has to be read against that floor, which is why this
         reports BALANCED accuracy and AUC alongside it.
      2. The test split holds 21 people. One participant moving is ~5 points of
         accuracy, so the point estimate barely constrains anything.

    So this ignores the fixed split and runs repeated stratified k-fold over the
    whole cohort, reporting mean +/- SD across folds, for every combination of
    feature view and model. The fusion head only earns its 1552 dimensions if it
    beats metadata-only and duration-only here.

USAGE
    # after src/fusion/train.py has written its encode cache
    python src/eval/benchmark.py --cache outputs/cache/fusion_vectors.npz
    python src/eval/benchmark.py --cache ... --qc "<kobo>/sessions/QC.csv"
    python src/eval/benchmark.py --cache ... --permute      # leakage check

READING THE OUTPUT
    Compare every row against `constant`. A model that cannot beat the constant
    predictor on balanced accuracy has learned nothing, whatever its accuracy
    column says. Compare `fusion` against `metadata` and `duration`: if it does
    not clearly exceed both, the multimodal signal is not carrying the result.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fusion.aggregate import TEXT_DIM, AUDIO_DIM, METADATA_DIM
from src.fusion.model import HAMD_CASENESS_THRESHOLD, PHQ9_RANGE, HAMD_RANGE

warnings.filterwarnings("ignore")

TEXT_SLICE = slice(0, TEXT_DIM)
AUDIO_SLICE = slice(TEXT_DIM, TEXT_DIM + AUDIO_DIM)
META_SLICE = slice(TEXT_DIM + AUDIO_DIM, TEXT_DIM + AUDIO_DIM + METADATA_DIM)


# --------------------------------------------------------------------------
def load_cache(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    Y = d["Y"].astype(np.float64)
    pids = d["pids"]
    print(f"cache: X={X.shape}  Y={Y.shape}  participants={len(pids)}")
    return X, Y, pids


def attach_duration(pids, qc_path):
    """Duration features per participant, for the shortcut baseline."""
    qc = pd.read_csv(qc_path).set_index("participant_id")
    cols = [c for c in ("speech_sec", "audio_sec", "speech_per_answer") if c in qc.columns]
    rows = []
    for p in pids:
        p = int(p)
        rows.append([float(qc.at[p, c]) if p in qc.index and pd.notna(qc.at[p, c]) else 0.0
                     for c in cols])
    print(f"duration features: {cols}")
    return np.asarray(rows, dtype=np.float64)


def attach_prosody(pids, prosody_path, drop_duration=True):
    """Explicit prosodic features (pitch, energy, timing) per participant.

    These exist because wav2vec2-base-960h is an ASR model: its final layer
    encodes *what was said*, which the text branch already covers, and discards
    much of *how* it was said. Pitch variability, pause structure and speech
    rate are the paralinguistic markers depression is actually described by,
    and at N=135 a dozen interpretable numbers are a better bet than 768
    opaque ones."""
    pr = pd.read_csv(prosody_path).set_index("participant_id")
    cols = [c for c in pr.columns if pr[c].dtype.kind in "fi"]
    if drop_duration:
        # n_clips / dur_* are how MUCH was said, which duration-only already
        # tests. Excluding them is what makes "does acoustic character beat a
        # stopwatch" an honest question rather than a restatement of it.
        cols = [c for c in cols if not (c.startswith("dur_") or c == "n_clips")]
    rows, missing = [], 0
    for p in pids:
        p = int(p)
        if p in pr.index:
            rows.append([float(pr.at[p, c]) if pd.notna(pr.at[p, c]) else 0.0 for c in cols])
        else:
            rows.append([0.0] * len(cols)); missing += 1
    if missing:
        print(f"  ! {missing} participant(s) have no prosody row -> zero-filled")
    print(f"prosody features ({len(cols)}): {cols}")
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------
def make_models(seed):
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def ridge(alpha):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def pca_ridge(k, alpha):
        return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                             Ridge(alpha=alpha))

    def mlp():
        # the architecture currently in src/fusion/model.py (1552 -> 256 -> 64 -> 2)
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-4, max_iter=2000,
                         early_stopping=False, random_state=seed))

    return {
        "ridge(a=100)":  ridge(100.0),
        "ridge(a=1000)": ridge(1000.0),
        "pca32+ridge":   pca_ridge(32, 10.0),
        "mlp 256-64":    mlp(),
    }


class ConstantRegressor:
    """Predicts the training mean. The floor every model must clear."""
    def fit(self, X, y):
        self.mu_ = np.asarray(y, dtype=float).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mu_, (len(X), 1))


# --------------------------------------------------------------------------
def fold_metrics(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
    from scipy.stats import pearsonr

    phq_t, ham_t = y_true[:, 0], y_true[:, 1]
    phq_p = np.clip(y_pred[:, 0], *PHQ9_RANGE)
    ham_p = np.clip(y_pred[:, 1], *HAMD_RANGE)

    bt = (ham_t >= HAMD_CASENESS_THRESHOLD).astype(int)
    bp = (ham_p >= HAMD_CASENESS_THRESHOLD).astype(int)

    out = {
        "phq9_mae": float(np.mean(np.abs(phq_t - phq_p))),
        "hamd_mae": float(np.mean(np.abs(ham_t - ham_p))),
        "accuracy": float(accuracy_score(bt, bp)),
        "bal_acc": float(balanced_accuracy_score(bt, bp)) if len(set(bt)) > 1 else np.nan,
    }
    # correlation is undefined when a fold's predictions are constant
    out["hamd_r"] = (float(pearsonr(ham_t, ham_p)[0])
                     if np.std(ham_p) > 1e-9 and np.std(ham_t) > 1e-9 else np.nan)
    # AUC uses the continuous predicted HAM-D, so it is threshold-free
    out["auc"] = (float(roc_auc_score(bt, ham_p))
                  if len(set(bt)) > 1 and np.std(ham_p) > 1e-9 else np.nan)
    tp = int(((bt == 1) & (bp == 1)).sum()); fn = int(((bt == 1) & (bp == 0)).sum())
    tn = int(((bt == 0) & (bp == 0)).sum()); fp = int(((bt == 0) & (bp == 1)).sum())
    out["sens"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["spec"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out


def cross_validate(X, Y, model_factory, n_splits=5, n_repeats=5, seed=0):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.base import clone

    strat = (Y[:, 1] >= HAMD_CASENESS_THRESHOLD).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for tr, te in cv.split(X, strat):
        m = clone(model_factory) if hasattr(model_factory, "get_params") else model_factory
        m.fit(X[tr], Y[tr])
        rows.append(fold_metrics(Y[te], np.asarray(m.predict(X[te]), dtype=float)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def permutation_test(X, Y, model_factory, n_perm=50, folds=5, repeats=2, seed=0):
    """Proper null distribution for one (view, model) pair.

    A SINGLE label shuffle is one draw from the null, not the null itself. With
    768 features against 135 participants that distribution is wide and does not
    sit at 0.5 — some directions in X correlate with any fixed label vector by
    chance, and cross-validation reproduces that because the same fixed noise is
    present in every fold. So the honest question is not "is the observed AUC
    above 0.5" but "where does it fall in the distribution of AUCs obtained from
    shuffled labels".
    """
    rng = np.random.default_rng(seed)
    obs = cross_validate(X, Y, model_factory, folds, repeats, seed)["auc"].mean()
    null = []
    for i in range(n_perm):
        Yp = Y[rng.permutation(len(Y))]
        null.append(cross_validate(X, Yp, model_factory, folds, repeats, seed + i)["auc"].mean())
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_perm} permutations", flush=True)
    null = np.array(null, dtype=float)
    # +1 correction: the observed value is itself one draw under the null
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float(p), "null": null}


def main():
    ap = argparse.ArgumentParser(description="Baselines + cross-validated evaluation.")
    ap.add_argument("--cache", default=os.path.join("outputs", "cache", "fusion_vectors.npz"))
    ap.add_argument("--qc", default=None, help="QC.csv, to add the duration-only baseline")
    ap.add_argument("--prosody", default=None,
                    help="prosody.csv (participant_id + feature columns), to add "
                         "the prosody views")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", action="store_true",
                    help="single label shuffle (indicative only — see --permtest)")
    ap.add_argument("--permtest", type=int, default=0, metavar="N",
                    help="proper permutation test: N shuffles per view, giving a null "
                         "distribution and an empirical p-value for the observed AUC")
    ap.add_argument("--out", default=os.path.join("outputs", "benchmark.csv"))
    a = ap.parse_args()

    if not os.path.exists(a.cache):
        sys.exit(f"No encode cache at {a.cache}. Run src/fusion/train.py first — it "
                 f"writes the cache while encoding.")

    X, Y, pids = load_cache(a.cache)
    strat = (Y[:, 1] >= HAMD_CASENESS_THRESHOLD).astype(int)
    print(f"caseness: {int(strat.sum())} cases / {int((1 - strat).sum())} non-cases "
          f"({strat.mean():.1%} positive)")
    print(f"a constant 'always a case' predictor therefore scores {max(strat.mean(), 1-strat.mean()):.1%} "
          f"accuracy and exactly 50.0% balanced accuracy\n")

    if a.permute:
        rng = np.random.default_rng(a.seed)
        Y = Y[rng.permutation(len(Y))]
        print("!! LABELS PERMUTED — anything scoring above chance here is leakage\n")

    views = {
        "fusion (1552d)": X,
        "text (768d)":    X[:, TEXT_SLICE],
        "audio (768d)":   X[:, AUDIO_SLICE],
        "metadata (16d)": X[:, META_SLICE],
    }
    if a.qc and os.path.exists(a.qc):
        views["duration only"] = attach_duration(pids, a.qc)
    if a.prosody and os.path.exists(a.prosody):
        P = attach_prosody(pids, a.prosody, drop_duration=True)
        Pd = attach_prosody(pids, a.prosody, drop_duration=False)
        views["prosody (no dur)"] = P
        views["prosody + dur"] = Pd
        views["text+prosody"] = np.hstack([X[:, TEXT_SLICE], P])
        views["all (no w2v)"] = np.hstack([X[:, TEXT_SLICE], P, X[:, META_SLICE]])

    if a.permtest:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
        print(f"PERMUTATION TEST — {a.permtest} shuffles per view, ridge(a=1000)\n")
        rows = []
        for vname, Xv in views.items():
            print(f"  {vname} ...", flush=True)
            r = permutation_test(Xv, Y, model, a.permtest, a.folds, 2, a.seed)
            rows.append({"view": vname, **{k: v for k, v in r.items() if k != "null"}})
            print(f"    observed AUC {r['observed']:.3f} | null {r['null_mean']:.3f}"
                  f"±{r['null_sd']:.3f} (95th pct {r['null_p95']:.3f}) | p = {r['p_value']:.3f}\n")
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
        print("\nA view is only carrying real signal if its observed AUC exceeds the 95th")
        print("percentile of its own null, i.e. p < 0.05. Comparing against 0.5 is wrong")
        print("here — the null does not sit at 0.5 when features outnumber participants.")
        out.to_csv(a.out.replace(".csv", "_permtest.csv"), index=False)
        return

    results = []
    # the floor, computed once on the full feature set (it ignores X anyway)
    df = cross_validate(X, Y, ConstantRegressor(), a.folds, a.repeats, a.seed)
    results.append(("constant (train mean)", "—", df))

    for vname, Xv in views.items():
        for mname, model in make_models(a.seed).items():
            # a 4-parameter view does not need PCA to 32 components
            if "pca" in mname and Xv.shape[1] <= 40:
                continue
            df = cross_validate(Xv, Y, model, a.folds, a.repeats, a.seed)
            results.append((vname, mname, df))

    # ---- report ----
    cols = ["bal_acc", "auc", "accuracy", "sens", "spec", "hamd_mae", "hamd_r", "phq9_mae"]
    hdr = (f"{'view':<16} {'model':<14} " +
           " ".join(f"{c:>13}" for c in cols))
    print("\n" + "=" * len(hdr))
    print(f"{a.repeats}x{a.folds}-fold stratified CV over {len(Y)} participants — mean +/- SD")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    rows_out = []
    for vname, mname, df in results:
        cells = []
        for c in cols:
            m, s = df[c].mean(), df[c].std()
            cells.append("nan" if np.isnan(m) else f"{m:.3f}±{s:.3f}")
        print(f"{vname:<16} {mname:<14} " + " ".join(f"{c:>13}" for c in cells))
        rows_out.append({"view": vname, "model": mname,
                         **{f"{c}_mean": df[c].mean() for c in cols},
                         **{f"{c}_sd": df[c].std() for c in cols}})
    print("=" * len(hdr))

    print("\nHow to read this:")
    print("  bal_acc 0.500 = chance. Accuracy near the base rate with bal_acc near 0.5")
    print("  means the model is predicting 'case' for everyone and has learned nothing.")
    print("  auc is threshold-free and the most honest single number here.")
    print("  If 'fusion' does not clearly beat 'metadata' and 'duration only',")
    print("  the multimodal signal is not what is driving the result.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows_out).to_csv(a.out, index=False)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
