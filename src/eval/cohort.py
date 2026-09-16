"""
Which participants an analysis is allowed to see.

The study grows in collection waves (data/kobo/waves.py). Wave 1 carries
train/dev/test splits and is what every reported result was computed on. A
later wave is held out: its participants have split == their wave name
("wave2"), so the wave stays a sample no model has seen.

Most consumers respect that without trying, because they select
split == "train". Three do not, by design: src/eval/benchmark.py and
src/eval/calibrate.py cross-validate over every row they are handed, and
scripts/fit_final_model.py fits on every row. Without a guard, a held-out wave
that reached a feature cache would be folded into training silently - no error,
just a slightly different number - and the study's only untouched sample would
be spent without anyone deciding to spend it.

So the rule lives here once. A participant is analysed only if their split is
one of TRAINING_SPLITS. Anything else is held out unless the caller passes
include_holdout=True, and so is a participant whose split cannot be determined,
because "unknown" is exactly what a newly-added wave looks like when a stale
LABELS.csv is being read.
"""
import os

import numpy as np
import pandas as pd

# Must match data/kobo/waves.py TRAINING_SPLITS (asserted in tests).
TRAINING_SPLITS = frozenset({"train", "dev", "test"})

DEFAULT_LABELS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "kobo", "sessions", "LABELS.csv"))


def splits_for(pids, labels_path=None, cache_splits=None):
    """Split per participant (None where unknown), or None if there is no source.

    LABELS.csv wins over splits stored in a cache: it is rewritten on every
    adapter run, whereas a cache records the splits as they were when it was
    encoded - before a wave was added, possibly.
    """
    if labels_path:
        lab = pd.read_csv(labels_path)
        by_pid = dict(zip(lab.participant_id.astype(int), lab.split.astype(str)))
        return [by_pid.get(int(p)) for p in pids]
    if cache_splits is not None:
        return [str(s) for s in cache_splits]
    return None


def select_cohort(pids, labels_path=None, cache_splits=None, include_holdout=False):
    """Boolean mask of participants to analyse, and a summary of what was dropped."""
    n = len(pids)
    splits = splits_for(pids, labels_path, cache_splits)
    if splits is None:
        return np.ones(n, dtype=bool), {
            "checked": False, "source": None, "analysed": n, "held_out": {},
            "unknown_split": 0, "include_holdout": bool(include_holdout)}

    keep = np.ones(n, dtype=bool)
    held, unknown = {}, 0
    for i, s in enumerate(splits):
        if s is None:
            unknown += 1
            keep[i] = include_holdout
        elif s not in TRAINING_SPLITS:
            held[s] = held.get(s, 0) + 1
            keep[i] = include_holdout
    return keep, {
        "checked": True, "source": "labels" if labels_path else "cache",
        "analysed": int(keep.sum()), "held_out": held, "unknown_split": unknown,
        "include_holdout": bool(include_holdout)}


def describe_cohort(info):
    if not info["checked"]:
        return (f"{info['analysed']} analysed - UNCHECKED: no split information, so "
                f"nothing guarantees a held-out wave is not among them")
    verb = "INCLUDED" if info["include_holdout"] else "excluded"
    held = ", ".join(f"{w}={k}" for w, k in sorted(info["held_out"].items())) or "none"
    return (f"{info['analysed']} analysed | held-out waves {verb}: {held} | "
            f"no known split ({verb}): {info['unknown_split']} | "
            f"splits read from {info['source']}")


def add_cohort_args(ap):
    ap.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help="LABELS.csv whose `split` column decides which participants are "
             "analysed. Pass 'none' only for a cache built before collection waves.")
    ap.add_argument(
        "--include-holdout", dest="include_holdout", action="store_true",
        help="also analyse held-out waves (e.g. wave2). This spends the hold-out: "
             "do it only after the frozen model has been evaluated on it.")


def resolve_labels(value):
    """--labels -> a path or None. A missing file is refused, never guessed around."""
    if value is None or str(value).strip().lower() == "none":
        return None
    if not os.path.exists(value):
        raise SystemExit(
            f"LABELS.csv not found at {value}.\n"
            f"It decides which participants are held out of analysis. Pass "
            f"--labels <path>, or --labels none for a cache that predates "
            f"collection waves.")
    return value
