"""
Cut a model release: pin the exact weights a field installation may serve.

A release is two things with one name:
    outputs/weights/RELEASE.json   the release name plus the SHA-256 of every
                                   weights file in it
    a git tag                      on the commit that contains that manifest

api/readiness.py refuses to score with weights whose hash is not in the
manifest. So swapping in a refitted file, or serving a local experiment through
DEP_WEIGHTS, stops scoring instead of quietly changing what every field score
means.

It takes two steps, because a tag has to point at a commit that already holds
the manifest it describes:

    1. python scripts/tag_release.py write --release model-2026.09.1 --notes "..."
       Writes the manifest. Read it, then commit it together with the weights.

    2. python scripts/tag_release.py tag
       Checks that the committed manifest still matches the weights on disk and
       that nothing the served model depends on is uncommitted, then creates an
       annotated git tag. Pushing the tag is left to you.

Releases are immutable. A name that is already tagged is refused - refit, and
cut a new release.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)

from api.readiness import sha256_file  # noqa: E402

WEIGHTS_DIR = os.path.join(REPO, "outputs", "weights")
MANIFEST = os.path.join(WEIGHTS_DIR, "RELEASE.json")
NAME_RE = re.compile(r"^model-\d{4}\.\d{2}\.\d+$")
# Everything the served model's behaviour depends on. A tag over uncommitted
# changes here would name a state nobody can check out again.
DEPENDS_ON = ["api", "src", "scripts", "web", "data/kobo", "outputs/weights", "requirements.txt"]


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def tag_exists(name) -> bool:
    return git("rev-parse", "-q", "--verify", f"refs/tags/{name}").returncode == 0


def weights_entry(path) -> dict:
    """What a weights file records about itself, plus its hash."""
    from src.fusion.model import load_head
    head = load_head(path)
    meta = getattr(head, "meta", {}) or {}
    dim = getattr(head, "input_dim", None) or int(head.W1.shape[1])
    return {
        "file": os.path.basename(path),
        "sha256": sha256_file(path),
        "fitted": meta.get("fitted"),
        "n_participants": meta.get("n_participants"),
        "feature_set": meta.get("feature_set"),
        "feature_dim": int(dim),
        "threshold": meta.get("threshold"),
        "cohort": meta.get("cohort"),
    }


def build_manifest(release, notes, weight_paths) -> dict:
    if not NAME_RE.match(release):
        raise SystemExit(f"release name {release!r} must look like model-YYYY.MM.N")
    if not weight_paths:
        raise SystemExit("no weights files to release")
    return {
        "release": release,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "evaluation": "docs/model-card.md",
        "weights": [weights_entry(p) for p in weight_paths],
    }


def verify_weights(manifest, weights_dir=WEIGHTS_DIR) -> list:
    """Problems with the weights a manifest lists. Empty means they all match."""
    problems = []
    for w in manifest.get("weights", []):
        path = os.path.join(weights_dir, w["file"])
        if not os.path.exists(path):
            problems.append(f"{w['file']} is listed but missing")
        elif sha256_file(path) != w["sha256"]:
            problems.append(f"{w['file']} has changed since the manifest was written")
    return problems


def cmd_write(a):
    if tag_exists(a.release):
        raise SystemExit(f"{a.release} is already tagged. Releases are immutable - "
                         f"use a new name.")
    paths = ([os.path.join(WEIGHTS_DIR, f) for f in a.weights] if a.weights else
             sorted(os.path.join(WEIGHTS_DIR, f) for f in os.listdir(WEIGHTS_DIR)
                    if f.endswith(".npz")))
    manifest = build_manifest(a.release, a.notes, paths)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {os.path.relpath(MANIFEST, REPO)}")
    print("next: review it, commit it with the weights, then run "
          "`python scripts/tag_release.py tag`")


def cmd_tag(a):
    if not os.path.exists(MANIFEST):
        raise SystemExit("no manifest - run `tag_release.py write` first")
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    name = manifest.get("release", "")
    if not NAME_RE.match(name):
        raise SystemExit(f"manifest release name {name!r} is invalid")
    if tag_exists(name):
        raise SystemExit(f"{name} is already tagged. Releases are immutable.")

    problems = verify_weights(manifest)
    dirty = git("status", "--porcelain", "--", *DEPENDS_ON).stdout.strip()
    if dirty:
        problems.append("uncommitted changes in files the model depends on:\n    "
                        + dirty.replace("\n", "\n    "))
    if problems:
        raise SystemExit("not tagging:\n  - " + "\n  - ".join(problems))

    body = [name, "", manifest.get("notes") or "", ""]
    body += [f"{w['file']}  sha256 {w['sha256'][:16]}  fitted {w.get('fitted')}  "
             f"n={w.get('n_participants')}" for w in manifest["weights"]]
    r = git("tag", "-a", name, "-m", "\n".join(body))
    if r.returncode != 0:
        raise SystemExit(f"git tag failed: {r.stderr.strip()}")
    commit = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"tagged {name} at {commit}")
    print(f"push it when ready: git push origin {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write outputs/weights/RELEASE.json")
    w.add_argument("--release", required=True, help="e.g. model-2026.09.1")
    w.add_argument("--notes", default="", help="what changed and why this is the release")
    w.add_argument("--weights", nargs="*", help="files in outputs/weights (default: all .npz)")
    sub.add_parser("tag", help="verify the committed manifest and create the git tag")
    a = ap.parse_args(argv)
    (cmd_write if a.cmd == "write" else cmd_tag)(a)


if __name__ == "__main__":
    main()
