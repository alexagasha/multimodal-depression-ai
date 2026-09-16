"""
Pre-handover check for a field machine.

Loads the API exactly as uvicorn would, and reports whether it will produce
scores. Exits 0 only when scoring is ready WITHOUT the development override:
DEP_ALLOW_UNVALIDATED is cleared before anything is checked, because a pass
under the override would prove nothing.

    python scripts/preflight.py

Run it on the machine that is going to the field, in the environment it will
run in (same Python, same environment variables), and do not hand it over on a
failure.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
os.environ.pop("DEP_ALLOW_UNVALIDATED", None)

import api.main as api_main  # noqa: E402  (loads encoders, weights and transcriber)

report = api_main._readiness()
release = api_main._RELEASE

print("\n" + "=" * 66)
print("PREFLIGHT")
print("=" * 66)
for c in report["checks"]:
    mark = "PASS" if c["ok"] else ("FAIL" if c["blocking"] else "WARN")
    print(f"  {mark}  {c['name']:<21} {c['detail']}")
print("-" * 66)
print(f"weights : {release['weights_path']}")
print(f"sha256  : {release['sha256'] or '-'}")
print(f"release : {release['release'] or 'none'}")
print("=" * 66)
if report["scoring_ready"]:
    print(f"READY - this installation will score with release {release['release']}.")
    sys.exit(0)
print("NOT READY - scoring will be refused. Fix the failures above before handover.")
sys.exit(1)
