"""
Collection waves: a held-out wave must stay out of training at every step.

A later wave is only worth anything as an unseen sample. The ways it stops
being one are all silent - a split reassigned, a cross-validation that reads
every row it is given, a participant renumbered - so each is pinned here.

No clinical data: every fixture is synthetic.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "data", "kobo"))

import adapt_kobo  # noqa: E402
import clean_kobo  # noqa: E402
import validate_kobo  # noqa: E402
import waves  # noqa: E402
from src.eval import cohort  # noqa: E402
from src.eval.benchmark import load_cache  # noqa: E402


def _registry(entries):
    return {"by_uuid": entries, "next_pid": 1001 + len(entries)}


def test_training_splits_agree_between_the_kobo_scripts_and_analysis():
    assert set(waves.TRAINING_SPLITS) == set(cohort.TRAINING_SPLITS)


# ---------------------------------------------------------------- adapter
def test_registry_written_before_waves_existed_is_wave1(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(_registry({"u1": {"pid": 1001, "sheet_pid": "P001",
                                               "split": "train"}})))
    reg = adapt_kobo._load_registry(str(p))
    assert reg["by_uuid"]["u1"]["wave"] == "wave1"


def _mixed_registry():
    entries = {}
    for i, s in enumerate(["train"] * 6 + ["dev"] * 2 + ["test"] * 2):
        entries[f"old{i}"] = {"pid": 1001 + i, "sheet_pid": f"P{i + 1:03d}",
                              "split": s, "wave": "wave1"}
    for i in range(6):
        entries[f"new{i}"] = {"pid": 1011 + i, "sheet_pid": f"P{11 + i:03d}",
                              "split": None, "wave": "wave2"}
    for i in range(4):   # wave-1 participants who only now became usable
        entries[f"late{i}"] = {"pid": 1017 + i, "sheet_pid": f"P{17 + i:03d}",
                               "split": None, "wave": "wave1"}
    return _registry(entries)


def _run_finalise(tmp_path, reg, reassign=False):
    uuids = list(reg["by_uuid"])
    qc = pd.DataFrame({
        "participant_id": [reg["by_uuid"][u]["pid"] for u in uuids],
        "sheet_pid": [reg["by_uuid"][u]["sheet_pid"] for u in uuids],
        "uuid": uuids, "reason": "", "clips": 29, "speech_sec": 200.0,
        "audio_sec": 300.0, "interviewer_code": "I1",
        "caseness": [i % 4 != 0 for i in range(len(uuids))],
    })
    labels = pd.DataFrame({"participant_id": qc.sheet_pid, "phq9_score": 10,
                           "hamd_score": 12, "phq9_item9_score": 0,
                           "hamd_suicide_item_score": 0})
    return adapt_kobo._finalise(str(tmp_path), qc, reg, str(tmp_path / "registry.json"),
                                labels, 30.0, 1.0, 1337, (0.70, 0.15, 0.15),
                                reassign=reassign)


def test_new_wave_is_held_out_and_no_wave1_split_moves(tmp_path):
    reg = _mixed_registry()
    before = {u: e["split"] for u, e in reg["by_uuid"].items() if u.startswith("old")}

    out = _run_finalise(tmp_path, reg)

    for u, s in before.items():
        assert reg["by_uuid"][u]["split"] == s, f"wave-1 split moved for {u}"
    for i in range(6):
        assert reg["by_uuid"][f"new{i}"]["split"] == "wave2"
    for i in range(4):
        assert reg["by_uuid"][f"late{i}"]["split"] in waves.TRAINING_SPLITS

    lab = pd.read_csv(tmp_path / "LABELS.csv")
    assert list(lab.columns) == validate_kobo.LABEL_COLS
    assert not lab[lab.wave == "wave2"].split.isin(waves.TRAINING_SPLITS).any()
    assert validate_kobo.wave_split_violations(lab) == []
    assert set(out.wave) == {"wave1", "wave2"}


def test_reassigning_splits_still_cannot_pull_a_held_out_wave_into_training(tmp_path):
    reg = _mixed_registry()
    _run_finalise(tmp_path, reg, reassign=True)
    assert all(reg["by_uuid"][f"new{i}"]["split"] == "wave2" for i in range(6))


def test_validator_names_both_directions_of_leakage():
    lab = pd.DataFrame({"participant_id": [1, 2, 3, 4],
                        "split": ["train", "wave2", "train", "wave2"],
                        "wave": ["wave1", "wave2", "wave2", "wave1"]})
    bad = validate_kobo.wave_split_violations(lab)
    assert (3, "wave2", "train") in bad      # held-out participant in training
    assert (4, "wave1", "wave2") in bad      # wave-1 participant given a hold-out
    assert len(bad) == 2


def test_wave_in_metadata_does_not_change_the_feature_vector():
    from src.pipelines.metadata_pipeline import encode_metadata
    meta = {"age_band": "26-35", "sex": "female", "marital_status": "single",
            "ethnicity": "bantu", "residence": "rural",
            "education_level": "tertiary_university", "employment_status": "student",
            "smartphone": "yes", "site": "lira"}
    assert np.array_equal(encode_metadata(meta), encode_metadata({**meta, "wave": "wave2"}))


# ---------------------------------------------------------------- analysis
def _cache(tmp_path, pids, splits=None):
    n = len(pids)
    kw = {"X": np.arange(n * 3, dtype=float).reshape(n, 3),
          "Y": np.column_stack([np.arange(n), np.arange(n)]).astype(float),
          "pids": np.array(pids)}
    if splits is not None:
        kw["splits"] = np.array(splits)
    path = tmp_path / "cache.npz"
    np.savez(path, **kw)
    return str(path)


def _labels(tmp_path, rows):
    path = tmp_path / "LABELS.csv"
    pd.DataFrame(rows, columns=["participant_id", "split"]).to_csv(path, index=False)
    return str(path)


def test_held_out_wave_is_excluded_from_analysis_by_default(tmp_path):
    cache = _cache(tmp_path, [1001, 1002, 1154, 1003])
    labels = _labels(tmp_path, [(1001, "train"), (1002, "dev"), (1154, "wave2"), (1003, "test")])
    X, Y, pids = load_cache(cache, labels=labels)
    assert list(pids) == [1001, 1002, 1003]
    assert X.shape == (3, 3) and Y.shape == (3, 2)
    assert X[2, 0] == 9.0, "rows must stay aligned with their participant after filtering"


def test_including_the_held_out_wave_is_an_explicit_choice(tmp_path):
    cache = _cache(tmp_path, [1001, 1154])
    labels = _labels(tmp_path, [(1001, "train"), (1154, "wave2")])
    _, _, pids, info = load_cache(cache, labels=labels, include_holdout=True, return_info=True)
    assert list(pids) == [1001, 1154]
    assert info["include_holdout"] is True and info["held_out"] == {"wave2": 1}


def test_participant_with_no_known_split_is_excluded(tmp_path):
    cache = _cache(tmp_path, [1001, 1200])
    labels = _labels(tmp_path, [(1001, "train")])
    _, _, pids, info = load_cache(cache, labels=labels, return_info=True)
    assert list(pids) == [1001] and info["unknown_split"] == 1


def test_labels_file_overrides_stale_splits_stored_in_a_cache(tmp_path):
    cache = _cache(tmp_path, [1001, 1154], splits=["train", "train"])
    labels = _labels(tmp_path, [(1001, "train"), (1154, "wave2")])
    _, _, pids = load_cache(cache, labels=labels)
    assert list(pids) == [1001]


def test_cache_splits_are_honoured_when_no_labels_are_given(tmp_path):
    cache = _cache(tmp_path, [1001, 1154], splits=["train", "wave2"])
    _, _, pids = load_cache(cache)
    assert list(pids) == [1001]


def test_a_missing_labels_file_is_refused_not_guessed(tmp_path):
    with pytest.raises(SystemExit):
        cohort.resolve_labels(str(tmp_path / "nope.csv"))
    assert cohort.resolve_labels("none") is None


# ---------------------------------------------------------------- cleaner
def test_known_submissions_keep_their_participant_id_whatever_the_row_order():
    reg = _registry({"a": {"pid": 1001, "sheet_pid": "P001"},
                     "b": {"pid": 1002, "sheet_pid": "P002"}})
    uuids = pd.Series(["new2", "b", "new1", "a"])
    t = pd.Series(pd.to_datetime(["2026-09-02", "2026-07-29", "2026-09-01", "2026-07-28"]))
    assert list(clean_kobo.pin_participant_ids(uuids, t, reg)) == ["P004", "P002", "P003", "P001"]


def test_waves_come_from_the_registry_not_from_row_position():
    reg = _registry({"a": {"pid": 1001, "sheet_pid": "P001"},
                     "b": {"pid": 1002, "sheet_pid": "P002", "wave": "wave1"}})
    w = clean_kobo.assign_waves(pd.Series(["x", "a", "b"]), reg, "wave2")
    assert list(w) == ["wave2", "wave1", "wave1"]


def test_interviewer_codes_are_pinned_and_a_new_account_appends():
    qc = pd.DataFrame({"uuid": ["a", "b", "c"], "interviewer_code": ["I1", "I2", "I1"]})
    accounts = pd.Series(["zed", "amy", "zed", "aaron", "amy"])
    uuids = pd.Series(["a", "b", "c", "n1", "n2"])
    # "aaron" sorts first, so the old rule would have made it I1 and recoded everyone
    assert clean_kobo.pin_interviewer_codes(accounts, uuids, qc) == {
        "zed": "I1", "amy": "I2", "aaron": "I3"}


def test_one_account_with_two_wave1_codes_is_refused():
    qc = pd.DataFrame({"uuid": ["a", "b"], "interviewer_code": ["I1", "I2"]})
    with pytest.raises(ValueError):
        clean_kobo.pin_interviewer_codes(pd.Series(["same", "same"]), pd.Series(["a", "b"]), qc)


def test_new_rows_dated_inside_wave1_are_reported():
    dates = pd.Series(pd.to_datetime(["2026-08-01", "2026-09-03", "2026-08-04"]).date)
    w = pd.Series(["wave1", "wave2", "wave2"])
    assert clean_kobo.date_crosscheck(dates, w, "2026-08-05") == [2]


def test_regression_gate_catches_a_changed_wave1_value():
    prior = pd.DataFrame({"participant_id": ["P001", "P002"], "hamd_score": [10, 3],
                          "interview_date": pd.to_datetime(["2026-07-28", "2026-07-29"])})
    new = pd.DataFrame({"participant_id": ["P001", "P002", "P003"], "hamd_score": [10, 4, 20],
                        "interview_date": pd.to_datetime(
                            ["2026-07-28", "2026-07-29", "2026-09-01"]).date,
                        "wave": ["wave1", "wave1", "wave2"]})
    problems = clean_kobo.regression_check(new, prior)
    assert len(problems) == 1 and problems[0].startswith("hamd_score")

    new.loc[1, "hamd_score"] = 3
    assert clean_kobo.regression_check(new, prior) == []


def test_regression_gate_catches_a_missing_wave1_participant():
    prior = pd.DataFrame({"participant_id": ["P001", "P002"], "hamd_score": [10, 3]})
    new = pd.DataFrame({"participant_id": ["P001", "P003"], "hamd_score": [10, 20],
                        "wave": ["wave1", "wave2"]})
    assert any("absent" in p for p in clean_kobo.regression_check(new, prior))


def test_the_key_file_is_refused_inside_the_repo(tmp_path):
    with pytest.raises(SystemExit):
        clean_kobo.main(["--raw", "x.xlsx", "--out-key", os.path.join(REPO, "key.xlsx"),
                         "--prior", str(tmp_path / "p.xlsx"), "--allow-no-prior"])


def test_rerunning_with_nothing_new_changes_nothing(tmp_path):
    """The normal re-run: every usable participant already has a split. No
    newcomers must neither crash nor move anything. (It crashed: Series.map over
    an empty frame yields object dtype, which pandas reads as column labels.)"""
    entries = {f"old{i}": {"pid": 1001 + i, "sheet_pid": f"P{i + 1:03d}",
                           "split": s, "wave": "wave1"}
               for i, s in enumerate(["train"] * 6 + ["dev"] * 2 + ["test"] * 2)}
    reg = _registry(entries)
    before = {u: e["split"] for u, e in entries.items()}
    _run_finalise(tmp_path, reg)
    assert {u: e["split"] for u, e in reg["by_uuid"].items()} == before


def test_site_choices_map_directly_and_a_new_choice_is_not_blanked():
    """Wave 2 added Mulago as a form choice. The wave-1 rule only recognised
    Butabika, so Mulago fell through to the empty "other" text and came out
    blank for 9 participants."""
    site = clean_kobo.canon_site
    assert site("Mulago", None) == "mulago"
    assert site("Butabika", float("nan")) == "butabika"
    assert site("Some New Hospital", None) == "some new hospital"
    # wave-1 behaviour is unchanged
    assert site("Other", "Lira University") == "lira"
    assert site("Other", "Ober") == "lira"
    assert site("Other", "Namataba") == "other"
    assert site("Other", None) is None
