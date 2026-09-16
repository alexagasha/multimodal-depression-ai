"""
Clean and de-identify a KoBoToolbox export - every collection wave, in one pass.

Recovered from the script that produced kobo_clean_analytic.xlsx for wave 1. It
lived in a session scratchpad and was never committed, so the "re-run the
cleaning step" instruction in data/kobo/README.md pointed at nothing in the
repo. The transformations are unchanged. What changed is everything that
assumed an export would only ever contain one wave.

HOW OLD AND NEW ROWS ARE TOLD APART
    KoBo's `_uuid` is the only stable key a submission has, and registry.json in
    the sessions directory records every uuid adapt_kobo.py has registered. A
    row whose uuid is in the registry keeps the wave recorded there. A row whose
    uuid is not is new, and gets --new-wave (default "wave2").

    Interview date is checked as corroboration, never used as the rule: a new
    row dated inside wave 1's collection window is reported, because it means
    either a late upload or a submission the registry never saw.

WHAT HAD TO STOP DEPENDING ON ROW ORDER
    participant_id  wave 1 numbered P001-P153 by submission-time rank. Ranking a
                    combined export would renumber wave 1 whenever a row moved.
                    Known uuids now keep the P-id the registry recorded; new
                    uuids continue from the highest, in submission order.
    interviewer     wave 1 coded I1-I5 from sorted account names, so a new
                    account sorting first would recode everyone. Codes are now
                    pinned from wave 1's QC.csv; new accounts get the next ones.

THE GATE
    The wave-1 rows of the combined export must reproduce the existing
    kobo_clean_analytic.xlsx labels sheet exactly - every column, every row.
    Columns are read by position, as they were for wave 1, and every row of one
    export shares one column layout, so a match on the old rows proves the
    mapping for the whole file. On any mismatch nothing is written.

OUTPUTS, both outside the repo
    --out-analytic  de-identified and pipeline-ready, with a `wave` column on the
                    labels sheet. A NEW file: the wave-1 workbook is never
                    overwritten.
    --out-key       re-identification key. SENSITIVE. No default - you choose
                    where the file holding patient names goes - and it is refused
                    anywhere under the kobo data directory or the repo, since the
                    first gets uploaded to Colab and the second gets committed.

Read-only with respect to the raw export, registry.json and QC.csv.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from waves import BASE_WAVE, wave_of  # noqa: E402

KOBO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "kobo"))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- wave-1 export layout --------------------------------------------------
# 0-based column positions in the raw export, exactly as wave 1 was read. They
# are validated by the regression gate rather than trusted.
WAVE1_NCOLS = 134
COL_DATE = 4
COL_FORM_PID = 5              # free text: real names, occupations, colliding numbers
COL_SITE, COL_SITE_OTHER = 6, 7
COL_INTERVIEWER_NAME = 8
COL_AGE, COL_SEX, COL_MARITAL, COL_ETHNIC = 9, 11, 12, 13
COL_RESID, COL_EDU, COL_EMPLOY, COL_PHONE = 15, 16, 17, 18
PHQ = list(range(19, 28))     # 9 items, 0-3
COL_PHQ_TOTAL = 29
HAM = list(range(31, 42))     # 11 items, 0-4
COL_HAM_TOTAL = 43
COL_LANGUAGE = 87
COL_SI_SCREEN = 112
PROBE_SECTION_START = 89
FREETEXT = {113: "si_description", 116: "q16_safe_comfortable", 117: "q17_worries",
            118: "q18_stop_using", 119: "q19_ai_concerns", 120: "q20_features",
            121: "q21_anything_else"}
# KoBo's own metadata columns have stable names, so these are read by name and
# their positions are only reported.
META_COLS = {"_id": 123, "_uuid": 124, "_submission_time": 125, "_submitted_by": 129}

AGE = {"18-25": "18-25", "26-35": "26-35", "36-45": "36-45",
       "46-55": "46-55", "56-65": "56-65"}
SEX = {"female": "female", "male": "male"}
MARITAL = {"married": "married", "divorced": "divorced",
           "widow/er": "widowed", "single": "single"}
ETHNIC = {"bantu": "bantu", "luo": "luo", "other": "other"}
RESID = {"urban": "urban", "semi-urban": "semi-urban", "rural": "rural"}
EDU = {"no formal education": "none", "primary": "primary",
       "secondary": "secondary", "tertiary / university": "tertiary_university"}
EMPLOY = {"employed": "employed", "unemployed": "unemployed",
          "self-employed": "self-employed", "student": "student"}
YESNO = {"yes": "yes", "no": "no"}

LIRA_PLACES = {"ober", "ireda", "ireda shamba", "odokomit", "omito",
               "boroboro", "ayere", "barapwo", "over hc", "masozi"}
# Site choices on the form that name a site directly. Wave 1 only ever used
# Butabika or Other; wave 2 added Mulago.
SITE_CHOICES = {"butabika": "butabika", "mulago": "mulago"}
# tokens that appear in the form's participant-ID field but are not names
NOT_NAMES = {"care", "taker", "giver", "student", "tailor", "farmer", "teacher",
             "worker", "patient", "house", "wife", "housewife", "business", "woman",
             "shop", "attendant", "carpenter", "electrician", "pregnant", "mother",
             "boda", "bodaboda"}

LABEL_COLUMNS = (["participant_id", "wave", "phq9_score", "hamd_score",
                  "caseness_hamd_ge7", "phq9_item9_score", "hamd_suicide_item_score",
                  "si_screen_yes", "referral_flag"]
                 + [f"phq9_item{k}" for k in range(1, 10)]
                 + [f"hamd_item{k}" for k in range(1, 12)]
                 + ["age_band", "sex", "marital_status", "ethnicity", "residence",
                    "education_level", "employment_status", "smartphone",
                    "site", "site_detail_raw", "language_most_comfortable",
                    "interviewer_code", "interview_date"])


def norm(s):
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return None
    return re.sub(r"\s+", " ", str(s)).replace("–", "-").replace("—", "-").strip()


def score(v):
    if pd.isna(v):
        return np.nan
    m = re.search(r"(\d+)", str(v))
    return float(m.group(1)) if m else np.nan


def canon_site(choice, other):
    """Recruitment site from the form's site choice and its "if other" text.

    A named choice maps directly. Wave 1 only used Butabika or Other, and the
    wave-1 rule checked for Butabika alone - so wave 2's new Mulago choice fell
    through to the empty "other" text and came out blank. An unrecognised
    choice is now kept as itself rather than blanked, so a new site shows up
    instead of disappearing.
    """
    c = (norm(choice) or "").lower()
    if c in SITE_CHOICES:
        return SITE_CHOICES[c]
    if c and c != "other":
        return c
    o = (norm(other) or "").lower()
    if "lira" in o or o in LIRA_PLACES:
        return "lira"
    return "other" if o else None


# ---- the wave-aware parts (pure, and tested) --------------------------------
def load_registry(sessions_dir):
    path = os.path.join(sessions_dir, "registry.json")
    if not os.path.exists(path):
        raise SystemExit(f"No registry at {path}. The registry is how old rows are "
                         f"recognised; without it every row would look new.")
    with open(path) as f:
        return json.load(f)


def assign_waves(uuids, registry, new_wave):
    """Registered uuids keep their recorded wave; anything else is new."""
    by = registry["by_uuid"]
    return pd.Series([wave_of(by[u]) if u in by else new_wave for u in uuids],
                     index=uuids.index)


def pin_participant_ids(uuids, submission_time, registry):
    """Known uuids keep their registry P-id; new ones continue after the highest."""
    by = registry["by_uuid"]
    pid = pd.Series([by[u]["sheet_pid"] if u in by and by[u].get("sheet_pid") else None
                     for u in uuids], index=uuids.index, dtype=object)
    taken = [str(e.get("sheet_pid", "")) for e in by.values()] + [str(p) for p in pid.dropna()]
    nums = [int(m.group(1)) for p in taken for m in [re.match(r"P(\d+)$", p)] if m]
    nxt = max(nums, default=0) + 1
    fresh = pid[pid.isna()].index
    for idx in submission_time.loc[fresh].sort_values(kind="stable").index:
        pid[idx] = f"P{nxt:03d}"
        nxt += 1
    dup = pid[pid.duplicated()]
    if len(dup):
        raise ValueError(f"duplicate participant ids: {sorted(set(dup))}")
    return pid


def pin_interviewer_codes(accounts, uuids, qc):
    """account -> interviewer code, fixed to wave 1's codes; new accounts append."""
    known = (qc.dropna(subset=["uuid", "interviewer_code"])
               .set_index("uuid")["interviewer_code"])
    pairs = pd.DataFrame({"account": accounts.values,
                          "code": uuids.map(known).values})
    fixed = {}
    for acct, g in pairs.dropna().groupby("account"):
        codes = set(g.code)
        if len(codes) > 1:
            raise ValueError(f"one account maps to several wave-1 codes: {sorted(codes)}")
        fixed[acct] = codes.pop()
    if len(set(fixed.values())) != len(fixed):
        raise ValueError("two accounts share one wave-1 interviewer code")
    nums = [int(m.group(1)) for c in fixed.values() for m in [re.match(r"I(\d+)$", c)] if m]
    nxt = max(nums, default=0) + 1
    for acct in sorted(set(accounts.dropna()) - set(fixed)):
        fixed[acct] = f"I{nxt}"
        nxt += 1
    return fixed


def date_crosscheck(dates, waves, base_max):
    """Index of new-wave rows dated on or before wave 1's last interview."""
    mask = (waves != BASE_WAVE) & (pd.to_datetime(dates) <= pd.Timestamp(base_max))
    return list(dates.index[mask.values])


def _canon(v):
    """One comparable form per value: a fresh frame and one read back from Excel
    disagree on the type of dates, booleans, nullable ints and missing values."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if hasattr(v, "isoformat"):
        return str(v)[:10]
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    return str(v)


def regression_check(new_labels, prior_labels):
    """Wave-1 rows of the new labels sheet against the wave-1 workbook.

    Returns human-readable mismatches; an empty list means wave 1 reproduced.
    """
    problems = []
    old = new_labels[new_labels.wave == BASE_WAVE].set_index("participant_id")
    prior = prior_labels.set_index("participant_id")
    absent = sorted(set(prior.index) - set(old.index))
    extra = sorted(set(old.index) - set(prior.index))
    if absent:
        problems.append(f"{len(absent)} wave-1 participant(s) absent from this export: "
                        f"{absent[:10]}")
    if extra:
        problems.append(f"{len(extra)} participant(s) marked wave 1 but not in the "
                        f"wave-1 workbook: {extra[:10]}")
    lost = [c for c in prior.columns if c not in old.columns]
    if lost:
        problems.append(f"columns missing from the new labels sheet: {lost}")
    both = prior.index.intersection(old.index)
    for c in (c for c in prior.columns if c in old.columns):
        diff = [p for p in both if _canon(prior.at[p, c]) != _canon(old.at[p, c])]
        if diff:
            p = diff[0]
            problems.append(f"{c}: {len(diff)} row(s) differ, e.g. {p}: "
                            f"was {prior.at[p, c]!r}, now {old.at[p, c]!r}")
    return problems


# ---- cleaning ----------------------------------------------------------------
def clean(raw_path, sessions_dir, new_wave="wave2", prior_labels=None):
    df = pd.read_excel(raw_path)
    C = list(df.columns)
    n = len(df)
    report, problems = [], []

    def log(s):
        report.append(s)
        print(s)

    def col(i):
        return df[C[i]]

    log(f"source rows={n} cols={len(C)}")
    if len(C) != WAVE1_NCOLS:
        log(f"  ! wave 1's export had {WAVE1_NCOLS} columns and this one has {len(C)}; "
            f"positional columns may have moved - the regression gate will say")
    for name, pos in META_COLS.items():
        if name not in df.columns:
            raise SystemExit(f"export has no `{name}` column; KoBo metadata columns present: "
                             f"{[c for c in C if str(c).startswith('_')]}")
        if C.index(name) != pos:
            log(f"  ! `{name}` is at column {C.index(name)} (was {pos} in wave 1)")

    uuids = df["_uuid"].astype(str).str.strip()
    if uuids.duplicated().any():
        raise SystemExit(f"duplicate _uuid in export: {sorted(set(uuids[uuids.duplicated()]))[:5]}")

    registry = load_registry(sessions_dir)
    qc = pd.read_csv(os.path.join(sessions_dir, "QC.csv"))

    # 1. waves - the whole point of this version of the script
    wave = assign_waves(uuids, registry, new_wave)
    log(f"waves (by _uuid against registry.json): {wave.value_counts().to_dict()}")
    unseen = sorted(u for u in set(registry["by_uuid"]) - set(uuids)
                    if not u.startswith("NOAUDIO::"))
    if unseen:
        problems.append(f"{len(unseen)} registered wave-1 submission(s) are not in this "
                        f"export - it would silently shrink wave 1")

    # 2. participant ids, pinned
    df["participant_id"] = pin_participant_ids(
        uuids, pd.to_datetime(df["_submission_time"]), registry)
    fresh = df.loc[wave != BASE_WAVE, "participant_id"]
    if len(fresh):
        log(f"new participant ids: {fresh.min()}..{fresh.max()} ({len(fresh)})")

    # 3. interviewer codes, pinned
    acct_map = pin_interviewer_codes(df["_submitted_by"], uuids, qc)
    df["interviewer_code"] = df["_submitted_by"].map(acct_map)
    log(f"interviewers coded: {len(acct_map)} accounts -> {sorted(set(acct_map.values()))}")
    new_codes = sorted(set(df.loc[wave != BASE_WAVE, "interviewer_code"].dropna())
                       - set(qc.interviewer_code.dropna()))
    if new_codes:
        log(f"  new interviewer code(s) this wave: {new_codes}")

    # 4. item-level scores
    out = pd.DataFrame(index=df.index)
    out["participant_id"] = df["participant_id"]
    out["wave"] = wave
    for k, c in enumerate(PHQ, start=1):
        out[f"phq9_item{k}"] = col(c).map(score).astype("Int64")
    for k, c in enumerate(HAM, start=1):
        out[f"hamd_item{k}"] = col(c).map(score).astype("Int64")
    out["phq9_score"] = out[[f"phq9_item{k}" for k in range(1, 10)]].sum(axis=1)
    out["hamd_score"] = out[[f"hamd_item{k}" for k in range(1, 12)]].sum(axis=1)

    # recomputed totals must equal the form's own computed totals
    for name, total_col in (("phq9_score", COL_PHQ_TOTAL), ("hamd_score", COL_HAM_TOTAL)):
        ok = out[name].astype(float).eq(pd.to_numeric(col(total_col), errors="coerce"))
        if not ok.all():
            bad = sorted(out.loc[~ok, "participant_id"])
            raise SystemExit(f"{name} does not match the form's computed total for "
                             f"{len(bad)} row(s): {bad[:10]} - incomplete items, or the "
                             f"item columns have moved")
    log(f"totals verified. phq9 max={out['phq9_score'].max()} (of 27), "
        f"hamd max={out['hamd_score'].max()} (of 44)")

    # referral-flag inputs (rule-based, never model-derived)
    out["phq9_item9_score"] = out["phq9_item9"]
    out["hamd_suicide_item_score"] = out["hamd_item3"]
    si = col(COL_SI_SCREEN).astype(str).str.strip().str.lower().eq("yes")
    out["si_screen_yes"] = si
    out["referral_flag"] = (out["phq9_item9_score"] > 0) | (out["hamd_suicide_item_score"] > 0) | si
    log(f"referral_flag positives: {int(out['referral_flag'].sum())}/{n}")

    out["caseness_hamd_ge7"] = out["hamd_score"] >= 7
    log(f"caseness (HAM-D>=7): {int(out['caseness_hamd_ge7'].sum())} pos / "
        f"{int((~out['caseness_hamd_ge7']).sum())} neg  "
        f"({out['caseness_hamd_ge7'].mean():.1%} positive)")

    # 5. demographics -> the exact vocabulary of metadata_pipeline.py. The export
    #    writes age bands with an EN DASH; unnormalised, everyone encodes unknown.
    def mapcol(i, table, name):
        raw = col(i).map(norm)
        mapped = raw.str.lower().map(table)
        unmapped = sorted(set(raw[mapped.isna() & raw.notna()]))
        if unmapped:
            log(f"  ! {name}: {len(unmapped)} unmapped value(s) -> {unmapped} "
                f"({int(mapped.isna().sum())} row(s) left blank)")
        return mapped

    log("demographic normalisation:")
    out["age_band"] = mapcol(COL_AGE, AGE, "age_band")
    out["sex"] = mapcol(COL_SEX, SEX, "sex")
    out["marital_status"] = mapcol(COL_MARITAL, MARITAL, "marital_status")
    out["ethnicity"] = mapcol(COL_ETHNIC, ETHNIC, "ethnicity")
    out["residence"] = mapcol(COL_RESID, RESID, "residence")
    out["education_level"] = mapcol(COL_EDU, EDU, "education_level")
    out["employment_status"] = mapcol(COL_EMPLOY, EMPLOY, "employment_status")
    out["smartphone"] = mapcol(COL_PHONE, YESNO, "smartphone")

    # 6. site: collapse free-text spellings into recruitment sites (provenance only)
    choices = set(col(COL_SITE).map(norm).dropna().str.lower())
    unknown_choices = sorted(choices - set(SITE_CHOICES) - {"other"})
    if unknown_choices:
        log(f"  ! site choice(s) not seen before, kept as-is: {unknown_choices}")
    out["site"] = [canon_site(ch, ot) for ch, ot in zip(col(COL_SITE), col(COL_SITE_OTHER))]
    out["site_detail_raw"] = df.apply(
        lambda r: (norm(r[C[COL_SITE_OTHER]]) or "").lower() or None, axis=1)
    log(f"site: {out['site'].value_counts().to_dict()}")

    out["language_most_comfortable"] = col(COL_LANGUAGE).map(norm)
    out["interviewer_code"] = df["interviewer_code"]
    out["interview_date"] = pd.to_datetime(col(COL_DATE)).dt.date

    # corroborate the uuid rule with dates
    base_max = (pd.to_datetime(prior_labels.interview_date).max() if prior_labels is not None
                else pd.to_datetime(out.loc[wave == BASE_WAVE, "interview_date"]).max())
    if pd.notna(base_max) and (wave != BASE_WAVE).any():
        early = date_crosscheck(out["interview_date"], wave, base_max)
        if early:
            log(f"  ! {len(early)} {new_wave} row(s) dated on or before wave 1's last "
                f"interview ({base_max.date()}): {sorted(out.loc[early, 'participant_id'])[:10]}")
        else:
            log(f"date check: every {new_wave} interview is after {base_max.date()}")

    # 7. audio filename map, one column per prompt
    audio_cols = [i for i in range(len(C)) if col(i).astype(str).str.endswith(".m4a").any()]
    audio = pd.DataFrame(index=df.index)
    audio["participant_id"] = df["participant_id"]
    for k, i in enumerate(audio_cols, start=1):
        audio[f"q{k:02d}_file"] = col(i)
        audio[f"q{k:02d}_prompt"] = C[i]
    prompts = pd.DataFrame({
        "prompt_index": range(1, len(audio_cols) + 1),
        "column_name": [C[i] for i in audio_cols],
        "section": ["interview" if i < PROBE_SECTION_START else "clinical_probe"
                    for i in audio_cols],
    })
    log(f"audio: {len(audio_cols)} prompts x {n} participants = {len(audio_cols) * n} clips")

    # 8. free text -> its own sheet, with a name-leak scan
    ft = pd.DataFrame({"participant_id": df["participant_id"]})
    for i, name in FREETEXT.items():
        ft[name] = col(i).map(norm)
    known_names = {tok.lower() for c in (COL_FORM_PID, COL_INTERVIEWER_NAME)
                   for v in col(c).dropna().astype(str)
                   for tok in re.findall(r"[A-Za-z]{3,}", v)} - NOT_NAMES
    hits = []
    for name in FREETEXT.values():
        for idx, v in ft[name].dropna().items():
            overlap = {t.lower() for t in re.findall(r"[A-Za-z]{3,}", str(v))} & known_names
            if overlap:
                hits.append((ft.at[idx, "participant_id"], name))
    log(f"free-text name-leak scan: {len(hits)} cell(s) contain a token matching a "
        f"known participant/interviewer name")
    for pid_, field in hits[:15]:
        log(f"    {pid_} {field}")

    # 9. re-identification key (SENSITIVE)
    key = pd.DataFrame({
        "participant_id": df["participant_id"],
        "wave": wave,
        "kobo_uuid": uuids,
        "kobo_id": df["_id"],
        "form_participant_id_raw": col(COL_FORM_PID),
        "interviewer_name_raw": col(COL_INTERVIEWER_NAME),
        "interviewer_code": df["interviewer_code"],
        "submitted_by": df["_submitted_by"],
        "submission_time": df["_submission_time"],
    })

    return {"labels": out[LABEL_COLUMNS], "audio_map": audio, "prompts": prompts,
            "free_text": ft, "key": key, "report": report, "problems": problems,
            "known_names": known_names}


def _inside(path, root):
    p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:          # different drives
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean and de-identify a multi-wave KoBo export.")
    ap.add_argument("--raw", required=True,
                    help="KoBo XLSX export (all versions, English labels) holding old and new rows")
    ap.add_argument("--sessions", default=os.path.join(KOBO, "sessions"),
                    help="adapter output holding registry.json and QC.csv")
    ap.add_argument("--prior", default=os.path.join(KOBO, "kobo_clean_analytic.xlsx"),
                    help="the wave-1 cleaned workbook the old rows must reproduce")
    ap.add_argument("--out-analytic", dest="out_analytic",
                    default=os.path.join(KOBO, "kobo_clean_analytic_combined.xlsx"))
    ap.add_argument("--out-key", dest="out_key", required=True,
                    help="where the SENSITIVE re-identification key goes; refused inside "
                         "the kobo data directory or the repo")
    ap.add_argument("--new-wave", dest="new_wave", default="wave2")
    ap.add_argument("--allow-no-prior", dest="allow_no_prior", action="store_true",
                    help="skip the wave-1 regression gate - only if that workbook is gone")
    a = ap.parse_args(argv)

    for guarded, why in ((KOBO, "the kobo data directory is uploaded to Colab"),
                         (REPO, "the repo is committed")):
        if _inside(a.out_key, guarded):
            raise SystemExit(f"refusing to write the re-identification key under {guarded}: {why}")
    if _inside(a.out_analytic, REPO):
        raise SystemExit("refusing to write study data inside the repo")
    if os.path.normcase(os.path.abspath(a.out_analytic)) == os.path.normcase(os.path.abspath(a.prior)):
        raise SystemExit("refusing to overwrite the wave-1 workbook; write the combined file elsewhere")
    if a.new_wave == BASE_WAVE:
        raise SystemExit(f"--new-wave must differ from {BASE_WAVE}")

    prior = None
    if os.path.exists(a.prior):
        prior = pd.read_excel(a.prior, sheet_name="labels")
    elif not a.allow_no_prior:
        raise SystemExit(f"wave-1 workbook not found at {a.prior}; the regression gate needs "
                         f"it (--allow-no-prior skips the gate)")

    res = clean(a.raw, a.sessions, a.new_wave, prior)
    problems = list(res["problems"])
    if prior is not None:
        mism = regression_check(res["labels"], prior)
        problems += mism
        if not mism:
            n_old = int((res["labels"].wave == BASE_WAVE).sum())
            line = (f"regression gate: all {n_old} wave-1 rows reproduce "
                    f"{os.path.basename(a.prior)} exactly across {prior.shape[1]} columns")
            res["report"].append(line)
            print(line)

    if problems:
        print("\nNOTHING WRITTEN - this export does not reproduce wave 1:")
        for p in problems:
            print(f"  - {p}")
        return 1

    with pd.ExcelWriter(a.out_analytic, engine="openpyxl") as w:
        res["labels"].to_excel(w, sheet_name="labels", index=False)
        res["audio_map"].to_excel(w, sheet_name="audio_map", index=False)
        res["prompts"].to_excel(w, sheet_name="prompts", index=False)
        res["free_text"].to_excel(w, sheet_name="free_text", index=False)
        pd.DataFrame({"line": res["report"]}).to_excel(w, sheet_name="clean_log", index=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_key)), exist_ok=True)
    with pd.ExcelWriter(a.out_key, engine="openpyxl") as w:
        res["key"].to_excel(w, sheet_name="KEY_DO_NOT_SHARE", index=False)
    print(f"\nwrote {a.out_analytic}  labels={res['labels'].shape}")
    print(f"wrote {a.out_key}  (SENSITIVE)")

    lab = res["labels"]
    leaky = [(c, v) for c in lab.columns for v in lab[c].dropna().astype(str).unique()
             if {t.lower() for t in re.findall(r"[A-Za-z]{3,}", v)} & res["known_names"]]
    print("analytic-sheet identifier leak check:",
          "CLEAN" if not leaky else f"{len(leaky)} suspect cell value(s) - inspect before use")

    print("\nper wave:")
    for w_, g in lab.groupby("wave"):
        d = pd.to_datetime(g.interview_date)
        print(f"  {w_:<6} n={len(g):>3} | cases {int(g.caseness_hamd_ge7.sum())} / non-cases "
              f"{int((~g.caseness_hamd_ge7).sum())} | referral {int(g.referral_flag.sum())} | "
              f"{d.min().date()}..{d.max().date()}")
        print(f"         sites {g.site.value_counts().to_dict()} | "
              f"interviewers {g.interviewer_code.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
