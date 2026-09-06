"""
Monthly mental-health roll-up in the shape of Uganda's HMIS 105.

HMIS 105 is the monthly facility summation form. Its mental-health section was
widened after January 2020 to cover more mental, neurological and substance-use
conditions, and it is what Butabika and Lira already compile and submit. A
screening tool whose output cannot roll up into it does not replace any
paperwork — it adds a second system to maintain
(docs/clinical-documentation-plan.md 1.6).

TWO THINGS THIS MODULE IS DELIBERATE ABOUT.

Counts come from the CLINICIAN'S recorded PHQ-9 and HAM-D, never from the
model's estimate. National statistics must not be assembled from an
unvalidated prediction, and the model here is a screening adjunct evaluated on
153 participants. `model_estimated_cases` is reported alongside as a separate,
clearly-labelled figure so the two can be compared — that comparison is
useful — but the reportable number is the clinician's.

Only the depression row is populated. This system screens for depression; it
does not detect bipolar disorder, psychosis, epilepsy or substance use, and
emitting zeros for those rows would read as "none seen this month" rather than
"not assessed". They are omitted, and `not_assessed` says why.

The exact row labels and age bandings on the current HMIS 105 must be checked
against the form in use at the facility before anything here is transcribed
into a submission. This produces the counts, not the form.
"""
from collections import defaultdict

# HAM-D caseness for the study instrument (11 items, 0-44). Mirrors
# src/fusion/model.py's threshold rather than redefining it, so a change to
# the clinical definition cannot drift between scoring and reporting.
from src.fusion.model import HAMD_CASENESS_THRESHOLD

# PHQ-9 >= 10 is the conventional cut for clinically significant depression
# and is what most facility reporting uses.
PHQ9_CASE_THRESHOLD = 10

NOT_ASSESSED = [
    "bipolar_disorder",
    "schizophrenia",
    "epilepsy",
    "dementia",
    "anxiety_disorders",
    "alcohol_use_disorder",
    "drug_use_disorder",
    "childhood_mental_disorders",
]


def _month_of(iso_timestamp: str) -> str:
    """YYYY-MM, or "" if the timestamp is missing or malformed."""
    return (iso_timestamp or "")[:7]


def build_hmis105_mental_health(month, sessions, participants, scale_for, prediction_for) -> dict:
    """Depression counts for one calendar month, disaggregated by sex and age.

    `scale_for` and `prediction_for` are callables taking a session_id, so the
    caller owns storage access and this stays testable without a store.
    """
    by_id = {p["participant_id"]: p for p in participants}

    total = 0
    cases = 0
    model_cases = 0
    referrals = 0
    by_sex = defaultdict(int)
    by_age = defaultdict(int)
    # A patient seen more than once in the month is one person in the
    # denominator but several visits; both numbers are reported because the
    # form asks about attendances, not individuals.
    people = set()

    for s in sessions:
        if _month_of(s.get("created_at", "")) != month:
            continue
        scale = scale_for(s["session_id"])
        if scale is None:
            continue  # no clinician scores recorded, so nothing reportable

        total += 1
        people.add(s["participant_id"])
        if s.get("risk_flag"):
            referrals += 1

        is_case = (
            scale["phq9_total"] >= PHQ9_CASE_THRESHOLD
            or scale["hamd_total"] >= HAMD_CASENESS_THRESHOLD
        )
        if is_case:
            cases += 1
            participant = by_id.get(s["participant_id"], {})
            by_sex[participant.get("sex", "unknown")] += 1
            by_age[participant.get("age_band", "unknown")] += 1

        prediction = prediction_for(s["session_id"])
        if prediction and prediction.get("binary_pred"):
            model_cases += 1

    return {
        "month": month,
        "form": "HMIS 105 — mental health section (depression row only)",
        "source": "clinician-recorded PHQ-9 and HAM-D",
        "attendances_screened": total,
        "individuals_screened": len(people),
        "depression_cases": cases,
        "depression_cases_by_sex": dict(by_sex),
        "depression_cases_by_age_band": dict(by_age),
        "referrals_for_risk": referrals,
        "case_definition": (
            f"PHQ-9 >= {PHQ9_CASE_THRESHOLD} or HAM-D >= {HAMD_CASENESS_THRESHOLD} "
            f"as scored by the clinician"
        ),
        # Reported separately and never substituted for the line above.
        "model_estimated_cases": model_cases,
        "not_assessed": NOT_ASSESSED,
        "caveat": (
            "Depression only. This system does not assess the other HMIS 105 mental-health "
            "conditions, which are omitted rather than reported as zero. Check row labels and "
            "age bandings against the form in use before transcribing."
        ),
    }
