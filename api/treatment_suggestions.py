"""
Guideline-grounded next-step suggestions: paraphrased, non-proprietary
clinical heuristics, NOT a retrieval system over the actual copyrighted
guidelines. Same inlined-static-context approach as
api/subtype_differential.py's DSM-5 criteria, scoped to one feature rather
than full RAG infrastructure — see docs/system-roadmap.md Phase 5 for where
this graduates to real retrieval if the corpus grows.

ORDERED TO MATCH THE SETTING, NOT THE LITERATURE'S CENTRE OF GRAVITY.

These heuristics used to lead with STAR*D-style stepped care: switching within
and across antidepressant class, augmentation strategies, TCA-versus-SSRI
subtleties. That is US tertiary-care sequencing, and it is the wrong first
frame for Butabika and Lira, where staff are trained against WHO's mhGAP
Intervention Guide and the Uganda Clinical Guidelines
(docs/clinical-documentation-plan.md 1.6). mhGAP puts psychoeducation,
addressing current psychosocial stressors, reactivating social supports and
structured activity ahead of medication for milder presentations, and works
from a much narrower formulary than the literature assumes.

The subtype-specific considerations are kept, moved below the mhGAP-aligned
sequence, because they are still useful to a specialist reading them — they
just should not be the first thing suggested to a clinical officer in a
district facility.

Always advisory, always phrased as "consider"/"may warrant" — never a
directive or prescription. Explanatory only: this module NEVER feeds back
into phq9_pred, hamd_pred, binary_pred, or risk_flag.
"""
import json

from api.llm_utils import call_claude

TREATMENT_HEURISTICS = """
FIRST-LINE, IN THE ORDER A DISTRICT-LEVEL SERVICE WOULD WORK THROUGH THEM \
(mhGAP-IG depression module / Uganda Clinical Guidelines):
- Psychoeducation for the patient, and where they consent, for a family member: that \
depression is a common and treatable condition, not a personal failing or a spiritual one.
- Identify and address current psychosocial stressors — housing, income, bereavement, \
violence, caregiving burden, stigma. These are often the most modifiable thing available.
- Reactivate social supports and previously enjoyed activities; structured physical \
activity and a regular sleep routine.
- Mild presentations: the above alone is an adequate first step. Antidepressants are not \
routinely indicated as the opening move, and starting one before psychosocial \
intervention has been tried is a common over-treatment.
- Moderate-to-severe presentations, or inadequate response to the above: consider an \
antidepressant, working within the facility's actual formulary — in Uganda that is \
typically fluoxetine or amitriptyline rather than the wider range the literature assumes.
- Review at 4-6 weeks at an adequate dose before concluding a trial has failed, and \
follow up more often than that early on.
- Screen for a history of mania before starting an antidepressant; also consider \
concurrent alcohol or substance use, and treatable physical causes (anaemia, thyroid \
disease, HIV, medication side effects).
- Refer to specialist or district mental health services for psychotic features, high \
suicide risk, pregnancy or breastfeeding, adolescents, or non-response after adequate \
trials.

FURTHER SUBTYPE CONSIDERATIONS, where a specialist is involved:
- First depressive episode, mild-moderate severity: first-line SSRI or structured \
psychotherapy (CBT/IPT) are both reasonable; patient preference matters.
- Melancholic features present: pharmacotherapy tends to be favored over psychotherapy \
alone; consider earlier specialist review if severe.
- Atypical features present: some evidence favors certain SSRIs over TCAs; consider \
addressing hypersomnia/appetite change directly (sleep hygiene, activity scheduling).
- Anxious distress present: an agent effective for both mood and anxiety may be preferable; \
short-term augmentation is sometimes considered, weighed against dependence risk.
- Psychotic features present: antipsychotic augmentation or ECT referral is typically \
indicated, not antidepressant monotherapy alone — urgent specialist review.
- Inadequate response after an adequate trial (usually 4-6+ weeks at an adequate dose): \
consider dose optimization, switching within/across class, or augmentation before further \
monotherapy trials.
- Active referral flag (suicide/self-harm item): safety planning and a closer follow-up \
interval take priority over any medication change.
"""

PROMPT_TEMPLATE = """You are assisting a psychiatrist with next-step treatment considerations. \
This is advisory decision support only — never phrase anything as a directive or prescription; \
always use "consider" / "may warrant" / "worth discussing."

General heuristics to draw from (not a substitute for full clinical guidelines). The setting \
is a Ugandan mental health service, so prefer the first-line sequence over the specialist \
considerations unless this patient's data calls for the latter:
{heuristics}

This patient's screening data:
- PHQ-9 (AI estimate / clinician-scored): {phq9_pred:.1f} / {phq9_clinician}
- HAM-D (AI estimate / clinician-scored): {hamd_pred:.1f} / {hamd_clinician}
- Meets caseness threshold: {binary_pred}
- Referral flag active: {risk_flag}
- Subtype differential: {subtype_summary}

Suggest 2-3 short, advisory next-step considerations grounded in the heuristics above and this \
patient's data. Respond with ONLY a JSON object: {{"suggestions": ["...", "..."]}}
"""


def _summarize_subtypes(subtype_differential) -> str:
    if not subtype_differential:
        return "not available"
    notable = [
        name.replace("_", " ")
        for name, entry in subtype_differential.items()
        if entry.get("likelihood") in ("possible", "present")
    ]
    return ", ".join(notable) if notable else "no subtype features indicated"


def generate_suggestions(
    phq9_pred, hamd_pred, phq9_clinician, hamd_clinician, binary_pred, risk_flag,
    subtype_differential=None,
) -> list | None:
    """Returns a list of short advisory strings, or None if unavailable."""
    prompt = PROMPT_TEMPLATE.format(
        heuristics=TREATMENT_HEURISTICS,
        phq9_pred=phq9_pred, phq9_clinician=phq9_clinician,
        hamd_pred=hamd_pred, hamd_clinician=hamd_clinician,
        binary_pred=bool(binary_pred), risk_flag=bool(risk_flag),
        subtype_summary=_summarize_subtypes(subtype_differential),
    )
    raw = call_claude(prompt, max_tokens=400)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        suggestions = [str(s)[:300] for s in data.get("suggestions", [])][:3]
        return suggestions or None
    except Exception as e:
        print(f"[treatment_suggestions] bad model output ({type(e).__name__}); omitting.")
        return None
