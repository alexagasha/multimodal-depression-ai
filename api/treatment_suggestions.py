"""
Guideline-grounded next-step suggestions: paraphrased, non-proprietary
clinical heuristics (informed by general treatment-resistant-depression
stepped-care sequencing in the style of STAR*D, and DSM-5 subtype-treatment
matching), NOT a retrieval system over the actual copyrighted guidelines.
Same inlined-static-context approach as api/subtype_differential.py's DSM-5
criteria, scoped to one feature rather than full RAG infrastructure — see
docs/system-roadmap.md Phase 5 for where this graduates to real retrieval
if the corpus grows.

Always advisory, always phrased as "consider"/"may warrant" — never a
directive or prescription. Explanatory only: this module NEVER feeds back
into phq9_pred, hamd_pred, binary_pred, or risk_flag.
"""
import json

from api.llm_utils import call_claude

TREATMENT_HEURISTICS = """
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

General heuristics to draw from (not a substitute for full clinical guidelines):
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
