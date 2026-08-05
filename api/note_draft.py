"""
AI-drafted SOAP-style clinical note from the visit transcript + scores.
The clinician reviews/edits the draft, then submits it as a real
ClinicalNote via the existing POST /sessions/{id}/notes endpoint — this
module only produces a draft; it never writes to the clinical_notes store
itself. Computed on demand (not automatically at scoring time), since a
clinician may want to regenerate it after adding their own notes first.
"""
import json

from api.llm_utils import call_claude

SOAP_FIELDS = ("subjective", "objective", "assessment", "plan")

PROMPT_TEMPLATE = """You are drafting a SOAP-format clinical note for a psychiatrist to review \
and edit — this is a draft only, not a final note, and the psychiatrist remains fully \
responsible for its content and accuracy before it becomes part of the record.

Interview transcript (participant turns only):
\"\"\"
{transcript}
\"\"\"

Screening data:
- PHQ-9 (AI estimate / clinician-scored): {phq9_pred:.1f} / {phq9_clinician}
- HAM-D (AI estimate / clinician-scored): {hamd_pred:.1f} / {hamd_clinician}
- Referral flag active: {risk_flag}
- Subtype differential (AI-suggested): {subtype_summary}

Draft each SOAP section in 2-3 sentences, grounded only in the transcript and data above — do \
not invent clinical history not present here. Respond with ONLY a JSON object:
{{"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."}}
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


def draft_note(
    transcript_text, phq9_pred, hamd_pred, phq9_clinician, hamd_clinician, risk_flag,
    subtype_differential=None,
) -> dict | None:
    """Returns {"subjective": ..., "objective": ..., "assessment": ..., "plan": ...} or None."""
    if not transcript_text.strip():
        return None
    prompt = PROMPT_TEMPLATE.format(
        transcript=transcript_text,
        phq9_pred=phq9_pred, phq9_clinician=phq9_clinician,
        hamd_pred=hamd_pred, hamd_clinician=hamd_clinician,
        risk_flag=bool(risk_flag),
        subtype_summary=_summarize_subtypes(subtype_differential),
    )
    raw = call_claude(prompt, max_tokens=700)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return {field: str(data[field])[:1500] for field in SOAP_FIELDS}
    except Exception as e:
        print(f"[note_draft] bad model output ({type(e).__name__}); returning unavailable.")
        return None
