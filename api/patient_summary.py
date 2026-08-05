"""
Patient-facing after-visit summary: a short, plain-language, low-literacy-
friendly recap for the patient to take home. Contrasts deliberately with
api/genui.py's clinician-facing narrative, which assumes clinical
vocabulary (PHQ-9, HAM-D, "caseness") that a patient handout must avoid.
Pairs with the read-aloud pattern already in web/components/ScaleForm.tsx
(browser speechSynthesis) — the frontend can read this aloud the same way.

No numeric fallback template on purpose: getting the tone right for a
patient-facing message (especially when a referral flag is active) needs
actual language generation, not score arithmetic — a naive template risks
being clinically tone-deaf. Unavailable rather than invented, same
convention as the other GenAI modules here.
"""
from api.llm_utils import call_claude

PROMPT_TEMPLATE = """Write a short (3-4 sentence), warm, plain-language summary for a PATIENT \
(not a clinician) to take home after a depression screening visit. Aim for roughly a \
4th-grade reading level. Do not use clinical jargon or scale names (no "PHQ-9", "HAM-D", \
"caseness", "binary prediction"). Do not state or imply a diagnosis. Thank them for sharing, \
and encourage them to keep their next appointment.

{risk_note}

Internal context only, do not print any numbers or scores in your response:
- Screening indicated: {severity_word} level of symptoms
- Referral flag active: {risk_flag}
"""

RISK_NOTE = (
    "Because something in today's conversation needs closer follow-up, gently let them know "
    "their care team will be reaching out soon — reassuring, not alarming, and do not describe "
    "why."
)


def _severity_word(hamd_pred: float) -> str:
    if hamd_pred >= 24:
        return "high"
    if hamd_pred >= 14:
        return "moderate"
    if hamd_pred >= 7:
        return "mild"
    return "low"


def generate_patient_summary(hamd_pred: float, risk_flag: bool) -> str | None:
    prompt = PROMPT_TEMPLATE.format(
        risk_note=RISK_NOTE if risk_flag else "",
        severity_word=_severity_word(hamd_pred),
        risk_flag=bool(risk_flag),
    )
    return call_claude(prompt, max_tokens=250)
