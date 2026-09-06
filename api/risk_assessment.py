"""
Structured suicide-risk assessment record.

`src/safety/risk_flag.py` answers "does this need a risk assessment?" from two
questionnaire items. It has never recorded the answer. The banner fires, the
clinician acts, and nothing in the record says what they found or what they
decided — which is the single largest documentation gap in the system
(docs/clinical-documentation-plan.md 1.4, 2).

This module owns the vocabulary of that record and one narrow AI affordance.
The required elements are the ones a risk note is expected to carry: ideation
type, intent, plan, access to means, prior attempts, protective factors,
safety plan, disposition, and the clinician's reasoning for the level-of-care
decision — that last field carrying the most weight, because a disposition
without a stated rationale is the thing that cannot be defended later.

**No judgement field here is ever AI-completed.** The model's only job is
`find_risk_quotes`, which surfaces verbatim lines from the transcript for the
clinician to quote. That boundary is deliberate and mirrors risk_flag.py's:
the safety path must not depend on a model being available, correct, or
honest. A missing API key degrades quote suggestions to an empty list and
changes nothing else.
"""
import json

from api.llm_utils import call_claude

# Passive ("better off dead") and active ("thoughts of killing myself")
# ideation carry different urgency and must not collapse into one flag.
IDEATION_LEVELS = ("none", "passive", "active")
# What happens after this visit. Named rather than free text so the caseload
# view can count dispositions without parsing prose.
DISPOSITIONS = (
    "routine_follow_up",
    "urgent_follow_up",
    "referral",
    "same_day_referral",
    "admission",
)

QUOTE_PROMPT = """You are helping a clinician document a suicide-risk assessment. Below is an \
interview transcript (participant turns only).

\"\"\"
{transcript}
\"\"\"

Quote up to 5 short EXACT phrases from the transcript above — verbatim, never paraphrased — in \
which the participant refers to self-harm, suicide, hopelessness, being a burden, not wanting \
to go on, or reasons for living. Reasons for living matter as much as risk statements: a \
protective factor in the participant's own words belongs in the record too.

Do not interpret, rate, or summarise. Do not infer anything the participant did not say. If \
nothing in the transcript refers to any of this, return an empty list.

Respond with ONLY a JSON object: {{"quotes": ["...", "..."]}}
"""


def find_risk_quotes(transcript_text: str) -> list[str]:
    """Candidate verbatim quotes for the clinician to insert. Never an assessment.

    A patient's own words are more defensible in the record than a
    paraphrase, and we hold a timestamped transcript, so the expensive part of
    quoting accurately is already done. Returns [] rather than raising when
    there is no transcript or no LLM — the form stays fully usable without it.
    """
    if not transcript_text.strip():
        return []
    raw = call_claude(QUOTE_PROMPT.format(transcript=transcript_text), max_tokens=400)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
        # Only keep quotes that are actually present in the transcript. The
        # prompt says verbatim; this makes it true rather than trusted.
        return [
            str(q)[:200] for q in data.get("quotes", []) if str(q).strip() and str(q) in transcript_text
        ][:5]
    except Exception as e:
        print(f"[risk_assessment] bad model output ({type(e).__name__}); no quotes suggested.")
        return []
