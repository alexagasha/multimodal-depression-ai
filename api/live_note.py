"""
Running SOAP note drafted *during* the interview, from the partial transcript
alone.

Distinct from api/note_draft.py, which drafts the final note once scoring has
produced PHQ-9/HAM-D estimates, a risk flag, and a subtype differential. While
the interview is still in progress none of that exists yet, so this module
prompts on transcript-only and is explicit that the note is incomplete — the
point is that the clinician watches the note assemble itself as the patient
talks, rather than clicking a button afterwards and waiting.

Regenerating on every audio chunk would be wasteful and rate-limited, so
api/main.py throttles calls here (see LIVE_NOTE_MIN_NEW_CHARS) and caches the
last draft. Same honest-fallback convention as the rest of the GenAI modules:
returns None when ANTHROPIC_API_KEY isn't set or the call fails — there is
deliberately no non-LLM fallback.
"""
import json

from api.llm_utils import call_claude

SOAP_FIELDS = ("subjective", "objective", "assessment", "plan")

PROMPT_TEMPLATE = """You are drafting a SOAP-format clinical note for a psychiatrist, \
LIVE, while the interview is still in progress. This is an in-progress draft of an \
incomplete interview — not a final note — and the psychiatrist remains fully responsible \
for its content and accuracy before any of it becomes part of the record.

Interview transcript so far (may end mid-sentence):
\"\"\"
{transcript}
\"\"\"

Screening scores have NOT been computed yet, so do not state or estimate any PHQ-9/HAM-D \
score, severity band, or diagnosis.

Draft each SOAP section in 1-3 sentences, grounded ONLY in what the transcript actually \
contains — do not invent clinical history, symptoms, or plans not present above. Where the \
transcript so far gives you nothing for a section, use a short placeholder such as \
"Not yet discussed." Respond with ONLY a JSON object:
{{"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."}}
"""


def draft_live_note(transcript_text: str) -> dict | None:
    """Returns {"subjective", "objective", "assessment", "plan"} or None."""
    if not transcript_text.strip():
        return None
    raw = call_claude(PROMPT_TEMPLATE.format(transcript=transcript_text), max_tokens=600)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return {field: str(data[field])[:1500] for field in SOAP_FIELDS}
    except Exception as e:
        print(f"[live_note] bad model output ({type(e).__name__}); returning unavailable.")
        return None
