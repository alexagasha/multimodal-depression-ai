"""
Evidence quotes: extends the XAI modality-attribution bars (numeric only,
src/xai/attribution.py) with the actual transcript phrases behind the
dominant modality's contribution, so a clinician can sanity-check the score
against their own read of the interview.

Not a literal saliency map — the frozen/mock encoders don't support
meaningful gradient attribution at this stage (see src/xai/attribution.py's
docstring: permutation ablation over the whole modality block, not
per-token). This is an honest LLM read of which lines in the transcript are
doing the work, grounded only in the transcript text actually said —
same convention as api/subtype_differential.py: no fabricated evidence,
unavailable rather than invented when there's no LLM connection.
"""
import json

from api.llm_utils import call_claude

PROMPT_TEMPLATE = """You are helping a clinician sanity-check an automated depression-severity \
score against the interview transcript. The {modality} modality contributed most to this \
participant's predicted score (direction: {direction} depression).

Interview transcript (participant turns only):
\"\"\"
{transcript}
\"\"\"

Quote up to 3 short EXACT phrases from the transcript above (verbatim, not paraphrased) that \
best illustrate why the {modality} modality would carry the most weight. If nothing in the \
transcript clearly supports this, return an empty list rather than inventing something.

Respond with ONLY a JSON object: {{"quotes": ["...", "..."]}}
"""


def generate_evidence(transcript_text: str, modality_attributions: dict) -> dict | None:
    """
    Returns {"modality": str, "quotes": [str, ...]} for the dominant
    modality, or None if unavailable (no API key, no transcript, no
    attribution data, or the call/parse failed).
    """
    if not transcript_text.strip() or not modality_attributions:
        return None
    dominant = max(modality_attributions, key=lambda k: abs(modality_attributions[k]))
    direction = "toward" if modality_attributions[dominant] > 0 else "away from"

    prompt = PROMPT_TEMPLATE.format(modality=dominant, direction=direction, transcript=transcript_text)
    raw = call_claude(prompt, max_tokens=300)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        quotes = [str(q)[:200] for q in data.get("quotes", [])][:3]
        return {"modality": dominant, "quotes": quotes}
    except Exception as e:
        print(f"[evidence] bad model output ({type(e).__name__}); omitting from result.")
        return None
