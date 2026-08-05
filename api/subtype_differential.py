"""
Symptom-subtype differential: reads the interview transcript against
paraphrased DSM-5 depression-specifier criteria and produces a qualitative
differential across four subtypes — decision support for a specialist
choosing a treatment direction, not a diagnosis.

Same real-model-with-mock-fallback convention as api/genui.py: calls Claude
when ANTHROPIC_API_KEY is set, otherwise the feature is simply unavailable.
Unlike genui.py's narrative (which can reasonably degrade to a
scores-and-numbers template), a subtype differential requires actually
reading the transcript's language — there is no honest non-LLM fallback, so
this returns None rather than fabricating a differential from data that
doesn't support one.

Explanatory only, same safety boundary as api/genui.py: this module NEVER
computes or influences phq9_pred, hamd_pred, binary_pred, or risk_flag.
Psychotic features scored "present" is clinically significant on its own
(possible need for urgent referral / antipsychotic augmentation) but is
still AI-suggested, not a standalone alert — the UI labels it accordingly
and keeps it visually distinct from the rule-based referral banner in
src/safety/risk_flag.py.

This inlines the criteria directly in the prompt rather than a vector
index — the corpus is four short paraphrased summaries, small and static,
so retrieval infrastructure (Phase 5 in docs/system-roadmap.md) isn't
needed yet. Paraphrased, not verbatim: DSM-5 text is APA copyrighted.
"""
import json
import os

# Paraphrased summaries of DSM-5 MDD specifier criteria — not verbatim text.
SUBTYPE_CRITERIA = {
    "melancholic": (
        "Profound loss of pleasure in nearly all activities; mood doesn't brighten "
        "even briefly with good news; a distinct quality of low mood unlike ordinary "
        "sadness; worse in the morning; early-morning waking; noticeable slowing or "
        "agitation; poor appetite or weight loss; excessive or inappropriate guilt."
    ),
    "atypical": (
        "Mood brightens in response to positive events; increased appetite or weight "
        "gain; sleeping more than usual; a heavy, leaden feeling in arms or legs; a "
        "long-standing pattern of being very sensitive to perceived rejection."
    ),
    "anxious_distress": (
        "Feeling keyed up or tense; unusually restless; difficulty concentrating "
        "because of worry; fear that something awful might happen; a sense of "
        "possibly losing control."
    ),
    "psychotic_features": (
        "Delusional beliefs (e.g. guilt, poverty, illness, nihilism) or "
        "hallucinations occurring alongside the depressive episode. Clinically "
        "significant if present — may warrant urgent psychiatric referral."
    ),
}

LIKELIHOODS = {"none", "possible", "present"}

PROMPT_TEMPLATE = """You are assisting a psychiatrist with a symptom-subtype differential \
for a depression screening interview. This is decision support only, not a diagnosis.

For each of the four subtypes below, read the interview transcript and judge whether the \
described features are: "none" (no indication), "possible" (some suggestive but limited \
evidence), or "present" (clear supporting evidence in the transcript). Base this ONLY on \
what the participant actually said — do not infer beyond the text.

Subtypes and their criteria:
{criteria_block}

Interview transcript (participant turns only):
\"\"\"
{transcript}
\"\"\"

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"melancholic": {{"likelihood": "...", "rationale": "..."}}, \
"atypical": {{"likelihood": "...", "rationale": "..."}}, \
"anxious_distress": {{"likelihood": "...", "rationale": "..."}}, \
"psychotic_features": {{"likelihood": "...", "rationale": "..."}}}}
Each "rationale" must be one short sentence, quoting or closely paraphrasing the transcript, \
or "no supporting evidence in transcript" if likelihood is "none".
"""


def _validate(data: dict) -> dict:
    """Defensive validation of the model's JSON — malformed output degrades to
    unavailable rather than showing a clinician a garbled differential."""
    result = {}
    for subtype in SUBTYPE_CRITERIA:
        entry = data.get(subtype)
        if not isinstance(entry, dict):
            raise ValueError(f"missing/malformed entry for {subtype}")
        likelihood = entry.get("likelihood")
        if likelihood not in LIKELIHOODS:
            raise ValueError(f"invalid likelihood {likelihood!r} for {subtype}")
        result[subtype] = {
            "likelihood": likelihood,
            "rationale": str(entry.get("rationale", ""))[:300],
        }
    return result


def generate_differential(transcript_text: str) -> dict | None:
    """
    Returns {subtype: {"likelihood": "none"|"possible"|"present", "rationale": str}}
    for each of SUBTYPE_CRITERIA, or None if unavailable (no API key, no
    transcript text, or the call/parse failed) — callers must handle None as
    "not shown," never as "no subtypes present."
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not transcript_text.strip():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        criteria_block = "\n".join(
            f"- {name.replace('_', ' ')}: {desc}" for name, desc in SUBTYPE_CRITERIA.items()
        )
        prompt = PROMPT_TEMPLATE.format(criteria_block=criteria_block, transcript=transcript_text)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        return _validate(json.loads(raw))
    except Exception as e:  # anthropic missing, bad JSON, network, etc. -> unavailable
        print(f"[subtype_differential] unavailable ({type(e).__name__}); omitting from result.")
        return None
