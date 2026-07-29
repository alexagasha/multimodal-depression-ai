"""
GenUI narrative: turns raw scores + XAI attribution into a short,
clinician-readable summary for the score modal.

Same real-model-with-mock-fallback convention as the encoder classes: calls
Claude when ANTHROPIC_API_KEY is set and the `anthropic` package is
installed, otherwise falls back to a deterministic template so the score
modal always has something to render locally.

Explanatory only. This module NEVER computes or influences phq9_pred,
hamd_pred, binary_pred, or risk_flag — those are produced upstream by
src/fusion/model.py (deterministic) and src/safety/risk_flag.py (rule-based)
before this is ever called. See docs/data-collection-tool-migration.md §4 and
the plan's Phase 3/5 safety-boundary notes.

Phase 5 (RAG) upgrades PROMPT_TEMPLATE with retrieved DSM-5 criterion
snippets; not implemented yet — this is the in-session-data-only MVP.
"""
import os

PROMPT_TEMPLATE = """You are assisting a clinician reviewing a depression-severity \
screening result. Write a short (3-4 sentence), plain-language, clinician-readable \
summary. Do not state or imply a diagnosis; describe only what the data shows. If a \
referral flag is present, do not soften or omit it, but do not repeat it verbatim \
(the UI already shows a dedicated banner).

PHQ-9 (self-report, 0-27): {phq9_pred:.1f} predicted (true label for reference, if \
available, is not shown to the model)
HAM-D (clinician-rated, 0-44): {hamd_pred:.1f} predicted
Binary caseness (HAM-D >= {hamd_threshold}): {binary_pred}
Per-modality attribution (share of the prediction driven by each modality, \
signed toward/away from depression): {attributions}
Referral flag active: {risk_flag}
"""


def _template_narrative(phq9_pred, hamd_pred, binary_pred, attributions, risk_flag) -> str:
    dominant = max(attributions, key=lambda k: abs(attributions[k])) if attributions else None
    direction = "toward" if dominant and attributions[dominant] > 0 else "away from"
    lead = (
        f"The predicted scores are PHQ-9 {phq9_pred:.1f}/27 and HAM-D {hamd_pred:.1f}/44 "
        f"({'meets' if binary_pred else 'does not meet'} the caseness threshold)."
    )
    modality_note = (
        f" The {dominant} modality contributed most to the prediction, pushing it "
        f"{direction} depression."
        if dominant else ""
    )
    risk_note = (
        " A referral flag is active for this session — see the safety banner; this is "
        "independent of the model prediction above."
        if risk_flag else ""
    )
    return (
        lead + modality_note + risk_note +
        " This summary is AI-generated commentary based on the pipeline's own output, "
        "not a diagnosis."
    )


def generate_narrative(phq9_pred, hamd_pred, binary_pred, attributions: dict,
                        risk_flag: bool, hamd_threshold: float) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = PROMPT_TEMPLATE.format(
                phq9_pred=phq9_pred, hamd_pred=hamd_pred, binary_pred=bool(binary_pred),
                attributions=attributions, risk_flag=bool(risk_flag),
                hamd_threshold=hamd_threshold,
            )
            resp = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:  # anthropic missing, no network, bad key, etc. -> template
            print(f"[genui] real LLM unavailable ({type(e).__name__}); using template fallback.")
    return _template_narrative(phq9_pred, hamd_pred, binary_pred, attributions, risk_flag)
