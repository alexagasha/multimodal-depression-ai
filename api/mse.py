"""
Mental status examination scaffold for the Objective section.

The MSE is the clinical spine the SOAP note's Objective section has been
missing: ten domains of structured observation, without which "Objective" is
just more prose. See docs/clinical-documentation-plan.md 1.3.

THE GOVERNING RULE IS THAT WE ONLY HAVE AUDIO OF THE PARTICIPANT.

Three domains — appearance, motor behaviour, attitude toward the examiner —
cannot be observed from a recording at all, and are returned as explicitly
not observable rather than guessed. This is the whole reason the module is
worth having: an MSE that quietly invented an appearance finding would be
worse than no MSE, because it would look complete. The clinician fills those
in; the system says plainly that it cannot.

Speech is the one domain the system can populate with measurement rather than
inference. src/pipelines/prosody_pipeline.py already computes speaking rate,
pause structure and pitch per turn — the same features the severity model is
fitted on. Those values are reported as measurements, with no normal range
asserted: this cohort has no published reference distribution, and inventing
one to label a participant's speech "slowed" would be exactly the kind of
plausible fabrication the rest of the pipeline is careful to avoid.

The remaining domains are read off the transcript by the LLM under
instructions to return null for anything the transcript does not evidence.
Same honest-fallback convention as the other GenAI modules: no key, no
guesses — the scaffold still renders, with every LLM-derived domain blank for
the clinician to complete.
"""
import json

from api.llm_utils import call_claude
from src.pipelines.prosody_pipeline import FEATURE_NAMES

# source: what could fill this domain, and therefore how much to trust it.
#   not_observable — audio cannot support it; the clinician must observe it
#   measured       — computed from the waveform
#   transcript     — read off what the participant said
DOMAINS = [
    ("appearance", "Appearance", "not_observable"),
    ("behaviour", "Behaviour / motor activity", "not_observable"),
    ("attitude", "Attitude toward examiner", "not_observable"),
    ("speech", "Speech", "measured"),
    ("mood", "Mood", "transcript"),
    ("affect", "Affect", "transcript"),
    ("thought_process", "Thought process", "transcript"),
    ("thought_content", "Thought content", "transcript"),
    ("perception", "Perception", "transcript"),
    ("insight_judgement", "Insight and judgement", "transcript"),
    ("cognition", "Cognition", "transcript"),
]

NOT_OBSERVABLE_NOTE = "Not observable from audio — requires in-person observation."

# Plain-language names for the prosody measures worth showing a clinician.
# A subset of FEATURE_NAMES: the means, which describe the participant's
# speech, rather than the SDs, which describe variability across turns and
# do not read as a clinical observation.
SPEECH_MEASURES = [
    ("speech_frac_mean", "Proportion of turn spent speaking", ""),
    ("pause_frac_mean", "Proportion of turn spent in pause", ""),
    ("n_pauses_per_s_mean", "Pauses per second", "/s"),
    ("mean_pause_mean", "Mean pause length", "s"),
    ("longest_pause_mean", "Longest pause", "s"),
    ("f0_mean_mean", "Mean pitch", "Hz"),
    ("f0_range_st_mean", "Pitch range", "semitones"),
    ("db_range_mean", "Loudness range", "dB"),
]

PROMPT_TEMPLATE = """You are filling in the transcript-derived domains of a mental status \
examination for a psychiatrist to review and complete. This is a draft only.

Interview transcript (participant turns only):
\"\"\"
{transcript}
\"\"\"

Fill in ONLY these domains, each in one short clinical sentence:
- mood: the participant's own stated emotional state. Quote their words where possible.
- affect: only what the transcript itself evidences. You cannot see the participant.
- thought_process: coherence, organisation, tangentiality, circumstantiality.
- thought_content: preoccupations, worries, guilt, hopelessness, and any expressed thoughts \
of death or self-harm.
- perception: only if the participant reports hallucinations or perceptual disturbance.
- insight_judgement: awareness of their difficulties and reasoning about them.
- cognition: only what is evident from the conversation — no formal testing was performed.

Use null, not a guess, for any domain the transcript does not evidence. Do not infer beyond \
what was said. Do not mention scores, severity bands or diagnoses.

Respond with ONLY a JSON object whose keys are exactly the seven domain names above.
"""

LLM_DOMAINS = (
    "mood", "affect", "thought_process", "thought_content",
    "perception", "insight_judgement", "cognition",
)


def _speech_domain(prosody_vector) -> dict:
    """Measured speech characteristics, reported without asserting a norm."""
    if prosody_vector is None or len(prosody_vector) != len(FEATURE_NAMES):
        return {"finding": None, "measures": None,
                "note": "Speech measures unavailable for this recording."}

    values = dict(zip(FEATURE_NAMES, (float(v) for v in prosody_vector)))
    # An all-zero vector means no turn was long enough to measure — usually an
    # empty transcript. Rendering that as "mean pitch 0 Hz" would present an
    # absence of measurement as a measurement, which is the specific failure
    # this module exists to avoid. Real speech cannot be simultaneously 0 Hz
    # and 0% voiced, so the test is unambiguous.
    if not any(values.values()):
        return {"finding": None, "measures": None,
                "note": "No speech was measurable in this recording — the transcript has no "
                        "turns long enough to analyse. Nothing is being reported as zero."}

    measures = [
        {"label": label, "value": round(values[key], 3), "unit": unit}
        for key, label, unit in SPEECH_MEASURES
        if key in values
    ]
    return {
        "finding": None,
        "measures": measures,
        "note": "Measured from the recording. No reference range is established for this "
                "cohort, so these are observations to interpret, not findings.",
    }


def build_mse(transcript_text: str, prosody_vector=None) -> dict:
    """Returns {"domains": [...]} — always the full ten-domain scaffold.

    Never returns None. A missing LLM or an empty transcript blanks the
    transcript-derived domains; it does not remove them. A blank domain the
    clinician can see and fill is useful; a silently absent one is not.
    """
    llm_findings = {}
    if transcript_text.strip():
        raw = call_claude(PROMPT_TEMPLATE.format(transcript=transcript_text), max_tokens=700)
        if raw is not None:
            try:
                data = json.loads(raw)
                llm_findings = {
                    d: (str(data[d])[:600] if data.get(d) else None)
                    for d in LLM_DOMAINS
                    if d in data
                }
            except Exception as e:
                print(f"[mse] bad model output ({type(e).__name__}); domains left blank.")

    domains = []
    for key, label, source in DOMAINS:
        entry = {"domain": key, "label": label, "source": source,
                 "finding": None, "measures": None, "note": None}
        if source == "not_observable":
            entry["note"] = NOT_OBSERVABLE_NOTE
        elif source == "measured":
            entry.update(_speech_domain(prosody_vector))
        else:
            entry["finding"] = llm_findings.get(key)
        domains.append(entry)
    return {"domains": domains}
