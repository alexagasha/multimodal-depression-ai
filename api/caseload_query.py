"""
Natural-language caseload queries: ask a question over the patient roster
("which patients haven't improved in 3 visits") and get a plain-language
answer plus the matching patient IDs to highlight in the UI. The whole
roster is passed as context — no function-calling/tool-use infrastructure
needed at local-demo scale (a handful of patients, not thousands).

Note: the roster passed in here is the same shape GET /participants
already returns to the browser, so no additional PII exposure beyond what
the roster UI already shows.
"""
import json

from api.llm_utils import call_claude

PROMPT_TEMPLATE = """You are helping a psychiatrist search their patient roster with a natural- \
language question. Answer ONLY from the roster data below — never invent a patient or a fact \
not present in it.

Roster (JSON, one entry per patient):
{roster_json}

Question: {question}

Respond with ONLY a JSON object: {{"answer": "one or two plain-language sentences", \
"matching_patient_ids": ["participant_id", ...]}}
If no patients match, use an empty list and say so in the answer.
"""


def answer_query(question: str, roster: list) -> dict | None:
    """Returns {"answer": str, "matching_patient_ids": [str, ...]} or None if unavailable."""
    if not question.strip():
        return None
    prompt = PROMPT_TEMPLATE.format(roster_json=json.dumps(roster, default=str), question=question)
    raw = call_claude(prompt, max_tokens=500)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return {
            "answer": str(data.get("answer", ""))[:1000],
            "matching_patient_ids": [str(x) for x in data.get("matching_patient_ids", [])],
        }
    except Exception as e:
        print(f"[caseload_query] bad model output ({type(e).__name__}); returning unavailable.")
        return None
