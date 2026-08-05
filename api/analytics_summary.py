"""
Optional GenAI narrative summarizing practice-level analytics — a
population view for a clinician managing a caseload, not just one patient
at a time. The underlying numbers are always computed deterministically in
api/main.py's GET /analytics endpoint; this module only narrates them in
plain language, grounded strictly in the numbers it's given. Same
unavailable-without-key convention as the other narrative generators.
"""
import json

from api.llm_utils import call_claude

PROMPT_TEMPLATE = """Summarize this caseload snapshot for a psychiatrist in 2-3 plain-language \
sentences. Use ONLY the numbers given below — do not estimate or invent any figure not present.

{stats_json}
"""


def generate_analytics_narrative(stats: dict) -> str | None:
    prompt = PROMPT_TEMPLATE.format(stats_json=json.dumps(stats, default=str))
    return call_claude(prompt, max_tokens=250)
