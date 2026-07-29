"""
Suicide/self-harm referral flag: PHQ-9 item 9 and the HAM-D suicide domain.

Rule-based on purpose. The source instrument (QUESTIONAIRE-V2_R) requires
"immediate risk assessment and referral" for these items regardless of the
ML model's prediction or training state — this must never depend on
FusionHead, XAI, or any generated narrative. It is checked and surfaced
even when the fusion head is untrained (near-random weights).

PHQ-9 item 9 ("thoughts that you would be better off dead or of hurting
yourself in some way") is scored 0-3, same as every other PHQ-9 item.
HAM-D's suicide domain is scored 0-4 (it's one of the 4-point items).
Any score above "not at all" (0) triggers the flag.
"""

PHQ9_ITEM9_MIN, PHQ9_ITEM9_MAX = 0, 3
HAMD_SUICIDE_MIN, HAMD_SUICIDE_MAX = 0, 4

RISK_THRESHOLD = 1  # any score >= 1 ("several days" / mild or worse) triggers referral


def flag_risk(phq9_item9: int, hamd_suicide_item: int) -> bool:
    """
    Returns True if either risk-carrying item is elevated, independent of any
    ML prediction. Never gate this on model confidence, training status, or
    a generated narrative — it is a standalone safety check.
    """
    if phq9_item9 is None and hamd_suicide_item is None:
        raise ValueError("flag_risk requires at least one of phq9_item9 / hamd_suicide_item")

    phq9_hit = phq9_item9 is not None and phq9_item9 >= RISK_THRESHOLD
    hamd_hit = hamd_suicide_item is not None and hamd_suicide_item >= RISK_THRESHOLD
    return bool(phq9_hit or hamd_hit)


if __name__ == "__main__":
    print(flag_risk(0, 0))  # False
    print(flag_risk(2, 0))  # True
    print(flag_risk(None, 3))  # True
