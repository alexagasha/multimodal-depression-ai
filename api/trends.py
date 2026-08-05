"""
Rule-based (NOT LLM) trend detection across a patient's visit history:
relapse early-warning (severity trajectory) and risk trajectory (repeated
or rising referral-flag pattern). Deterministic on purpose — safety-adjacent
signals shouldn't depend on LLM availability or be prone to hallucination,
same philosophy as src/safety/risk_flag.py. Neither function calls an LLM.

Both take a chronologically-sorted list of *enriched* sessions (i.e.
api/main.py's _enrich_session output: has hamd_pred, risk_flag).
"""

MIN_VISITS_FOR_RELAPSE_WARNING = 3
RELAPSE_HAMD_DELTA = 3.0  # points on the /44 scale
RISK_TRAJECTORY_WINDOW = 3
RISK_TRAJECTORY_MIN_FLAGGED = 2


def relapse_warning(visits: list) -> dict:
    """
    Flags a deteriorating trajectory: the most recent N scored visits are
    monotonically non-improving (HAM-D never drops) and the last is
    meaningfully worse than the first of that window — not just noise
    around a flat baseline.
    """
    scored = [v for v in visits if v.get("hamd_pred") is not None]
    if len(scored) < MIN_VISITS_FOR_RELAPSE_WARNING:
        return {"flag": False, "reason": "fewer than 3 scored visits on record"}

    window = scored[-MIN_VISITS_FOR_RELAPSE_WARNING:]
    values = [v["hamd_pred"] for v in window]
    non_improving = all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))
    meaningfully_worse = (values[-1] - values[0]) >= RELAPSE_HAMD_DELTA
    flag = non_improving and meaningfully_worse
    return {
        "flag": flag,
        "reason": (
            f"HAM-D rose from {values[0]:.1f} to {values[-1]:.1f} over the last "
            f"{len(window)} visits without improvement"
            if flag
            else "no sustained worsening detected"
        ),
    }


def risk_trajectory(visits: list) -> dict:
    """
    Flags an escalating pattern: 2+ of the last 3 scale-response-bearing
    visits carried an active referral flag. A single flagged visit is
    already surfaced by the rule-based risk_flag itself (RiskBanner); this
    is specifically about a repeating pattern across visits.
    """
    with_scales = [v for v in visits if v.get("risk_flag") is not None]
    if not with_scales:
        return {"flag": False, "reason": "no scale responses recorded yet"}

    window = with_scales[-RISK_TRAJECTORY_WINDOW:]
    n_flagged = sum(1 for v in window if v["risk_flag"])
    flag = n_flagged >= RISK_TRAJECTORY_MIN_FLAGGED
    return {
        "flag": flag,
        "reason": f"{n_flagged} of the last {len(window)} visits carried an active referral flag",
    }
