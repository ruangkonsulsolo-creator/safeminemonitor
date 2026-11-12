"""
Preliminary risk assessment (HIRARC-style Likelihood x Severity matrix).

Methodological basis (see docs/risk_methodology.md):
  * Risk = Likelihood (L, 1-5) x Severity (S, 1-5), the semi-quantitative
    matrix used in HIRARC (DOSH Malaysia, 2008) and common OHS practice.
  * Risk bands: 1-4 Low, 5-12 Medium, 15-25 High (HIRARC banding).
  * Severity S is fixed per behaviour category and justified from the
    literature / accident statistics (it is a property of the *hazard*).
  * Likelihood L is derived from *observed exposure* in the video: the
    share of the worker's observed time spent in the unsafe state, and how
    often the behaviour recurs.  It is therefore data-driven, not arbitrary.
  * Recommendations follow the hierarchy of controls in ISO 45001:2018
    clause 8.1.2 (elimination > substitution > engineering > administrative > PPE).

All numbers here are *preliminary* and must be validated with the
supervisor / site HSE experts (e.g., Delphi or expert-rating survey).
"""
from __future__ import annotations

from collections import Counter

# ---------------------------------------------------------------- severity
SEVERITY = {
    # category: (S on the HIRARC 2008 severity scale, justification)
    "NO_HELMET": (3, "Head injury from falling rock/objects (struck-by). Credible worst case for one worker: "
                     "serious, non-fatal injury with possible permanent disability (HIRARC 'Serious')."),
    "NO_VEST": (3, "Reduced conspicuity raises struck-by-vehicle risk around haul roads and loading areas; "
                   "credible outcome: serious injury (HIRARC 'Serious')."),
    "RESTRICTED_ZONE": (4, "Entry into blasting, highwall/edge or exclusion zones exposes the worker to falls of "
                           "ground, falls from height or blast effects; single fatality credible (HIRARC 'Fatal')."),
    "MACHINE_PROXIMITY": (4, "Powered haulage / mobile-equipment contact is among the most frequent causes of mining "
                             "fatalities in MSHA statistics; single fatality credible (HIRARC 'Fatal')."),
}

# Scales follow DOSH Malaysia (2008) HIRARC Guidelines, Tables A and B.
SEVERITY_SCALE = {1: "Negligible (first-aid type injury)", 2: "Minor (disabling, not permanent)",
                  3: "Serious (non-fatal, permanent disability)", 4: "Fatal (single fatality)",
                  5: "Catastrophic (numerous fatalities)"}
# Likelihood descriptors from HIRARC; the quantitative anchor (observed exposure
# share of the worker's time on camera) is this study's operationalisation.
LIKELIHOOD_SCALE = {1: "Inconceivable - exposure <10% of observed time",
                    2: "Remote - exposure 10-25%", 3: "Conceivable - exposure 25-50%",
                    4: "Possible - exposure 50-75%",
                    5: "Most likely - exposure >75%, or behaviour repeated >=3 times"}


def likelihood_from_exposure(exposure_ratio: float, recurrences: int = 1) -> int:
    """Map observed exposure ratio (0-1) and recurrence to L in 1..5."""
    e = max(0.0, min(1.0, exposure_ratio or 0.0))
    if e < 0.10:
        L = 1
    elif e < 0.25:
        L = 2
    elif e < 0.50:
        L = 3
    elif e < 0.75:
        L = 4
    else:
        L = 5
    if recurrences >= 3:        # behaviour repeated by the same worker -> habitual
        L = min(5, L + 1)
    return L


def risk_level(score: int) -> str:
    if score >= 15:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


RECOMMENDATIONS = {
    "NO_HELMET": {
        "High": "Stop work for the worker immediately; enforce helmet before re-entry. Install PPE check-point "
                "at zone access (engineering); toolbox talk for the crew (administrative).",
        "Medium": "Supervisor verbal correction and record; reinforce PPE rule at shift briefing.",
        "Low": "Record observation; include in weekly behaviour-based safety (BBS) feedback.",
    },
    "NO_VEST": {
        "High": "Remove worker from traffic area until hi-vis vest is worn; review traffic management plan.",
        "Medium": "Supervisor correction; ensure vest availability at site entrance.",
        "Low": "Record observation; BBS feedback.",
    },
    "RESTRICTED_ZONE": {
        "High": "Evacuate worker from the zone; install physical barricade / interlock (engineering); "
                "review permit-to-work and zone signage.",
        "Medium": "Re-brief exclusion-zone boundaries; improve signage and demarcation.",
        "Low": "Record and monitor recurrence.",
    },
    "MACHINE_PROXIMITY": {
        "High": "Stop equipment; enforce exclusion radius / spotter; consider proximity-warning system "
                "(engineering control) and pedestrian-vehicle segregation.",
        "Medium": "Review pedestrian-vehicle interaction plan; assign spotter for the task.",
        "Low": "Record; reinforce line-of-fire awareness training.",
    },
}


def assess_events(events: list[dict]) -> list[dict]:
    """Attach L, S, risk score, level and recommendation to each event (in place)."""
    rec_count = Counter((e.get("track_id"), e["category"]) for e in events)
    for e in events:
        S = SEVERITY.get(e["category"], (3, ""))[0]
        L = likelihood_from_exposure(e.get("exposure_ratio", 0.0),
                                     rec_count[(e.get("track_id"), e["category"])])
        score = L * S
        e.update(likelihood=L, severity=S, risk_score=score, risk_level=risk_level(score))
    return events


def recommendation(category: str, level: str) -> str:
    return RECOMMENDATIONS.get(category, {}).get(level, "Review with HSE officer.")


def site_risk_summary(events: list[dict]) -> dict:
    if not events:
        return {"overall_level": "Low", "max_score": 0, "mean_score": 0, "by_level": {}, "by_category": {}}
    scores = [e["risk_score"] for e in events]
    by_level = Counter(e["risk_level"] for e in events)
    by_cat = {}
    for e in events:
        c = by_cat.setdefault(e["category"], {"count": 0, "max_score": 0})
        c["count"] += 1
        c["max_score"] = max(c["max_score"], e["risk_score"])
    for c, v in by_cat.items():
        v["level"] = risk_level(v["max_score"])
        v["recommendation"] = recommendation(c, v["level"])
    return {"overall_level": risk_level(max(scores)), "max_score": max(scores),
            "mean_score": round(sum(scores) / len(scores), 2),
            "by_level": dict(by_level), "by_category": by_cat}


def matrix_definition() -> dict:
    return {
        "severity": {k: {"S": v[0], "basis": v[1]} for k, v in SEVERITY.items()},
        "severity_scale": SEVERITY_SCALE,
        "likelihood_scale": LIKELIHOOD_SCALE,
        "bands": {"Low": "1-4", "Medium": "5-12", "High": "15-25"},
        "method": "HIRARC semi-quantitative matrix: Risk = Likelihood x Severity",
        "controls": "ISO 45001:2018 cl. 8.1.2 hierarchy of controls",
    }
