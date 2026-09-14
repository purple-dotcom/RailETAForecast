"""
Rule-based widening of the (p10, p90) band around p50.

Why this exists on top of the ML quantile spread: the quantile models can
only learn to widen the band from patterns present in ~19.5K training rows,
covering 4 routes. That's thin evidence for rare combinations (e.g. a
single-line section IN monsoon season for a route that's rarely single-line
at all). A rule-based multiplier, keyed to the same section-risk flags a
human dispatcher would look at, is a cheap and interpretable safety net for
exactly the cases the ML has the least data on -- which is also why the
original design explicitly asked for this as a separate layer rather than
trusting the quantile model alone (see conversation earlier in this log).
"""

WIDEN_PER_SINGLE_LINE_SECTION = 0.15
WIDEN_PER_MONSOON_RISK_POINT = 0.08
WIDEN_PER_FOG_RISK_POINT = 0.08


def widen_band(p10_min, p50_min, p90_min, feature_row):
    """Returns (widened_p10_min, widened_p90_min, confidence_note).
    p50 is never touched -- only the spread around it changes."""
    spread_below = max(0.0, p50_min - p10_min)
    spread_above = max(0.0, p90_min - p50_min)

    multiplier = 1.0
    reasons = []

    n_single_line = feature_row.get("cum_single_line_sections", 0)
    if n_single_line > 0:
        multiplier += WIDEN_PER_SINGLE_LINE_SECTION * n_single_line
        reasons.append("single-line section" + ("s" if n_single_line > 1 else "") + " ahead")

    if feature_row.get("is_monsoon") and feature_row.get("cum_monsoon_risk_score", 0) > 0:
        multiplier += WIDEN_PER_MONSOON_RISK_POINT * feature_row["cum_monsoon_risk_score"]
        reasons.append("monsoon risk ahead")

    if feature_row.get("is_winter_fog") and feature_row.get("cum_fog_risk_score", 0) > 0:
        multiplier += WIDEN_PER_FOG_RISK_POINT * feature_row["cum_fog_risk_score"]
        reasons.append("fog risk ahead")

    widened_p10 = p50_min - spread_below * multiplier
    widened_p90 = p50_min + spread_above * multiplier

    if reasons:
        confidence_note = "widening due to " + " and ".join(reasons)
    else:
        confidence_note = "normal confidence band"

    return widened_p10, widened_p90, confidence_note


if __name__ == "__main__":
    row_clean = {"cum_single_line_sections": 0, "cum_monsoon_risk_score": 0,
                 "cum_fog_risk_score": 0, "is_monsoon": 0, "is_winter_fog": 0}
    row_risky = {"cum_single_line_sections": 2, "cum_monsoon_risk_score": 5,
                 "cum_fog_risk_score": 0, "is_monsoon": 1, "is_winter_fog": 0}

    for label, row in [("clean double-line trunk section", row_clean),
                        ("single-line + monsoon-risk section", row_risky)]:
        p10, p90, note = widen_band(p10_min=5.0, p50_min=10.0, p90_min=18.0, feature_row=row)
        print(f"{label}:")
        print(f"  original band: 5.0 - 18.0 (spread {5.0} below / {8.0} above p50)")
        print(f"  widened band:  {p10:.1f} - {p90:.1f}")
        print(f"  note: {note}\n")