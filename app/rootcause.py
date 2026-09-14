"""
Two related but distinct outputs (see SCHEMA.md's StationETA):

  top_reasons: per-prediction SHAP feature attribution on the p50 model --
  "why did the model predict this specific correction". Uses
  shap.TreeExplainer, the exact/fast algorithm for tree ensembles (same
  choice explainability/shap_poc.py already proved out on a toy model --
  this is the real version, on the real p50 quantile model).

  cause_tags: a SEPARATE, simpler, rule-based mapping from the top-impact
  features to known operational cause categories (signal_halt, weather,
  etc.) -- this was the original design ask: "rule-based classifier reusing
  the same feature set", not a second ML model. FEATURE_TO_CAUSE below is
  that rule table.
"""

import numpy as np
import shap

from features import FEATURE_NAMES, FEATURE_LABELS

MIN_IMPACT_MIN_FOR_REASON = 0.3   # don't report noise-level SHAP contributions
MIN_IMPACT_MIN_FOR_CAUSE_TAG = 0.5

# Maps a feature to the operational cause category it's evidence for.
# Deliberately excludes structural/state features (current_delay_min,
# station_index, distance_km) -- those describe WHERE we are, not WHY new
# delay is being added, which is what a cause tag is supposed to explain.
FEATURE_TO_CAUSE = {
    "cum_single_line_sections": "congestion_preceding_train",
    "cum_monsoon_risk_score": "weather",
    "cum_fog_risk_score": "weather",
    "is_monsoon": "weather",
    "is_winter_fog": "weather",
    "is_festival_surge": "congestion_preceding_train",
    "preceding_train_delay_min": "congestion_preceding_train",
}


class RootCauseExplainer:
    """Wraps a single TreeExplainer instance -- construct once per loaded
    p50 model (SHAP's TreeExplainer setup cost is nontrivial; reuse it
    across requests rather than rebuilding per call)."""

    def __init__(self, p50_model):
        self.explainer = shap.TreeExplainer(p50_model)

    def explain(self, feature_row):
        """feature_row: a FeatureVector dict (see SCHEMA.md).
        Returns (top_reasons, cause_tags)."""
        X = np.array([[feature_row[f] for f in FEATURE_NAMES]])
        shap_values = self.explainer.shap_values(X)[0]
        contribs = sorted(zip(FEATURE_NAMES, shap_values), key=lambda x: -abs(x[1]))

        top_reasons = []
        for fname, val in contribs:
            if abs(val) < MIN_IMPACT_MIN_FOR_REASON:
                continue
            top_reasons.append({"factor": FEATURE_LABELS[fname], "impact_min": round(float(val), 1)})
            if len(top_reasons) >= 3:
                break

        cause_tags = []
        for fname, val in contribs:
            if abs(val) < MIN_IMPACT_MIN_FOR_CAUSE_TAG:
                continue
            cause = FEATURE_TO_CAUSE.get(fname)
            if cause and cause not in cause_tags:
                cause_tags.append(cause)

        return top_reasons, cause_tags


if __name__ == "__main__":
    import os
    import lightgbm as lgb

    HERE = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.join(HERE, "..", "models", "artifacts")
    model_p50 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "residual_p50.txt"))
    explainer = RootCauseExplainer(model_p50)

    sample_row = {
        "cum_single_line_sections": 2, "cum_monsoon_risk_score": 5, "cum_fog_risk_score": 0,
        "distance_km": 331, "station_index": 3, "is_monsoon": 1, "is_winter_fog": 0,
        "is_festival_surge": 0, "current_delay_min": 12.0, "preceding_train_delay_min": 8.0,
    }
    top_reasons, cause_tags = explainer.explain(sample_row)
    print("Sample: single-line + monsoon-risk section, mid-journey")
    print("Top reasons:")
    for r in top_reasons:
        sign = "+" if r["impact_min"] >= 0 else ""
        print(f"  {sign}{r['impact_min']} min: {r['factor']}")
    print(f"Cause tags: {cause_tags}")