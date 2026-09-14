"""
shap_poc.py -- FEASIBILITY PROOF, not the production module.

Question being answered: does SHAP on a LightGBM model actually produce the
"+8 min: single-line section congestion, +3 min: preceding train delay,
-2 min: recovery margin ahead" style output Eli described, cheaply, for a
beginner-level team? This script trains a small illustrative model on the
synthetic historical_runs.csv and checks.

WHAT THIS IS NOT: the real residual-correction model (that predicts a
correction on top of the deterministic baseline, per-section, with the full
real feature set including live weather/congestion/preceding-train signals).
This is a coarser model (predicts total trip delay from a handful of
run-level features) purely to prove the SHAP + counterfactual MECHANISM
works and is cheap. The real module gets built properly once the baseline
calculator exists.

Why LightGBM + SHAP specifically, and why NOT an RNN here:
  - SHAP has an EXACT, fast algorithm for tree ensembles (TreeSHAP) --
    polynomial time, no sampling/approximation needed. That's what's used
    below (shap.TreeExplainer).
  - For neural nets (including RNNs), SHAP has to fall back to approximate
    methods (DeepSHAP / GradientSHAP / KernelSHAP) which are slower, noisier,
    and need more tuning to get sensible results.
  - We already decided against RNN/LSTM/GNN for the propagation module for
    unrelated reasons (data volume, team skill level). This is a second,
    independent reason that decision was right: it also makes explainability
    cheap instead of expensive. Nothing about "RNN" is relevant to us at all.
"""

import os
import json
import csv
from collections import defaultdict
from datetime import datetime

import lightgbm as lgb
import shap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "output")

# Human-readable labels for the toy feature set. In the real module this
# mapping is the main thing you maintain by hand as new features get added.
FEATURE_LABELS = {
    "single_line_frac": "single-line section exposure on this route",
    "distance_km": "total journey distance",
    "is_monsoon": "monsoon-season conditions",
    "is_winter_fog": "winter fog-season conditions",
    "is_festival_surge": "festival-season traffic surge",
    "is_single_line_route": "route runs through single-line track",
}


def load_features():
    with open(os.path.join(DATA_DIR, "seed_trains.json")) as f:
        seed = json.load(f)
    with open(os.path.join(DATA_DIR, "network_attributes.json")) as f:
        network = json.load(f)

    distance_by_train = {t["train_no"]: t["stations"][-1]["km"] for t in seed["trains"]}

    single_line_frac_by_train = {}
    for t in seed["trains"]:
        tn = t["train_no"]
        secs = [s for s in network["sections"] if s["train_no"] == tn]
        frac = sum(1 for s in secs if s.get("single_line")) / len(secs) if secs else 0.0
        single_line_frac_by_train[tn] = frac

    rows = []
    with open(os.path.join(OUT_DIR, "historical_runs.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["sched_dep"] != "":  # only final-destination rows have ground truth trip delay
                continue
            tn = r["train_no"]
            rows.append({
                "single_line_frac": single_line_frac_by_train.get(tn, 0.0),
                "distance_km": float(distance_by_train.get(tn, 0)),
                "is_monsoon": 1 if r["season"] == "monsoon" else 0,
                "is_winter_fog": 1 if r["season"] == "winter_fog" else 0,
                "is_festival_surge": 1 if r["season"] == "festival_surge" else 0,
                "is_single_line_route": 1 if single_line_frac_by_train.get(tn, 0.0) > 0 else 0,
                "delay_min": float(r["delay_min"]),
                "train_no": tn,
                "season": r["season"],
            })
    return rows


def main():
    rows = load_features()
    feature_names = ["single_line_frac", "distance_km", "is_monsoon",
                      "is_winter_fog", "is_festival_surge", "is_single_line_route"]

    X = np.array([[r[f] for f in feature_names] for r in rows])
    y = np.array([r["delay_min"] for r in rows])

    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    model = lgb.train(
        {"objective": "regression", "verbosity": -1, "min_data_in_leaf": 20},
        train_data,
        num_boost_round=100,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value

    print(f"Model base value (average predicted delay across all data): {base_value:.1f} min\n")

    # --- Pick a few interesting sample predictions to show top-3 reasons ---
    # one monsoon + single-line row, one normal + trunk row
    monsoon_idx = next(i for i, r in enumerate(rows)
                        if r["is_monsoon"] and r["is_single_line_route"])
    normal_idx = next(i for i, r in enumerate(rows)
                       if r["season"] == "normal" and not r["is_single_line_route"])

    for label, idx in [("MONSOON, single-line route sample", monsoon_idx),
                        ("NORMAL, double-line trunk sample", normal_idx)]:
        print(f"=== {label} (train {rows[idx]['train_no']}) ===")
        pred = model.predict(X[idx:idx+1])[0]
        actual = rows[idx]["delay_min"]
        print(f"Predicted delay: {pred:.1f} min | Actual (ground truth): {actual:.1f} min")

        contribs = list(zip(feature_names, shap_values[idx]))
        contribs.sort(key=lambda x: -abs(x[1]))
        print("Top 3 reasons:")
        for fname, val in contribs[:3]:
            sign = "+" if val >= 0 else "-"
            print(f"  {sign}{abs(val):.1f} min: {FEATURE_LABELS[fname]}")
        print()

    # --- Counterfactual: "if it weren't monsoon, what would ETA have been?" ---
    print("=== Counterfactual: monsoon sample, 'what if it weren't monsoon season?' ===")
    original_pred = model.predict(X[monsoon_idx:monsoon_idx+1])[0]

    cf_row = dict(zip(feature_names, X[monsoon_idx]))
    cf_row["is_monsoon"] = 0
    cf_X = np.array([[cf_row[f] for f in feature_names]])
    cf_pred = model.predict(cf_X)[0]

    delta = original_pred - cf_pred
    print(f"Predicted ETA delay with monsoon conditions:    {original_pred:.1f} min")
    print(f"Predicted ETA delay if NOT monsoon (all else equal): {cf_pred:.1f} min")
    print(f"-> Monsoon conditions are adding approximately {delta:.1f} min to this prediction.")
    print()

    print("=== Counterfactual: same sample, 'what if it weren't a single-line route?' ===")
    cf_row2 = dict(zip(feature_names, X[monsoon_idx]))
    cf_row2["is_single_line_route"] = 0
    cf_row2["single_line_frac"] = 0.0
    cf_X2 = np.array([[cf_row2[f] for f in feature_names]])
    cf_pred2 = model.predict(cf_X2)[0]
    delta2 = original_pred - cf_pred2
    print(f"Predicted ETA delay if double-line instead: {cf_pred2:.1f} min")
    print(f"-> Single-line exposure is adding approximately {delta2:.1f} min to this prediction.")


if __name__ == "__main__":
    main()