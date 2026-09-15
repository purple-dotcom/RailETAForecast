"""
Trains three LightGBM quantile regressors (p10 / p50 / p90) predicting the
RESIDUAL on top of the deterministic baseline (see app/baseline.py), and
evaluates them against:
  1. The baseline alone (does the ML correction actually help?)
  2. Quantile coverage (does p10/p90 actually bracket ~80% of real outcomes?)

Saves trained models + the train-only section_stats they were trained
against to models/artifacts/, so the API loads a fixed, versioned set of
artifacts rather than retraining on every request.
"""

import os
import json
import lightgbm as lgb
import numpy as np

from build_training_table import build_train_test
from features import FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def to_matrix(X_dicts):
    return np.array([[row[f] for f in FEATURE_NAMES] for row in X_dicts])


def train_quantile_models(X_train, y_train):
    X = to_matrix(X_train)
    y = np.array(y_train)
    models = {}
    for name, alpha in QUANTILES.items():
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
        model = lgb.train(
            {
                "objective": "quantile", "alpha": alpha,
                "verbosity": -1, "min_data_in_leaf": 30,
                "num_leaves": 15, "learning_rate": 0.05,
            },
            train_data,
            num_boost_round=200,
        )
        models[name] = model
    return models


def evaluate(models, X_test, y_test, meta_test):
    X = to_matrix(X_test)
    preds = {name: m.predict(X) for name, m in models.items()}

    actual = np.array([m["actual_delay_min"] for m in meta_test])
    baseline = np.array([m["baseline_delay_min"] for m in meta_test])

    baseline_mae = np.mean(np.abs(actual - baseline))
    corrected_p50 = baseline + preds["p50"]
    model_mae = np.mean(np.abs(actual - corrected_p50))

    corrected_p10 = baseline + preds["p10"]
    corrected_p90 = baseline + preds["p90"]
    coverage_80 = np.mean((actual >= corrected_p10) & (actual <= corrected_p90))
    above_p90_frac = np.mean(actual > corrected_p90)
    below_p10_frac = np.mean(actual < corrected_p10)

    print("=== Evaluation on held-out test period ===")
    print(f"Baseline-only MAE:        {baseline_mae:.2f} min")
    print(f"Baseline + p50 correction MAE: {model_mae:.2f} min")
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    print(f"Improvement: {improvement:+.1f}%")
    print()
    print(f"p10-p90 coverage (target ~80%): {coverage_80:.1%}")
    print(f"  fraction actual > p90 (target ~10%): {above_p90_frac:.1%}")
    print(f"  fraction actual < p10 (target ~10%): {below_p10_frac:.1%}")

    print("\n=== MAE improvement by season (checking WHERE the model earns its keep) ===")
    seasons = sorted(set(m["season"] for m in meta_test))
    season_breakdown = []
    for season in seasons:
        idx = [i for i, m in enumerate(meta_test) if m["season"] == season]
        if not idx:
            continue
        a = actual[idx]
        base = baseline[idx]
        corr = corrected_p50[idx]
        b_mae = np.mean(np.abs(a - base))
        m_mae = np.mean(np.abs(a - corr))
        pct = (b_mae - m_mae) / b_mae * 100 if b_mae > 0 else 0.0
        print(f"  {season:16s} n={len(idx):5d}  baseline_MAE={b_mae:5.2f}  model_MAE={m_mae:5.2f}  improvement={pct:+5.1f}%")
        season_breakdown.append({"season": season, "n": len(idx), "baseline_mae": round(float(b_mae), 2),
                                  "model_mae": round(float(m_mae), 2), "improvement_pct": round(float(pct), 1)})

    return {
        "baseline_mae": float(baseline_mae),
        "model_mae": float(model_mae),
        "improvement_pct": float(improvement),
        "coverage_80": float(coverage_80),
        "above_p90_frac": float(above_p90_frac),
        "below_p10_frac": float(below_p10_frac),
        "n_test_rows": len(y_test),
        "season_breakdown": season_breakdown,
    }


def save_artifacts(models, section_stats, fallback_stats, eval_metrics):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    for name, model in models.items():
        model.save_model(os.path.join(ARTIFACTS_DIR, f"residual_{name}.txt"))

    # section_stats keys are tuples -- JSON needs string keys.
    section_stats_json = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in section_stats.items()}
    with open(os.path.join(ARTIFACTS_DIR, "section_stats.json"), "w") as f:
        json.dump({"section_stats": section_stats_json, "fallback_stats": fallback_stats}, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"\nSaved 3 models + section_stats + eval_metrics + feature_names to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    data = build_train_test()
    print(f"Train dates: {data['n_train_dates']}   Test dates: {data['n_test_dates']} (stratified)")
    print(f"Train rows: {len(data['X_train'])}   Test rows: {len(data['X_test'])}\n")

    models = train_quantile_models(data["X_train"], data["y_train"])
    metrics = evaluate(models, data["X_test"], data["y_test"], data["meta_test"])
    save_artifacts(models, data["section_stats"], data["fallback_stats"], metrics)