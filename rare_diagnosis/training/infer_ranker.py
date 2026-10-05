#!/usr/bin/env python3
"""
Diagnosis XGBoost ranking inference (standalone).

Loads trained XGBoost fold models and a pre-built feature CSV,
produces ensemble-averaged ranked predictions.

Usage:
    python -m rare_diagnosis.training.infer_ranker \\
        --input-dir outputs/diagnosis_demo_primary/features \\
        --model-dir outputs/diagnosis_demo_primary/model/models \\
        --config rare_diagnosis/training/best_hyperopt_config_primary.json \\
        --out-dir outputs/diagnosis_demo_primary/model/inference
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

META_COLS = {
    "split", "case_id", "stage", "orphacode", "label",
    "diagnosis_name", "gt_matches_score_5", "gt_positive_count",
    "has_positive_label", "case_entropy",
}


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    df.fillna(0, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_fold_models(model_dir: str) -> List:
    """Load all current-format XGBoost fold models from a directory."""

    model_path = Path(model_dir)
    models = []

    for p in sorted(model_path.glob("xgboost_fold_*.json")):
        booster = xgb.Booster()
        booster.load_model(str(p))
        models.append(booster)
        logger.info("Loaded: %s", p.name)

    if not models:
        raise FileNotFoundError(
            f"No XGBoost fold models found in {model_dir}. "
            "Expected xgboost_fold_*.json"
        )

    logger.info("Loaded %d fold models", len(models))
    return models


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def calculate_metrics(predictions_df: pd.DataFrame, k_values: List[int] = None) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5]
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    total = 0

    for _, group in predictions_df.groupby("case_id"):
        total += 1
        gt_rows = group[group["label"] == 1]
        if len(gt_rows) == 0:
            continue
        found_rank = gt_rows["rank"].min()
        for k in k_values:
            if found_rank <= k:
                hits[k] += 1
        mrr_sum += 1.0 / found_rank

    results = {"total_cases": total}
    for k in k_values:
        results[f"acc@{k}"] = hits[k] / total if total > 0 else 0
    results["mrr"] = mrr_sum / total if total > 0 else 0
    return results


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def infer(args):
    # The config remains useful documentation for the training setup, but the
    # exact dynamic feature schema is stored in each trained XGBoost model.
    config_feat_cols = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        config_feat_cols = config.get("feature_names")
        logger.info("Loaded config: %s", args.config)

    # Load data (try features.test.csv first, then features.csv)
    df = None
    for fname in ["features.test.csv", "features.csv"]:
        fpath = os.path.join(args.input_dir, fname)
        if os.path.exists(fpath):
            logger.info("Loading: %s", fpath)
            df = load_data(fpath)
            break
    if df is None:
        raise FileNotFoundError(f"No feature CSV found in {args.input_dir}")

    # Load fold models
    models = load_fold_models(args.model_dir)

    # Prefer the schema embedded in the trained model. Fall back to the config
    # only for old model files that do not store feature names.
    feat_cols = list(models[0].feature_names or config_feat_cols or [])
    if not feat_cols:
        feat_cols = [c for c in df.columns if c not in META_COLS]
    for model in models[1:]:
        if model.feature_names and list(model.feature_names) != feat_cols:
            raise ValueError("Fold models contain inconsistent feature schemas")
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Feature CSV does not match the trained dynamic model schema; "
            f"missing columns: {missing}"
        )
    logger.info("  %d rows, %d features", len(df), len(feat_cols))

    # Sort by case_id for group consistency
    df = df.sort_values("case_id").reset_index(drop=True)
    X = df[feat_cols]

    # Ensemble prediction (average across folds)
    logger.info("Running inference with %d model(s)...", len(models))
    preds = np.zeros(len(df))
    dmat = xgb.DMatrix(X)
    for i, model in enumerate(models, 1):
        preds += model.predict(dmat)
        logger.info("  Model %d/%d predicted", i, len(models))
    preds /= len(models)

    df = df.copy()
    df["ensemble_score"] = preds
    df["rank"] = df.groupby("case_id")["ensemble_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    # Compute metrics if labels are available
    has_labels = "label" in df.columns and df["label"].sum() > 0
    if has_labels:
        metrics = calculate_metrics(df)
        logger.info(
            "Acc@1=%.2f%% Acc@3=%.2f%% Acc@5=%.2f%% MRR=%.4f (N=%d)",
            metrics["acc@1"] * 100, metrics["acc@3"] * 100,
            metrics["acc@5"] * 100, metrics["mrr"], metrics["total_cases"],
        )

    # Build output JSON
    json_results = {}
    for cid, grp in df.groupby("case_id"):
        gt_rows = grp[grp["label"] == 1] if has_labels else pd.DataFrame()
        if len(gt_rows) > 0:
            gt_row = gt_rows.iloc[0]
            gt_info = {
                "orphacode": int(gt_row["orphacode"]),
                "diagnosis_name": str(gt_row.get("diagnosis_name", "")),
                "is_recalled": 1,
            }
        else:
            gt_info = {
                "orphacode": None,
                "diagnosis_name": "GT not in candidate pool",
                "is_recalled": 0,
            }

        grp_sorted = grp.sort_values("ensemble_score", ascending=False)
        predictions = []
        for rank_val, (_, row) in enumerate(grp_sorted.iterrows(), 1):
            pred = {
                "rank": rank_val,
                "orphacode": int(row["orphacode"]),
                "diagnosis_name": str(row.get("diagnosis_name", "")),
                "score": float(row["ensemble_score"]),
            }
            if has_labels:
                pred["label"] = int(row["label"])
                pred["is_correct"] = int(row["label"] == 1)
            predictions.append(pred)

        json_results[str(cid)] = {
            "ground_truth": gt_info,
            "predictions": predictions,
        }

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "test_predictions_ranked.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", json_path)

    csv_cols = ["case_id", "rank", "orphacode", "diagnosis_name", "ensemble_score"]
    if has_labels:
        csv_cols.append("label")
    df.sort_values(["case_id", "rank"])[csv_cols].to_csv(
        os.path.join(args.out_dir, "test_predictions_ranked.csv"), index=False
    )

    logger.info("Finished. Results saved to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnosis XGBoost ranking inference.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with features CSV (features.test.csv or features.csv)")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing xgboost_fold_*.json files")
    parser.add_argument("--config", default=None,
                        help="Path to best_hyperopt_config.json for feature names")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    infer(args)
