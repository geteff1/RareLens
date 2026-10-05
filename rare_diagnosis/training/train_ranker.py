#!/usr/bin/env python3
"""
Train diagnosis ranking model (XGBoost LTR).

Loads features CSV, performs GroupKFold cross-validation, and outputs
ensemble predictions.

Supports loading best hyperparameters from best_hyperopt_config.json.

Usage:
    python -m rare_diagnosis.training.train_ranker \
        --input-dir outputs/diagnosis_demo_primary/features \
        --out-dir outputs/diagnosis_demo_primary/model \
        --config rare_diagnosis/training/best_hyperopt_config_primary.json \
        --use-gpu
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monotonicity constraints (auto-inferred from feature names)
# ---------------------------------------------------------------------------

def get_monotone_constraints(feature_names: List[str]) -> Tuple[int, ...]:
    """Auto-infer monotonicity constraints from feature name semantics."""
    constraints = []
    for feat in feature_names:
        f = feat.lower()
        c = 0
        if "rank" in f and "std" not in f:
            c = -1
        elif "conf" in f and "confusion" not in f:
            c = 1
        elif "score" in f and "certainty" not in f:
            c = 1
        elif "sem_sim" in f or "r_sim" in f:
            c = 1
        elif "certainty_score" in f:
            c = 1
        elif "mean_reasoning_len" in f:
            c = 1
        elif "hit__" in f:
            c = 1
        elif "appear_count" in f:
            c = 1
        elif "agreement" in f:
            c = 1
        elif "kings_consensus" in f:
            c = 1
        elif "is_unique_candidate" in f:
            c = -1
        elif "ancestor_match" in f:
            c = 1
        constraints.append(c)
    return tuple(constraints)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

META_COLS = {
    "split", "case_id", "stage", "orphacode", "label",
    "diagnosis_name", "gt_matches_score_5", "gt_positive_count",
    "has_positive_label", "case_entropy",
}

MODEL_FEATURE_PREFIXES = ("rank__", "conf__", "hit__", "z_conf__", "r_sim__")


def is_model_feature(name: str) -> bool:
    return name.startswith(MODEL_FEATURE_PREFIXES)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    df.fillna(0, inplace=True)
    return df


def prep_data(df: pd.DataFrame, feat_cols: List[str]):
    df = df.sort_values(by=["case_id"])
    X = df[feat_cols]
    y = df["label"]
    group = df.groupby("case_id", sort=False).size().to_numpy()
    return X, y, group, df


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
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Load data
    train_path = os.path.join(args.input_dir, "features.train.csv")
    test_path = os.path.join(args.input_dir, "features.test.csv")
    logger.info("Loading: %s", train_path)
    df_train = load_data(train_path)
    logger.info("Loading: %s", test_path)
    df_test = load_data(test_path)

    # Feature columns
    available_feat_cols = [c for c in df_train.columns if c not in META_COLS]
    feat_cols = list(available_feat_cols)

    # Load best params from config if provided
    params: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        best_params = config.get("best_params", {})
        params.update(best_params)
        config_features = config.get("feature_names")
        if config_features:
            # Keep the tuned non-model feature subset, but take model-specific
            # columns from this run's dynamic feature schema. This allows a
            # user-selected model subset without retaining columns for models
            # that were not generated.
            available = set(df_train.columns)
            base_features = [
                f for f in config_features
                if f in available and f not in META_COLS and not is_model_feature(f)
            ]
            dynamic_model_features = [f for f in available_feat_cols if is_model_feature(f)]
            feat_cols = base_features + dynamic_model_features
        logger.info("Loaded config: %s (%d params, %d features)", args.config, len(best_params), len(feat_cols))

    # Override with CLI args
    if args.lr:
        params["learning_rate"] = args.lr
    if args.n_estimators:
        params["n_estimators"] = args.n_estimators

    constraints = get_monotone_constraints(feat_cols)
    logger.info("Feature count: %d", len(feat_cols))

    # Prepare test set
    X_test, _, _, sorted_test_df = prep_data(df_test, feat_cols)

    # GroupKFold training
    n_splits = args.n_splits
    gkf = GroupKFold(n_splits=n_splits)
    model_dir = os.path.join(args.out_dir, "models")
    results_dir = args.results_dir or os.path.join(args.out_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    test_preds = np.zeros(len(df_test))
    cv_scores = []

    logger.info("Training %d-fold GroupKFold on %d rows, %d features", n_splits, len(df_train), len(feat_cols))

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df_train, df_train["label"], groups=df_train["case_id"])
    ):
        logger.info("  Fold %d/%d: training...", fold + 1, n_splits)
        fold_train = df_train.iloc[train_idx].sort_values("case_id").reset_index(drop=True)
        fold_val = df_train.iloc[val_idx].sort_values("case_id").reset_index(drop=True)

        X_tr, y_tr, g_tr, _ = prep_data(fold_train, feat_cols)
        X_val, y_val, g_val, sorted_val = prep_data(fold_val, feat_cols)

        model = xgb.XGBRanker(
            objective=params.get("objective", "rank:ndcg"),
            tree_method="hist",
            device="cuda" if args.use_gpu else "cpu",
            eval_metric="ndcg@3",
            monotone_constraints=constraints,
            random_state=args.seed + fold,
            early_stopping_rounds=params.get("early_stopping_rounds", 100),
            n_estimators=params.get("n_estimators", 3000),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.85),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            gamma=params.get("gamma", 0.0),
            min_child_weight=params.get("min_child_weight", 1),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
        )

        model.fit(
            X_tr, y_tr, group=g_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[g_val],
            verbose=100,
        )

        # Save fold model
        model.get_booster().save_model(
            os.path.join(model_dir, f"xgboost_fold_{fold + 1}.json")
        )

        # Validation metrics
        val_pred = model.predict(X_val)
        sorted_val = sorted_val.copy()
        sorted_val["pred_score"] = val_pred
        sorted_val["rank"] = sorted_val.groupby("case_id")["pred_score"].rank(ascending=False, method="first")
        metrics = calculate_metrics(sorted_val)
        cv_scores.append(metrics["acc@1"])
        logger.info(
            "  Fold %d/%d: Acc@1=%.2f%% Acc@3=%.2f%% Acc@5=%.2f%% MRR=%.4f",
            fold + 1, n_splits,
            metrics["acc@1"] * 100, metrics["acc@3"] * 100,
            metrics["acc@5"] * 100, metrics["mrr"],
        )

        # Test predictions
        test_preds += model.predict(X_test)
        del model, X_tr, y_tr, g_tr, X_val, y_val, g_val
        gc.collect()

    logger.info("=" * 40)
    logger.info("Avg CV Acc@1: %.2f%% (Std: %.4f)", np.mean(cv_scores) * 100, np.std(cv_scores))

    # Save ensemble predictions as JSON
    test_preds /= n_splits
    sorted_test_df = sorted_test_df.copy()
    sorted_test_df["ensemble_score"] = test_preds
    sorted_test_df["rank"] = sorted_test_df.groupby("case_id")["ensemble_score"].rank(ascending=False, method="first")

    # Final test metrics
    test_metrics = calculate_metrics(sorted_test_df)
    logger.info(
        "Test Acc@1=%.2f%% Acc@3=%.2f%% Acc@5=%.2f%% MRR=%.4f",
        test_metrics["acc@1"] * 100, test_metrics["acc@3"] * 100,
        test_metrics["acc@5"] * 100, test_metrics["mrr"],
    )

    # Save predictions JSON
    json_results = {}
    for cid, grp in sorted_test_df.groupby("case_id"):
        gt_rows = grp[grp["label"] == 1]
        if len(gt_rows) > 0:
            gt_row = gt_rows.iloc[0]
            gt_info = {
                "orphacode": int(gt_row["orphacode"]),
                "diagnosis_name": str(gt_row.get("diagnosis_name", "")),
                "is_recalled": 1,
            }
        else:
            gt_info = {"orphacode": None, "diagnosis_name": "GT not in candidate pool", "is_recalled": 0}

        grp_sorted = grp.sort_values("ensemble_score", ascending=False)
        predictions = []
        for rank, (_, row) in enumerate(grp_sorted.iterrows(), 1):
            predictions.append({
                "rank": rank,
                "orphacode": int(row["orphacode"]),
                "diagnosis_name": str(row.get("diagnosis_name", "")),
                "score": float(row["ensemble_score"]),
                "label": int(row["label"]),
                "is_correct": int(row["label"] == 1),
            })
        json_results[str(cid)] = {"ground_truth": gt_info, "predictions": predictions}

    json_path = os.path.join(results_dir, "test_predictions_ranked.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # Also save as CSV
    csv_cols = ["case_id", "rank", "orphacode", "diagnosis_name", "ensemble_score", "label"]
    sorted_test_df[csv_cols].to_csv(
        os.path.join(results_dir, "test_predictions_ranked.csv"), index=False
    )

    logger.info("Finished. Models saved to %s; ranked results saved to %s", model_dir, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diagnosis ranking model (XGBoost).")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with features.train.csv and features.test.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for test_predictions_ranked.{json,csv} (default: <out-dir>/results)")
    parser.add_argument("--config", default=None,
                        help="Path to best_hyperopt_config.json for loading best params")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override n_estimators")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()
    train(args)
