#!/usr/bin/env python3
"""
Step 3: Train XGBoost Learning-to-Rank model for treatment reranking.

Uses GroupKFold cross-validation and ensemble prediction on the test set.

Usage:
    python train_ranker.py \\
        --data-dir /data/features \\
        --out-dir /data/models/treatment \\
        --objective rank:ndcg \\
        --save-models
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if "data_io" in sys.modules and not sys.modules["data_io"].__file__.startswith(str(THIS_DIR)):
    del sys.modules["data_io"]

from data_io import load_features_csv, export_ranked_json, save_predictions_csv


# ---------------------------------------------------------------------------
# Feature group management & ablation
# ---------------------------------------------------------------------------

def parse_group_list(text: str) -> set[str]:
    if not text:
        return set()
    out = set()
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.add(x)
    return out


def group_feature_columns(feat_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "rank_model": [],
        "importance_model": [],
        "hit_model": [],
        "model_support": [],
        "semantic_nli": [],
        "stage3_eval": [],
        "consensus_stat": [],
        "importance_agg": [],
        "rationale_text": [],
        # Fine-grained auxiliary feature groups
        "aux_sem_sim": [],
        "aux_nli": [],
        "aux_soft_consensus": [],
        "aux_rationale_text": [],
        "aux_stage3_coverage": [],
        "aux_stage3_appropriate": [],
        "aux_stage3_performed": [],
        "aux_stage3_completeness": [],
        "aux_stage3_helpfulness": [],
        "aux_stage3_safety": [],
        "aux_stage3_quality": [],
        "other": [],
    }

    consensus_cols = {
        "case_type_entropy", "pos_count_union", "weighted_rank_score",
        "rank_std", "appear_count", "agreement_ratio",
        "kings_consensus", "is_king_top1",
    }
    semantic_cols = {"sem_sim_candidate", "feat_nli_entailment", "feat_soft_consensus"}
    importance_agg_cols = {"mean_importance", "weighted_mean_importance"}
    rationale_cols = {"rationale_len_mean", "rationale_certainty_mean"}
    model_support_cols = {
        "model_support_weight_sum", "model_support_weight_ratio",
        "model_support_weight_mean", "model_support_weight_max",
        "topk_support_count", "topk_support_ratio",
        "rank_min", "rank_max", "rank_mean", "rank_median",
        "rank_top1_count", "rank_top3_count", "rank_top5_count",
        "imp_min", "imp_max", "imp_std", "w_mean_rank",
    }

    def add(col: str, *names: str) -> None:
        for n in names:
            groups[n].append(col)

    for c in feat_cols:
        if c.startswith("rank__"):
            add(c, "rank_model")
        elif c.startswith("imp__"):
            add(c, "importance_model")
        elif c.startswith("hit__"):
            add(c, "hit_model")
        elif c == "sem_sim_candidate":
            add(c, "semantic_nli", "aux_sem_sim")
        elif c == "feat_nli_entailment":
            add(c, "semantic_nli", "aux_nli")
        elif c == "feat_soft_consensus":
            add(c, "semantic_nli", "aux_soft_consensus")
        elif c == "feat_eval_scored_models" or c == "feat_eval_coverage_ratio":
            add(c, "stage3_eval", "aux_stage3_coverage")
        elif c == "feat_eval_appropriate_ratio":
            add(c, "stage3_eval", "aux_stage3_appropriate")
        elif c == "feat_eval_performed_ratio" or c == "feat_eval_performed_any":
            add(c, "stage3_eval", "aux_stage3_performed")
        elif c == "feat_eval_completeness_mean":
            add(c, "stage3_eval", "aux_stage3_completeness")
        elif c == "feat_eval_helpfulness_mean":
            add(c, "stage3_eval", "aux_stage3_helpfulness")
        elif c == "feat_eval_safety_mean":
            add(c, "stage3_eval", "aux_stage3_safety")
        elif c == "feat_eval_quality_mean" or c == "feat_eval_quality_std":
            add(c, "stage3_eval", "aux_stage3_quality")
        elif c.startswith("feat_eval_"):
            add(c, "stage3_eval")
        elif c in semantic_cols:
            add(c, "semantic_nli")
        elif c in consensus_cols:
            add(c, "consensus_stat")
        elif c in importance_agg_cols:
            add(c, "importance_agg")
        elif c in rationale_cols:
            add(c, "rationale_text", "aux_rationale_text")
        elif c in model_support_cols:
            add(c, "model_support")
        else:
            add(c, "other")
    return groups


def select_features_with_ablation(
    feat_cols: list[str],
    drop_groups: set[str],
) -> tuple[list[str], dict[str, list[str]], set[str]]:
    groups = group_feature_columns(feat_cols)

    expanded = set(drop_groups)
    if "all_model_specific" in expanded:
        expanded.update({"rank_model", "importance_model", "hit_model"})
        expanded.remove("all_model_specific")

    unknown = sorted(g for g in expanded if g not in groups)
    if unknown:
        raise SystemExit(
            f"Unknown feature groups in --drop-feature-groups: {unknown}. "
            f"Valid groups: {sorted(groups.keys()) + ['all_model_specific']}"
        )

    drop_cols: set[str] = set()
    for g in expanded:
        drop_cols.update(groups[g])

    selected = [c for c in feat_cols if c not in drop_cols]
    if not selected:
        raise SystemExit("All features were dropped by ablation. Keep at least one feature.")
    return selected, groups, expanded


def filter_model_specific_features(
    feat_cols: list[str],
    keep_models: list[str] | None,
) -> list[str]:
    if not keep_models:
        return feat_cols
    keep = {m.strip() for m in keep_models if m.strip()}
    if not keep:
        return feat_cols

    out = []
    for c in feat_cols:
        if c.startswith("rank__") or c.startswith("imp__") or c.startswith("hit__"):
            name = c.split("__", 1)[1]
            if name in keep:
                out.append(c)
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby("case_id", sort=False).size().to_numpy()


def _oversample_subset(df_subset: pd.DataFrame, multiplier: float, random_state: int) -> list[pd.DataFrame]:
    if multiplier <= 1.0 or df_subset.empty:
        return []
    parts: list[pd.DataFrame] = []
    int_part = int(np.floor(multiplier))
    frac_part = float(multiplier - int_part)
    extra_full_copies = max(0, int_part - 1)
    for _ in range(extra_full_copies):
        parts.append(df_subset.copy())
    if frac_part > 0:
        sampled = df_subset.sample(frac=frac_part, replace=False, random_state=random_state)
        if not sampled.empty:
            parts.append(sampled)
    return parts


def oversample_hard_negatives(
    df: pd.DataFrame,
    rank_col: str | None,
    topk: int,
    hard_neg_weight: float,
) -> pd.DataFrame:
    if not rank_col or hard_neg_weight <= 1.0:
        return df
    if rank_col not in df.columns:
        raise SystemExit(f"{rank_col} not found in training features for hard negative mining")

    rank_vals = pd.to_numeric(df[rank_col], errors="coerce").fillna(999.0)
    hard_neg = (df["label"] == 0) & (rank_vals <= max(1, int(topk)))
    hard_df = df.loc[hard_neg].copy()
    if hard_df.empty:
        return df

    parts = [df]
    parts.extend(_oversample_subset(hard_df, hard_neg_weight, 42))
    out = pd.concat(parts, ignore_index=True)
    return out


def _series_or_zeros(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=float), index=df.index, dtype=float)


def build_train_relevance(
    df: pd.DataFrame,
    use_aux_supervision: bool,
    objective: str,
) -> np.ndarray:
    y_bin = (
        pd.to_numeric(df["label"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.int32)
    )
    if not use_aux_supervision:
        return y_bin
    if objective == "rank:map":
        return y_bin

    completeness = _series_or_zeros(df, "feat_eval_completeness_mean").clip(lower=0.0, upper=5.0).to_numpy(dtype=float)
    helpfulness = _series_or_zeros(df, "feat_eval_helpfulness_mean").clip(lower=0.0, upper=5.0).to_numpy(dtype=float)
    safety = _series_or_zeros(df, "feat_eval_safety_mean").clip(lower=0.0, upper=5.0).to_numpy(dtype=float)

    bonus = np.zeros_like(y_bin, dtype=np.int32)
    bonus += (completeness >= 4.0).astype(np.int32)
    bonus += (helpfulness >= 4.0).astype(np.int32)
    bonus += (safety >= 4.0).astype(np.int32)

    rel = y_bin + bonus * y_bin
    return rel.astype(np.int32)


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def can_use_cuda(force_cpu: bool) -> bool:
    if force_cpu:
        return False
    try:
        return bool(xgb.build_info().get("USE_CUDA"))
    except Exception:
        return False


def train_one_fold(
    df_train, df_val, feat_cols, y_train, y_val,
    objective, target_k, force_cpu,
):
    eval_metrics = [f"ndcg@{target_k}"]
    if target_k != 3:
        eval_metrics.append("ndcg@3")

    params = {
        "objective": objective,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "eval_metric": eval_metrics,
        "device": "cuda" if can_use_cuda(force_cpu) else "cpu",
    }
    g_train = get_groups(df_train)
    g_val = get_groups(df_val)
    model = xgb.XGBRanker(**params)
    model.fit(
        df_train[feat_cols], y_train,
        group=g_train,
        eval_set=[(df_val[feat_cols], y_val)],
        eval_group=[g_val],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VALID_OBJECTIVES = {"rank:ndcg", "rank:map", "rank:pairwise"}


def calculate_metrics(df: pd.DataFrame, scores: np.ndarray, k_values=(1, 3, 5)) -> dict:
    """Hit@k / MRR per case_id (candidates ranked by score desc).

    Mirrors rare_diagnosis.training.train_ranker.calculate_metrics: every case
    counts toward the denominator; cases with no positive label score 0.
    """
    work = df[["case_id", "label"]].copy()
    work["_score"] = np.asarray(scores, dtype=float)
    work["_rank"] = work.groupby("case_id")["_score"].rank(ascending=False, method="first")
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    total = 0
    for _, group in work.groupby("case_id"):
        total += 1
        gt = group[group["label"] == 1]
        if len(gt) == 0:
            continue
        found_rank = gt["_rank"].min()
        for k in k_values:
            if found_rank <= k:
                hits[k] += 1
        mrr_sum += 1.0 / found_rank
    res = {"total": total}
    for k in k_values:
        res[f"hit@{k}"] = hits[k] / total if total else 0.0
    res["mrr"] = mrr_sum / total if total else 0.0
    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Train XGBoost L2R with GroupKFold and output ensemble scores."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Dir with features_train.csv and features_test.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for ranked test predictions (default: <out-dir>/results)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of GroupKFold splits")
    parser.add_argument("--target-k", type=int, default=3,
                        help="Target cutoff K for training eval metric")
    parser.add_argument("--objective", type=str, default="rank:ndcg",
                        choices=sorted(VALID_OBJECTIVES),
                        help="XGBoost ranking objective")
    parser.add_argument("--drop-feature-groups", type=str, default="",
                        help="Comma-separated feature groups to drop for ablation")
    parser.add_argument("--use-aux-supervision", type=int, default=0, choices=[0, 1],
                        help="Use completeness/helpfulness/safety to build graded train relevance")
    parser.add_argument("--keep-models", type=str, default="",
                        help="Comma-separated model names to keep for model-specific features")
    parser.add_argument("--hard-neg-rank-col", type=str, default=None,
                        help="Rank column for hard negative mining, e.g. rank__gpt-5")
    parser.add_argument("--hard-neg-topk", type=int, default=3)
    parser.add_argument("--hard-neg-weight", type=float, default=1.0,
                        help="Hard negative oversampling weight (>1 enables mining)")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--save-models", action="store_true",
                        help="Save per-fold model files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir) if args.results_dir else out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_train, feat_cols = load_features_csv(data_dir / "features_train.csv")
    df_test, _ = load_features_csv(data_dir / "features_test.csv")

    drop_groups = parse_group_list(args.drop_feature_groups)
    feat_cols, grouped_cols, applied_drop_groups = select_features_with_ablation(feat_cols, drop_groups)
    keep_models = [m.strip() for m in str(args.keep_models).split(",") if m.strip()]
    feat_cols = filter_model_specific_features(feat_cols, keep_models)

    if "label" not in df_train.columns:
        raise SystemExit("features_train.csv must contain 'label' column")

    n_splits = max(2, int(args.n_splits))
    target_k = max(1, int(args.target_k))
    gkf = GroupKFold(n_splits=n_splits)

    test_pred_sum = np.zeros(len(df_test), dtype=float)
    avg_importance = np.zeros(len(feat_cols), dtype=float)

    logger.info("Training %d-fold GroupKFold on %d rows, %d features, objective=%s, target_k=%d",
                n_splits, len(df_train), len(feat_cols), args.objective, target_k)

    cv_hit1: list[float] = []
    cv_mrr: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, df_train["label"], df_train["case_id"])):
        df_fold_train = df_train.iloc[train_idx].copy().sort_values("case_id")
        df_fold_val = df_train.iloc[val_idx].copy().sort_values("case_id")
        df_fold_train = oversample_hard_negatives(
            df_fold_train, args.hard_neg_rank_col, args.hard_neg_topk, args.hard_neg_weight,
        ).sort_values("case_id")

        y_fold_train = build_train_relevance(df_fold_train, bool(args.use_aux_supervision), args.objective)
        y_fold_val = build_train_relevance(df_fold_val, bool(args.use_aux_supervision), args.objective)

        logger.info("  Fold %d/%d: train=%d val=%d ... training", fold + 1, n_splits, len(df_fold_train), len(df_fold_val))
        model = train_one_fold(
            df_fold_train, df_fold_val, feat_cols,
            y_fold_train, y_fold_val, args.objective, target_k, args.force_cpu,
        )
        val_scores = np.asarray(model.predict(df_fold_val[feat_cols]), dtype=float)
        vm = calculate_metrics(df_fold_val, val_scores)
        cv_hit1.append(vm["hit@1"])
        cv_mrr.append(vm["mrr"])
        logger.info("  Fold %d/%d: Hit@1=%.2f%% Hit@3=%.2f%% Hit@5=%.2f%% MRR=%.4f",
                    fold + 1, n_splits, vm["hit@1"] * 100, vm["hit@3"] * 100, vm["hit@5"] * 100, vm["mrr"])

        test_pred_sum += np.asarray(model.predict(df_test[feat_cols]), dtype=float)
        avg_importance += np.asarray(model.feature_importances_, dtype=float)

        if args.save_models:
            model_dir = out_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(model_dir / f"model_fold{fold + 1}.json")

    final_scores = test_pred_sum / n_splits
    avg_importance /= n_splits

    if cv_hit1:
        logger.info("Avg CV Hit@1: %.2f%% (Std: %.4f)", np.mean(cv_hit1) * 100, np.std(cv_hit1))
    tm = calculate_metrics(df_test, final_scores)
    logger.info("Test Hit@1=%.2f%% Hit@3=%.2f%% Hit@5=%.2f%% MRR=%.4f",
                tm["hit@1"] * 100, tm["hit@3"] * 100, tm["hit@5"] * 100, tm["mrr"])

    save_predictions_csv(
        df_test,
        final_scores,
        results_dir / "test_predictions_ensemble.csv",
        score_col="ensemble_score",
    )
    export_ranked_json(
        df_test,
        final_scores,
        results_dir / "ranked_results.json",
        score_col="ensemble_score",
    )


    fi_path = out_dir / "feature_importance_ensemble.csv"
    fi_df = pd.DataFrame({"feature": feat_cols, "importance": avg_importance})
    fi_df.sort_values(by="importance", ascending=False).to_csv(fi_path, index=False)

    logger.info("Finished. Models saved to %s; ranked results saved to %s", out_dir, results_dir)

    feat_used_path = out_dir / "feature_columns_used.txt"
    with feat_used_path.open("w", encoding="utf-8") as f:
        f.write(f"objective={args.objective}\n")
        f.write(f"target_k={int(target_k)}\n")
        f.write(f"n_splits={int(n_splits)}\n")
        f.write(f"drop_feature_groups={sorted(applied_drop_groups)}\n")
        f.write(f"keep_models={keep_models}\n")
        f.write(f"use_aux_supervision={bool(args.use_aux_supervision)}\n")
        f.write(f"hard_neg_rank_col={args.hard_neg_rank_col}\n")
        f.write(f"hard_neg_topk={int(args.hard_neg_topk)}\n")
        f.write(f"hard_neg_weight={float(args.hard_neg_weight)}\n")
        f.write(f"n_features={len(feat_cols)}\n")
        for g in sorted(grouped_cols.keys()):
            f.write(f"{g}_count={len(grouped_cols[g])}\n")
        f.write("\n[features]\n")
        for c in feat_cols:
            f.write(f"{c}\n")


if __name__ == "__main__":
    main()
