#!/usr/bin/env python3
"""
data_io.py -- Shared data loading / saving utilities for treatment ranking.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Shared regex constants
# ---------------------------------------------------------------------------

RE_DIGIT = re.compile(r"\d+")
RE_ALPHANUM = re.compile(r"[^a-z0-9\s]+")

YES_SET = {"yes", "y", "true", "1"}


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    t = s.lower()
    t = RE_ALPHANUM.sub(" ", t)
    t = " ".join(t.split())
    return t


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(str(x).strip())
        if math.isfinite(val):
            return val
        return default
    except Exception:
        return default


def extract_idx(key: str) -> int:
    m = RE_DIGIT.findall(str(key))
    return int(m[0]) if m else 10**9


def to_yes(v: Any) -> int:
    if v is None:
        return 0
    return 1 if str(v).strip().lower() in YES_SET else 0


def clip_score_1_to_5(v: Any) -> float:
    x = safe_float(v, 0.0)
    if x <= 0:
        return 0.0
    return float(max(1.0, min(5.0, x)))


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

# Row identifiers and text fields kept in the CSV for grouping and reporting.
ROW_META_COLS = {
    "case_id",
    "candidate_key",
    "treatment_type",
    "specific_treatment",
    "dosage_or_details",
    "treatment_rationale",
    "anticipated_treatment_response",
    "safety_considerations",
    "label",
}

# Analysis/supervision columns are deliberately retained in the generated CSV,
# but are not model inputs.  In particular, ``pos_count_union`` is derived from
# positive labels and must not enter training.  The support/rank summaries were
# not used by the leakage-safe 50-feature Treatment model schema.
ANALYSIS_META_COLS = {
    "pos_count_union",
    "model_support_weight_sum",
    "model_support_weight_ratio",
    "model_support_weight_mean",
    "model_support_weight_max",
    "topk_support_count",
    "topk_support_ratio",
    "rank_min",
    "rank_max",
    "rank_mean",
    "rank_median",
    "rank_top1_count",
    "rank_top3_count",
    "rank_top5_count",
    "imp_min",
    "imp_max",
    "imp_std",
    "w_mean_rank",
}

META_COLS = ROW_META_COLS | ANALYSIS_META_COLS
META_PREFIXES = ("feat_eval_",)

# Kept as a compatibility alias for callers that imported the old name.
IGNORE_COLS = META_COLS


def is_model_feature_column(column: str) -> bool:
    """Return whether a CSV column is eligible as an XGBoost input feature."""
    return column not in META_COLS and not column.startswith(META_PREFIXES)


def load_features_csv(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load feature CSV and infer model feature columns."""
    df = pd.read_csv(path, low_memory=False, dtype={"case_id": "string"})
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    feat_cols = [c for c in df.columns if is_model_feature_column(c)]
    return df, feat_cols


def export_ranked_json(
    df: pd.DataFrame,
    scores,
    out_path: Path,
    score_col: str = "score",
) -> None:
    """Group by case_id, sort by score desc, export minimal JSON."""
    work_df = df.copy()
    work_df[score_col] = scores

    meta_cols = ["candidate_key", "treatment_type", "specific_treatment"]
    keep_cols = ["case_id", "label", score_col] + [c for c in meta_cols if c in work_df.columns]
    work_df = work_df[keep_cols]

    output = []
    for case_id, group in work_df.groupby("case_id"):
        sorted_group = group.sort_values(by=score_col, ascending=False)
        candidates = []
        for _, row in sorted_group.iterrows():
            item = {
                "label": int(row.get("label", 0)),
                "score": float(row[score_col]),
            }
            if "candidate_key" in row:
                item["candidate_key"] = row["candidate_key"]
            if "specific_treatment" in row:
                item["specific_treatment"] = str(row["specific_treatment"])
            candidates.append(item)
        output.append({"case_id": str(case_id), "candidates": candidates})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def save_predictions_csv(
    df: pd.DataFrame,
    scores,
    out_path: Path,
    score_col: str = "score",
) -> None:
    keep = [c for c in ["case_id", "candidate_key", "label"] if c in df.columns]
    out_df = df[keep].copy()
    out_df[score_col] = scores
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
