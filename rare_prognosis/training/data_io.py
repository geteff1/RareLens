"""
Shared I/O utilities and task configuration for prognosis training pipeline.

All task definitions, label normalization, JSON loading, and common helpers
are centralized here to avoid duplication across scripts.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """Unified task configuration for all prognosis sub-tasks."""
    name: str
    # GT extraction
    gt_section: Optional[str]       # None => root-level key; "quality_of_life" => nested
    gt_key: str                     # key inside the section (or root)
    # Prediction extraction
    pred_section: str               # e.g. "overall_outcome"
    pred_key: str                   # e.g. "outcome_category"
    expl_key: str = "explanation"   # key for explanation text
    # Label synonyms  {raw_lower: canonical}
    synonyms: Dict[str, str] = field(default_factory=dict)
    # Result CSV location  (subdir, filename)
    result_csv: Tuple[str, str] = ("", "")
    # Model bundle filename
    bundle: str = ""

    @property
    def pred_path(self) -> Tuple[str, str]:
        return (self.pred_section, self.pred_key)

    @property
    def expl_path(self) -> Tuple[str, str]:
        return (self.pred_section, self.expl_key)


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "overall_outcome": TaskConfig(
        name="overall_outcome",
        gt_section=None,
        gt_key="overall_outcome",
        pred_section="overall_outcome",
        pred_key="outcome_category",
        synonyms={"death": "terminal", "progressed": "progression"},
        result_csv=("overall", "result.csv"),
        bundle="overall_outcome_C2_stacking_gbdt.pkl",
    ),
    "functional_status": TaskConfig(
        name="functional_status",
        gt_section="quality_of_life",
        gt_key="functional_status",
        pred_section="functional_status",
        pred_key="status",
        synonyms={},
        result_csv=("functional", "result.csv"),
        bundle="functional_status_C2_stacking_gbdt.pkl",
    ),
    "symptom_burden": TaskConfig(
        name="symptom_burden",
        gt_section="quality_of_life",
        gt_key="symptom_burden",
        pred_section="symptom_burden",
        pred_key="burden",
        synonyms={},
        result_csv=("symptom", "result.csv"),
        bundle="symptom_burden_C2_stacking_gbdt.pkl",
    ),
}

DEFAULT_EXPL_KEYWORDS: List[str] = [
    "progression", "progress", "stable", "stabil", "terminal",
    "death", "metast", "response", "improve", "worsen",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load a JSON file, returning None on error."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_nested(obj: Any, keys: Tuple[str, ...]) -> Any:
    """Traverse nested dicts by a sequence of keys."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def normalize_label(label: Any, task: str) -> Optional[str]:
    """Normalize a label string using task-specific synonyms."""
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    if not key:
        return None
    return TASK_CONFIGS[task].synonyms.get(key, key)


# ---------------------------------------------------------------------------
# Result CSV loading
# ---------------------------------------------------------------------------

@dataclass
class ResultData:
    """Parsed contents of a prognosis result CSV."""
    train_ids: List[str]
    test_ids: List[str]
    gt_by_id: Dict[str, str]
    pred_by_id: Dict[str, str]
    split_by_id: Dict[str, str]


def load_result_csv(path: Path, task: str) -> ResultData:
    """Load a prognosis result CSV and return all parsed fields."""
    train_ids: List[str] = []
    test_ids: List[str] = []
    gt_by_id: Dict[str, str] = {}
    pred_by_id: Dict[str, str] = {}
    split_by_id: Dict[str, str] = {}
    seen: set = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("case_id") or "").strip()
            split = str(row.get("split") or "").strip().lower()
            if not cid or split not in ("train", "test"):
                continue
            split_by_id[cid] = split
            gt = normalize_label(row.get("gt"), task)
            if gt is not None:
                gt_by_id[cid] = gt
            pred = normalize_label(row.get("prediction"), task)
            if pred is not None:
                pred_by_id[cid] = pred
            if cid not in seen:
                seen.add(cid)
                (train_ids if split == "train" else test_ids).append(cid)
    return ResultData(
        train_ids=train_ids,
        test_ids=test_ids,
        gt_by_id=gt_by_id,
        pred_by_id=pred_by_id,
        split_by_id=split_by_id,
    )


# ---------------------------------------------------------------------------
# CSV writing helper
# ---------------------------------------------------------------------------

def write_eval_csv(path: Path, rows: List[Dict]) -> None:
    """Write a list of row dicts to CSV, auto-detecting fieldnames."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote: {path}")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def list_model_dirs(
    models_root: Path,
    sample_case_id: str,
    ignore: Optional[set] = None,
) -> List[Path]:
    """Discover model directories that contain output for a sample case."""
    if ignore is None:
        ignore = {"RarePrognosis", "Output_prog", "Output_prog.zip"}
    out: List[Path] = []
    for p in sorted(models_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir() or p.name in ignore:
            continue
        if (p / sample_case_id / "prognosis_prediction_output.json").is_file():
            out.append(p)
    return out
