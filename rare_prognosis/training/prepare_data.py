#!/usr/bin/env python3
"""
Prepare data for the prognosis stacking pipeline.

Converts raw case outputs and LLM predictions into the directory structure
expected by build_features.py, train_models.py, and infer_models.py.

Inputs:
  - case_root/<case_id>/prognosis_new.json  (GT labels)
  - llm/<model>/<case_id>/prognosis_prediction_output.json

Outputs (into --result-root):
  {overall,functional,symptom}/result.csv

Usage:
    python prepare_data.py \\
        --case-root data_500 \\
        --llm-root outputs/prognosis_demo/llm_outputs \\
        --result-root outputs/prognosis_demo/results \\
        --train-ids outputs/prognosis_demo/splits/train.json \\
        --test-ids outputs/prognosis_demo/splits/test.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import TASK_CONFIGS


def _load_gt_from_prognosis_new(
    case_root: Path,
    case_ids: List[str],
) -> Dict[str, Dict[str, str]]:
    """Extract GT labels per task from <case_root>/<case_id>/prognosis_new.json.

    Structure:
      { "overall_outcome": "...",
        "quality_of_life": { "functional_status": "...", "symptom_burden": "..." }, ... }
    """
    gt: Dict[str, Dict[str, str]] = {t: {} for t in TASK_CONFIGS}
    for cid in case_ids:
        path = case_root / cid / "prognosis_new.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        label = obj.get("overall_outcome")
        if isinstance(label, str) and label.strip():
            gt["overall_outcome"][cid] = label.strip()
        qol = obj.get("quality_of_life", {})
        if isinstance(qol, dict):
            for key in ("functional_status", "symptom_burden"):
                label = qol.get(key)
                if isinstance(label, str) and label.strip():
                    gt[key][cid] = label.strip()
    return gt


def _write_result_csv(
    path: Path,
    train_ids: List[str],
    test_ids: List[str],
    gt: Dict[str, str],
) -> None:
    """Write a result CSV with case_id, split, prediction(empty), gt, correct, method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "prediction", "gt", "correct", "method"])
        for split, case_ids in (("train", train_ids), ("test", test_ids)):
            for cid in case_ids:
                label = gt.get(cid, "")
                w.writerow([cid, split, "", label, "", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for prognosis stacking pipeline.")
    parser.add_argument("--case-root", required=True,
                        help="Root of case data (contains <case_id>/prognosis_new.json)")
    parser.add_argument("--llm-root", required=True,
                        help="Root of LLM outputs (contains <model>/<case_id>/prognosis_prediction_output.json)")
    parser.add_argument("--result-root", required=True,
                        help="Directory where task result CSVs are written")
    parser.add_argument("--train-ids", default=None,
                        help="Optional JSON array defining the training split")
    parser.add_argument("--test-ids", default=None,
                        help="Optional JSON array defining the test split")
    args = parser.parse_args()

    if bool(args.train_ids) != bool(args.test_ids):
        parser.error("--train-ids and --test-ids must be provided together")

    case_root = Path(args.case_root)
    llm_root = Path(args.llm_root)
    result_root = Path(args.result_root)

    # Discover case IDs from LLM output dirs
    model_dirs = sorted([p for p in llm_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise SystemExit(f"No model directories found under {llm_root}")

    case_ids_set: set = set()
    for md in model_dirs:
        for p in md.iterdir():
            if p.is_dir() and (p / "prognosis_prediction_output.json").is_file():
                case_ids_set.add(p.name)
    case_ids = sorted(case_ids_set)
    if not case_ids:
        raise SystemExit("No case IDs found")

    if args.train_ids:
        with Path(args.train_ids).open("r", encoding="utf-8") as f:
            train_ids = [str(x) for x in json.load(f)]
        with Path(args.test_ids).open("r", encoding="utf-8") as f:
            test_ids = [str(x) for x in json.load(f)]
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise SystemExit(f"Train/test overlap: {sorted(overlap)}")
        missing = (set(train_ids) | set(test_ids)) - case_ids_set
        if missing:
            raise SystemExit(f"Split cases missing from LLM outputs: {sorted(missing)}")
        case_ids = train_ids + test_ids
    else:
        train_ids = case_ids
        test_ids = []
    logger.info("Found %d cases: %s", len(case_ids), case_ids)
    logger.info("Found %d models: %s", len(model_dirs), [m.name for m in model_dirs])

    # 1. Extract GT labels from prognosis_new.json
    gt = _load_gt_from_prognosis_new(case_root, case_ids)

    for task in gt:
        logger.info("  [%s] GT labels: %d", task, len(gt[task]))

    # 2. Create initial result CSVs, one per prognosis task.
    for task, cfg in TASK_CONFIGS.items():
        subdir, fname = cfg.result_csv
        result_path = result_root / subdir / fname
        _write_result_csv(result_path, train_ids, test_ids, gt[task])
        logger.info("  result CSV: %s", result_path)

    logger.info("Finished. Results saved to %s", result_root)


if __name__ == "__main__":
    main()
