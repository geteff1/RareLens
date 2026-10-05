"""
Build prognosis stacking feature CSVs from multi-model outputs.

For each task (overall_outcome / functional_status / symptom_burden):
  <out-dir>/<task>/features.train.csv   — case_id, gt_label, feat_0, ..., feat_N
  <out-dir>/<task>/features.test.csv
  <out-dir>/<task>/meta.json            — base_model_names, labels, expl_keywords, expl_standardize

Usage:
    python -m rare_prognosis.training.build_features \\
        --results-root outputs/prognosis_demo/results \\
        --models-root outputs/prognosis_demo/llm_outputs \\
        --train-ids outputs/prognosis_demo/splits/train.json \\
        --test-ids outputs/prognosis_demo/splits/test.json \\
        --out-dir rare_prognosis/training/features
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

from data_io import (
    TASK_CONFIGS, DEFAULT_EXPL_KEYWORDS,
    load_json, get_nested, normalize_label, list_model_dirs, load_result_csv,
)
from ensemble_utils import encode_features, extract_text_features


def load_preds_and_expls(
    model_dirs: List[Path],
    case_ids: List[str],
    task: str,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, Dict[str, str]]]:
    cfg = TASK_CONFIGS[task]
    preds: Dict[str, Dict[str, Optional[str]]] = {m.name: {} for m in model_dirs}
    expls: Dict[str, Dict[str, str]] = {m.name: {} for m in model_dirs}
    for md in model_dirs:
        for cid in case_ids:
            obj = load_json(md / cid / "prognosis_prediction_output.json")
            if not isinstance(obj, dict):
                preds[md.name][cid] = None
                expls[md.name][cid] = ""
                continue
            preds[md.name][cid] = normalize_label(get_nested(obj, cfg.pred_path), task)
            raw = get_nested(obj, cfg.expl_path)
            expls[md.name][cid] = raw.strip() if isinstance(raw, str) else ""
    return preds, expls


def build_label_list(
    gt_by_id: Dict[str, str],
    preds: Dict[str, Dict[str, Optional[str]]],
    train_ids: List[str],
) -> List[str]:
    labels: set = set()
    for cid in train_ids:
        if cid in gt_by_id:
            labels.add(gt_by_id[cid])
        for mp in preds.values():
            pred = mp.get(cid)
            if pred is not None:
                labels.add(pred)
    return sorted(labels)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _compute_standardization(
    X: List[List[float]],
) -> Tuple[List[float], List[float]]:
    if not X or not X[0]:
        return [], []
    n, d = len(X), len(X[0])
    means = [sum(row[i] for row in X) / n for i in range(d)]
    stds = [(sum((row[i] - means[i]) ** 2 for row in X) / n) ** 0.5 for i in range(d)]
    stds = [s if s > 0 else 1.0 for s in stds]
    return means, stds


def _apply_standardization(
    X: List[List[float]], means: List[float], stds: List[float]
) -> List[List[float]]:
    return [
        [(float(v) - means[i]) / stds[i] for i, v in enumerate(row)]
        for row in X
    ]


def build_task_features(
    *,
    task: str,
    models_root: Path,
    train_ids: List[str],
    test_ids: List[str],
    gt_by_id: Dict[str, str],
    expl_keywords: List[str],
    out_dir: Path,
) -> None:
    all_ids = sorted(set(train_ids + test_ids))
    if not all_ids:
        raise SystemExit(f"[{task}] no case IDs")

    model_dirs = list_model_dirs(models_root, all_ids[0])
    if not model_dirs:
        raise SystemExit(f"[{task}] no model dirs found under {models_root}")

    base_model_names = sorted(d.name for d in model_dirs)
    logger.info("[%s] models=%d", task, len(base_model_names))

    preds, expls = load_preds_and_expls(model_dirs, all_ids, task)
    labels = build_label_list(gt_by_id, preds, train_ids)
    if not labels:
        raise SystemExit(f"[{task}] empty label list — check GT / pred dirs")

    train_ids_gt = [cid for cid in train_ids if cid in gt_by_id]
    test_ids_gt = [cid for cid in test_ids if cid in gt_by_id]
    if not train_ids_gt:
        raise SystemExit(f"[{task}] no GT for training cases")

    def _base_enc(ids: List[str]) -> List[List[float]]:
        return encode_features(
            ids,
            {m: {cid: preds[m].get(cid) for cid in ids} for m in base_model_names},
            labels,
        )

    def _expl_raw(ids: List[str]) -> Tuple[List[List[float]], List[str]]:
        rows: List[List[float]] = []
        cols: Optional[List[str]] = None
        for cid in ids:
            agg = "\n".join(
                expls.get(m, {}).get(cid, "") or ""
                for m in base_model_names
                if (expls.get(m, {}).get(cid, "") or "")
            )
            feats = extract_text_features(agg, expl_keywords)
            if cols is None:
                cols = list(feats.keys())
            rows.append([float(feats[c]) for c in (cols or [])])
        return rows, (cols or [])

    # Train features
    X_train_base = _base_enc(train_ids_gt)
    X_train_expl_raw, expl_cols = _expl_raw(train_ids_gt)
    means, stds = _compute_standardization(X_train_expl_raw)
    X_train_expl = _apply_standardization(X_train_expl_raw, means, stds)
    X_train = [list(b) + list(e) for b, e in zip(X_train_base, X_train_expl)]

    # Test features (standardized with train stats)
    X_test_base = _base_enc(test_ids_gt)
    X_test_expl_raw, _ = _expl_raw(test_ids_gt)
    X_test_expl = _apply_standardization(X_test_expl_raw, means, stds)
    X_test = [list(b) + list(e) for b, e in zip(X_test_base, X_test_expl)]

    col_names = [f"base_{m}_{l}" for m in base_model_names for l in labels] + expl_cols

    # Save
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(path: Path, ids: List[str], X: List[List[float]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "gt_label"] + col_names)
            for i, cid in enumerate(ids):
                w.writerow([cid, gt_by_id.get(cid, "")] + X[i])
        logger.info("  wrote %s (%d rows)", path, len(ids))

    _write_csv(task_dir / "features.train.csv", train_ids_gt, X_train)
    if test_ids_gt:
        _write_csv(task_dir / "features.test.csv", test_ids_gt, X_test)

    meta = {
        "base_model_names": base_model_names,
        "labels": labels,
        "expl_keywords": list(expl_keywords),
        "expl_standardize": {"means": means, "stds": stds},
        "col_names": col_names,
    }
    (task_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("  wrote meta: %s", task_dir / "meta.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build prognosis stacking feature CSVs.")
    p.add_argument("--results-root", required=True,
                   help="Directory containing <task>/result.csv files")
    p.add_argument("--models-root", required=True,
                   help="Root containing per-model LLM prognosis outputs")
    p.add_argument("--train-ids", required=True)
    p.add_argument("--test-ids", required=True)
    p.add_argument("--out-dir", default="rare_prognosis/training/features")
    p.add_argument("--task", choices=("overall_outcome", "functional_status", "symptom_burden", "all"), default="all")
    p.add_argument("--expl-keywords", nargs="*", default=DEFAULT_EXPL_KEYWORDS)
    args = p.parse_args()

    results_root = Path(args.results_root)
    models_root = Path(args.models_root)
    out_dir = Path(args.out_dir)
    train_allow = set(load_json(Path(args.train_ids)) or [])
    test_allow = set(load_json(Path(args.test_ids)) or [])
    if not train_allow:
        raise SystemExit(f"train ids empty: {args.train_ids}")

    tasks = list(TASK_CONFIGS) if args.task == "all" else [args.task]
    for i, task in enumerate(tasks, 1):
        cfg = TASK_CONFIGS[task]
        result_path = results_root / cfg.result_csv[0] / cfg.result_csv[1]
        if not result_path.is_file():
            raise SystemExit(f"[{task}] missing result csv: {result_path}")
        result = load_result_csv(result_path, task)
        tr = [cid for cid in result.train_ids if cid in train_allow]
        te = [cid for cid in result.test_ids if cid in test_allow]
        gt = {cid: lbl for cid, lbl in result.gt_by_id.items() if cid in train_allow or cid in test_allow}
        logger.info("[%s] train=%d test=%d gt=%d", task, len(tr), len(te), len(gt))
        build_task_features(
            task=task,
            models_root=models_root,
            train_ids=tr,
            test_ids=te,
            gt_by_id=gt,
            expl_keywords=list(args.expl_keywords),
            out_dir=out_dir,
        )
        logger.info("[%s] done (%d/%d)", task, i, len(tasks))

    logger.info("Finished. Features saved to %s", out_dir)


if __name__ == "__main__":
    main()
