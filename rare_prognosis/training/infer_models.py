"""
Offline stacking inference: load bundle pkl, build features for each case, write result CSV.

Usage:
    python -m rare_prognosis.training.infer_models \\
        --results-root outputs/prognosis_demo/results \\
        --models-root outputs/prognosis_demo/llm_outputs \\
        --train-ids outputs/prognosis_demo/splits/train.json \\
        --test-ids outputs/prognosis_demo/splits/test.json \\
        --models-dir outputs/prognosis_demo/model

Add --no-write to only evaluate without overwriting the result CSVs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import TASK_CONFIGS, load_json, load_result_csv, get_nested, normalize_label
from ensemble_utils import encode_features, extract_text_features


def _predict_with_fold_models(
    fold_models: list,
    X: list,
    class_list: List[str],
) -> Any:
    """Average predict_proba across fold models, return label for single sample."""
    import numpy as np

    if len(fold_models) == 1:
        pred = fold_models[0].predict(X)
        return _unwrap_pred(pred)

    n_classes = len(class_list)
    class_index = {c: i for i, c in enumerate(class_list)}
    proba_sum = np.zeros((len(X), n_classes), dtype=float)

    for m in fold_models:
        p = m.predict_proba(X)
        aligned = np.zeros((len(X), n_classes), dtype=float)
        for i, cls in enumerate(m.classes_):
            if cls in class_index:
                aligned[:, class_index[cls]] = p[:, i]
        proba_sum += aligned

    avg = proba_sum / len(fold_models)
    idx = int(np.argmax(avg[0]))
    return class_list[idx]


def _unwrap_pred(pred: Any) -> Any:
    if pred is None:
        return None
    if isinstance(pred, (list, tuple)):
        return _unwrap_pred(pred[0]) if pred else None
    try:
        import numpy as np
        if isinstance(pred, np.ndarray):
            if pred.size == 0:
                return None
            v = pred.reshape(-1)[0]
            return _unwrap_pred(v.item() if hasattr(v, "item") else v)
    except Exception:
        pass
    return pred


def _load_bundle(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Bundle deserialization failed (missing dependency): {path}\n"
            f"Error: {e}\n"
            "Ensure the same catboost/sklearn version is installed, or re-run train_models.py"
        ) from e
    if not isinstance(obj, dict) or "model" not in obj or "meta" not in obj:
        raise SystemExit(f"Invalid bundle format: {path}")
    return obj


# ---------------------------------------------------------------------------
# Feature building (single case)
# ---------------------------------------------------------------------------

def _build_features_for_case(
    *,
    case_id: str,
    task: str,
    models_root: Path,
    base_model_names: List[str],
    labels: List[str],
    expl_keywords: List[str],
    expl_means: List[float],
    expl_stds: List[float],
) -> Optional[List[float]]:
    spec = TASK_CONFIGS[task]
    preds_by_model: Dict[str, Optional[str]] = {}
    expl_by_model: Dict[str, str] = {}
    for m in base_model_names:
        obj = load_json(models_root / m / case_id / "prognosis_prediction_output.json")
        if not isinstance(obj, dict):
            preds_by_model[m] = None
            expl_by_model[m] = ""
            continue
        preds_by_model[m] = normalize_label(get_nested(obj, spec.pred_path), task)
        raw = get_nested(obj, spec.expl_path)
        expl_by_model[m] = raw.strip() if isinstance(raw, str) else ""

    X_base = encode_features(
        ["__one__"],
        {m: {"__one__": preds_by_model.get(m)} for m in base_model_names},
        labels,
    )
    if not X_base:
        return None
    row = list(X_base[0])

    if expl_keywords:
        agg = "\n".join(
            expl_by_model.get(m, "") for m in base_model_names if expl_by_model.get(m, "")
        )
        feats = extract_text_features(agg, expl_keywords)
        expl_row = [float(feats[c]) for c in feats]
        if expl_means and expl_stds and len(expl_means) == len(expl_row):
            expl_row = [
                (float(v) - float(expl_means[i])) / (float(expl_stds[i]) if float(expl_stds[i]) != 0 else 1.0)
                for i, v in enumerate(expl_row)
            ]
        row.extend(expl_row)
    return row


def _write_result_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case_id", "split", "prediction", "gt", "correct", "method"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Per-task inference
# ---------------------------------------------------------------------------

def infer_task(
    *,
    task: str,
    results_root: Path,
    models_root: Path,
    models_dir: Path,
    train_ids: List[str],
    test_ids: List[str],
    no_write: bool,
) -> None:
    spec = TASK_CONFIGS[task]
    result_path = results_root / spec.result_csv[0] / spec.result_csv[1]
    bundle_path = models_dir / spec.bundle

    if not result_path.is_file():
        raise SystemExit(f"[{task}] missing result csv: {result_path}")
    if not bundle_path.is_file():
        raise SystemExit(f"[{task}] missing bundle: {bundle_path}")

    result = load_result_csv(result_path, task)
    split_by = result.split_by_id
    gt_by = result.gt_by_id
    task_train = [cid for cid in train_ids if cid in gt_by]
    task_test = [cid for cid in test_ids if cid in gt_by]
    all_ids = task_train + task_test

    bundle = _load_bundle(bundle_path)
    model = bundle["model"]
    meta = bundle["meta"]
    base_model_names = list(meta.get("base_model_names") or [])
    labels = list(meta.get("labels") or [])
    expl_keywords = list(meta.get("expl_keywords") or [])
    expl_std = meta.get("expl_standardize") or {}
    means = list(expl_std.get("means") or [])
    stds = list(expl_std.get("stds") or [])

    # Support both single model and fold model list
    class_list = list(meta.get("class_list") or [])
    if isinstance(model, list):
        fold_models = model
    else:
        fold_models = [model]

    pred_by: Dict[str, str] = {}
    missing_feat = 0
    t0 = time.time()
    for cid in all_ids:
        x = _build_features_for_case(
            case_id=cid,
            task=task,
            models_root=models_root,
            base_model_names=base_model_names,
            labels=labels,
            expl_keywords=expl_keywords,
            expl_means=means,
            expl_stds=stds,
        )
        if x is None:
            missing_feat += 1
            continue
        try:
            pred = _predict_with_fold_models(fold_models, [x], class_list)
        except Exception:
            continue
        if pred is not None:
            pred_by[cid] = str(pred).strip()

    def _acc(ids: List[str]) -> Tuple[float, int, int]:
        c = t = 0
        for cid in ids:
            gt = gt_by.get(cid)
            if gt is None:
                continue
            p = pred_by.get(cid)
            if p is None:
                continue
            t += 1
            if p == gt:
                c += 1
        return (c / t) if t else 0.0, c, t

    test_acc, test_c, test_n = _acc(task_test)
    logger.info(
        "[%s] train=%d test=%d missing_feat=%d elapsed=%.1fs | test_acc=%.6f (%d/%d)",
        task, len(task_train), len(task_test),
        missing_feat, time.time() - t0,
        test_acc, test_c, test_n,
    )

    if no_write:
        return

    out_rows: List[Dict] = []
    for cid in all_ids:
        sp = split_by.get(cid)
        if sp not in ("train", "test"):
            continue
        gt = gt_by.get(cid, "")
        pred = pred_by.get(cid, "")
        correct = ""
        if gt and pred:
            norm_pred = normalize_label(pred, task)
            norm_gt = normalize_label(gt, task)
            correct = "True" if norm_pred == norm_gt else "False"
        out_rows.append(dict(
            case_id=cid, split=sp, prediction=pred, gt=gt,
            correct=correct, method=bundle_path.stem,
        ))
    _write_result_csv(result_path, out_rows)
    logger.info("[OK] wrote: %s", result_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prognosis stacking offline inference.")
    p.add_argument("--results-root", required=True,
                   help="Directory containing task result CSVs")
    p.add_argument("--models-root", required=True,
                   help="Root containing per-model LLM prognosis outputs")
    p.add_argument("--train-ids", required=True)
    p.add_argument("--test-ids", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--no-write", action="store_true", help="Only evaluate, do not overwrite result CSVs")
    p.add_argument("--task", choices=("overall_outcome", "functional_status", "symptom_burden", "all"), default="all")
    args = p.parse_args()

    train_ids = sorted(set(load_json(Path(args.train_ids)) or []))
    test_ids = sorted(set(load_json(Path(args.test_ids)) or []))
    if not train_ids:
        raise SystemExit(f"train ids empty: {args.train_ids}")

    tasks = list(TASK_CONFIGS) if args.task == "all" else [args.task]
    for task in tasks:
        infer_task(
            task=task,
            results_root=Path(args.results_root),
            models_root=Path(args.models_root),
            models_dir=Path(args.models_dir),
            train_ids=train_ids,
            test_ids=test_ids,
            no_write=args.no_write,
        )
    logger.info("Finished. Inference complete.")


if __name__ == "__main__":
    main()
