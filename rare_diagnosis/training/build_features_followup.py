#!/usr/bin/env python3
"""
Build diagnosis ranking features for the FOLLOW-UP consultation stage.

Same as primary, plus:
  - Query text includes the complete follow_up_consultation.json record.
  - Model outputs are merged from primary and follow-up diagnosis files.

Elite-Weighted Feature Engineering (V4.4 - Relaxed Recall & Greedy Labeling).

Output: features.{train|test}.csv, groups.{train|test}.csv

Usage:
    python build_features_followup.py \\
        --query_root data_500 \\
        --primary_models_root outputs/diagnosis_demo_followup/llm_outputs \\
        --score_root outputs/diagnosis_demo_followup/judge_scores \\
        --out_dir outputs/diagnosis_demo_followup/features \\
        --train_ids outputs/diagnosis_demo_followup/splits/train.json \\
        --test_ids outputs/diagnosis_demo_followup/splits/test.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import math
import logging
import os
import numpy as np
import concurrent.futures
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from functools import partial
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# ── Elite configuration ──────────────────────────────────────────────────────

ELITE_MODELS_CONFIG = {
    # Kings (high weight)
    "GPT-5": 10.0,
    "o3-mini": 8.0,
    # Knights (medium weight)
    "GPT-3.5-Turbo": 6.0,
    "Gemini-2.5-Flash": 6.0,
    "DeepSeek-R1": 5.5,
    "Qwen3-235B-Instruct": 4.5,
    "Claude-Haiku-4.5": 3.0,
    # Pawns (low weight)
    "Qwen3-8B": 1.0,
    "Qwen3-14B": 1.0,
    "Qwen3-32B": 1.0,
    "GPT-4o-mini": 1.0,
}

KINGS_LIST = ["GPT-5", "o3-mini"]

# ── Logging & regex ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
RE_DIGIT = re.compile(r"\d+")
RE_ALPHANUM = re.compile(r"[^a-z0-9\s]")

# ── Semantic model (per-worker) ──────────────────────────────────────────────

_GLOBAL_SEMANTIC_MODEL = None
_worker_counter = None  # mp.Value shared counter for deterministic GPU assignment


def _init_worker_counter(counter):
    """Store the shared counter in each worker process."""
    global _worker_counter
    _worker_counter = counter


def init_worker(model_name: str, num_gpus: int):
    global _GLOBAL_SEMANTIC_MODEL
    if not HAS_SENTENCE_TRANSFORMERS:
        return
    try:
        # Use a shared atomic counter so workers get IDs 0, 1, 2, ...
        with _worker_counter.get_lock():
            worker_id = _worker_counter.value
            _worker_counter.value += 1
        gpu_id = worker_id % num_gpus if num_gpus > 0 else 0
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
        _GLOBAL_SEMANTIC_MODEL = SentenceTransformer(model_name, device=device)
        _GLOBAL_SEMANTIC_MODEL.encode("Warmup sequence.", show_progress_bar=False)
        logger.info("Worker %d initialized on %s", worker_id, device)
    except Exception as e:
        logger.error("Worker initialization failed: %s", e)


def _pool_initializer(counter, model_name, num_gpus):
    """Top-level initializer for ProcessPoolExecutor (must be picklable on Windows)."""
    _init_worker_counter(counter)
    init_worker(model_name, num_gpus)


# ── Ontology manager ────────────────────────────────────────────────────────

class OntologyManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, json_path: Optional[str] = None):
        if not hasattr(self, "data"):
            self.data = {}
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        self.data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load ontology: {e}")

    def extract_features(self, cand_code: int, query_text_lower: str) -> Dict[str, float]:
        if cand_code == -1:
            return {"ont_depth": 0.0, "ont_is_leaf": 0.0, "ont_ancestor_match": 0.0, "ont_num_parents": 0.0}

        info = self.data.get(str(cand_code), {})
        depth = float(info.get("depth", 0.0))
        is_leaf = 1.0 if info.get("is_leaf", False) else 0.0

        ancestor_match = 0.0
        parent_names = info.get("parent_names", [])
        for pname in parent_names:
            if pname and len(pname) > 3 and pname.lower() in query_text_lower:
                ancestor_match = 1.0
                break

        return {
            "ont_depth": depth,
            "ont_is_leaf": is_leaf,
            "ont_ancestor_match": ancestor_match,
            "ont_num_parents": float(len(parent_names)),
        }


# ── Utilities ────────────────────────────────────────────────────────────────

def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def clean_orphacode(val: Any) -> int:
    if val is None:
        return -1
    nums = RE_DIGIT.findall(str(val).strip())
    return int(nums[0]) if nums else -1


def normalize_name(text: str) -> str:
    if not text:
        return ""
    t = RE_ALPHANUM.sub(" ", text.lower())
    return " ".join(t.split())


def get_text_complexity_features(text: str) -> Dict[str, float]:
    if not text:
        return {"txt_len": 0, "txt_certainty": 0, "txt_uncertainty": 0}
    text_lower = text.lower()
    words = text_lower.split()
    certainty_markers = ["definitely", "confirms", "diagnostic of", "classic", "pathognomonic", "strong evidence"]
    uncertainty_markers = ["possible", "probable", "might", "could be", "rule out", "unlikely", "differential"]
    c_count = sum(text_lower.count(m) for m in certainty_markers)
    u_count = sum(text_lower.count(m) for m in uncertainty_markers)
    return {"txt_len": float(len(words)), "txt_certainty": float(c_count), "txt_uncertainty": float(u_count)}


def parse_model_output(model_obj: Any) -> List[Dict]:
    """Relaxed parser: accepts items even if Orphacode is missing."""
    if isinstance(model_obj, dict):
        for k in ["output", "result", "answer", "final_answer", "data"]:
            if isinstance(model_obj.get(k), (dict, list)):
                model_obj = model_obj[k]
                break

    items = model_obj if isinstance(model_obj, list) else []
    if isinstance(model_obj, dict):
        keys = sorted([k for k in model_obj.keys() if "diagnosis" in k.lower()], key=str)
        items = [model_obj[k] for k in keys if isinstance(model_obj[k], dict)]

    out = []
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue
        raw_oc = it.get("orphacode") or it.get("ORPHAcode") or it.get("orpha")
        oc = clean_orphacode(raw_oc)
        name = str(it.get("diagnosis_name") or it.get("disease_name") or "").strip()
        if oc == -1 and len(name) < 2:
            continue
        conf = float(it.get("confidence_score") or it.get("score") or it.get("confidence") or 0.0)
        rsn = str(it.get("diagnostic_reasoning") or it.get("reasoning") or "").strip()
        out.append({"orphacode": oc, "diagnosis_name": name, "confidence": conf, "rank": i, "reasoning": rsn})
    return out


# ── Follow-up specific helpers ───────────────────────────────────────────────

def flatten_json_to_text(data: Any) -> str:
    """Turn nested follow-up record content into a flat text string."""
    texts: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float)):
                texts.append(f"{k}: {v}")
            elif isinstance(v, (dict, list)):
                nested = flatten_json_to_text(v)
                if nested:
                    texts.append(f"{k}: {nested}")
    elif isinstance(data, list):
        texts.extend(filter(None, (flatten_json_to_text(item) for item in data)))
    return ". ".join(texts)


# ── Per-case worker ──────────────────────────────────────────────────────────

def process_single_case(
    case_id: str,
    *,
    split_name: str,
    args_dict: Dict,
    available_model_dirs: List[str],
) -> Tuple[List[Dict], List[Dict]]:

    ont_manager = OntologyManager(args_dict["ontology_path"])

    # 1. Load primary and follow-up patient data.
    q_root = Path(args_dict["query_root"])
    p_path = q_root / case_id / "primary_consultation.json"
    if not p_path.is_file():
        return [], []

    primary_obj = read_json(p_path)
    cc = primary_obj.get("medical_history", {}).get("chief_complaint", "")
    hpi = primary_obj.get("medical_history", {}).get("history_of_present_illness", "")

    followup_path = q_root / case_id / "follow_up_consultation.json"
    followup_text = ""
    if followup_path.is_file():
        followup_text = flatten_json_to_text(read_json(followup_path))
    sem_query_text = (
        f"Chief Complaint: {cc}. History: {hpi}. Follow-up Record: {followup_text}"
    ).strip()

    # Demographics
    pat_info = primary_obj.get("patient_info") or primary_obj.get("basic_information", {})
    age_raw = pat_info.get("age")
    pat_age = -1
    if age_raw is not None:
        age_str = str(age_raw)
        if age_str.isdigit():
            pat_age = int(age_str)
        elif "mo" in age_str.lower():
            pat_age = 0

    gender_raw = pat_info.get("gender")
    pat_is_male = -1
    if gender_raw:
        g = str(gender_raw).lower()
        if g in ("m", "male", "boy", "man"):
            pat_is_male = 1
        elif g in ("f", "female", "girl", "woman"):
            pat_is_male = 0

    negations = [" no ", " not ", " none ", " deny ", " denies ", " without "]
    neg_count = sum(sem_query_text.lower().count(n) for n in negations)
    input_word_len = len(sem_query_text.split())
    input_sent_count = sem_query_text.count(".") + sem_query_text.count(";")

    # 2. Load ground truth (union of all score files, evaluation_score == 5)
    gt_positive_names: Set[str] = set()
    gt_root_path = Path(args_dict["score_root"])

    score_files: List[Path] = []
    direct = gt_root_path / case_id / args_dict["score_fname"]
    if direct.is_file():
        score_files.append(direct)
    else:
        try:
            for item in gt_root_path.iterdir():
                if item.is_dir():
                    sub = item / case_id / args_dict["score_fname"]
                    if sub.is_file():
                        score_files.append(sub)
        except Exception:
            pass

    for sf in score_files:
        gt_data = read_json(sf)
        mld = gt_data.get("most_likely_diagnosis", {})
        items_to_check = mld.values() if isinstance(mld, dict) else (mld if isinstance(mld, list) else [])
        for val in items_to_check:
            if not isinstance(val, dict):
                continue
            try:
                score_int = int(val.get("evaluation_score") or 0)
            except (ValueError, TypeError):
                score_int = 0
            if score_int == 5:
                norm = normalize_name(str(val.get("diagnosis_name", "")))
                if norm:
                    gt_positive_names.add(norm)

    # 3. Build candidate pool (relaxed keying, merged primary + follow-up)
    pool: Dict[str, Dict] = {}
    target_models = [m for m in available_model_dirs if m in ELITE_MODELS_CONFIG]
    model_root = Path(args_dict["primary_models_root"])
    if not target_models:
        return [], []

    reasoning_texts: List[str] = []
    diagnosis_names: List[str] = []
    text_mapping: List[Dict] = []
    all_votes_orphacodes: List[int] = []

    for mn in target_models:
        mfp = model_root / mn / case_id / args_dict["primary_fname"]
        wmfp = model_root / mn / case_id / args_dict["followup_fname"]
        if not mfp.is_file() or not wmfp.is_file():
            continue

        # Pool candidates from both stages. Parse them separately so each stage
        # retains its original rank instead of pairing unrelated diagnosisN keys.
        observations = parse_model_output(read_json(mfp))
        observations.extend(parse_model_output(read_json(wmfp)))

        for obs in observations:
            oc = obs["orphacode"]
            name_raw = obs["diagnosis_name"]

            if oc != -1:
                key = f"OC_{oc}"
                all_votes_orphacodes.append(oc)
            else:
                norm_n = normalize_name(name_raw)
                if not norm_n:
                    continue
                key = f"NAME_{norm_n}"

            if key not in pool:
                pool[key] = {
                    "orphacode": oc,
                    "diagnosis_name": name_raw,
                    "all_names": set(),
                    "per_model": {},
                }

            if name_raw:
                pool[key]["all_names"].add(normalize_name(name_raw))
            if len(name_raw) > len(pool[key]["diagnosis_name"]):
                pool[key]["diagnosis_name"] = name_raw

            pool[key]["per_model"][mn] = {
                "rank": obs["rank"],
                "confidence": obs["confidence"],
                "reasoning": obs["reasoning"],
            }

            if obs["reasoning"] and len(obs["reasoning"]) > 10:
                text_mapping.append({"type": "reasoning", "key": key, "mn": mn})
                reasoning_texts.append(obs["reasoning"][:2000])

    # 3.1 Case-level vote entropy
    if all_votes_orphacodes:
        vote_counts = Counter(all_votes_orphacodes)
        total_votes = len(all_votes_orphacodes)
        case_entropy = -sum((c / total_votes) * math.log(c / total_votes + 1e-9) for c in vote_counts.values())
    else:
        case_entropy = 0.0

    # 4. Semantic encoding
    for key in list(pool.keys()):
        name = pool[key]["diagnosis_name"]
        if name:
            text_mapping.append({"type": "name", "key": key, "mn": None})
            diagnosis_names.append(name)

    if _GLOBAL_SEMANTIC_MODEL and sem_query_text:
        q_emb = _GLOBAL_SEMANTIC_MODEL.encode(sem_query_text[:3000], convert_to_tensor=True, show_progress_bar=False)
        all_texts = reasoning_texts + diagnosis_names
        if all_texts:
            all_embs = _GLOBAL_SEMANTIC_MODEL.encode(all_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            cos_scores = util.cos_sim(q_emb, all_embs)[0].cpu().numpy()
            cursor = 0
            for _ in reasoning_texts:
                meta = text_mapping[cursor]
                pool[meta["key"]]["per_model"][meta["mn"]]["sem_sim_reasoning"] = float(cos_scores[cursor])
                cursor += 1
            for _ in diagnosis_names:
                meta = text_mapping[cursor]
                pool[meta["key"]]["sem_sim_name"] = float(cos_scores[cursor])
                cursor += 1

    # 5. Feature aggregation
    all_confidences = [info["confidence"] for cand in pool.values() for info in cand["per_model"].values()]
    mean_conf_case = np.mean(all_confidences) if all_confidences else 0.0
    std_conf_case = np.std(all_confidences) if all_confidences else 1.0
    if std_conf_case == 0:
        std_conf_case = 1.0

    valid_cands = []
    for key, cand in pool.items():
        w_score = 0.0
        w_conf_sum = 0.0
        total_weight = 0.0
        ranks, reasoning_sims, txt_lens, txt_certainties = [], [], [], []
        kings_hit_top1 = 0
        kings_votes = 0

        for mn, info in cand["per_model"].items():
            weight = ELITE_MODELS_CONFIG.get(mn, 0.0)
            rank = info["rank"]
            conf = info["confidence"]
            ranks.append(rank)
            rsn_feats = get_text_complexity_features(info.get("reasoning", ""))
            txt_lens.append(rsn_feats["txt_len"])
            txt_certainties.append(rsn_feats["txt_certainty"] - rsn_feats["txt_uncertainty"])
            r_sim = info.get("sem_sim_reasoning", 0.0)
            if r_sim != 0.0:
                reasoning_sims.append(r_sim)
            w_score += weight * (1.0 / (rank + 0.5))
            w_conf_sum += weight * conf
            total_weight += weight
            if mn in KINGS_LIST:
                kings_votes += 1
                if rank == 1:
                    kings_hit_top1 = 1

        mean_w_conf = w_conf_sum / total_weight if total_weight > 0 else 0.0
        rank_std = float(np.std(ranks)) if len(ranks) > 1 else 0.0

        oc_val = cand["orphacode"]
        ont_feats = ont_manager.extract_features(oc_val, sem_query_text.lower())

        # Greedy label: any alias matches GT → label=1
        is_correct = int(any(alias in gt_positive_names for alias in cand["all_names"]))

        if oc_val != -1:
            oc_log_val = math.log1p(oc_val)
            oc_bucket_1000 = oc_val // 1000
            oc_last_digit = oc_val % 10
        else:
            oc_log_val = -1.0
            oc_bucket_1000 = -1
            oc_last_digit = -1

        diag_name = cand.get("diagnosis_name", "Unknown")

        row = {
            "split": split_name, "case_id": case_id,
            "orphacode": oc_val, "diagnosis_name": diag_name,
            "gt_positive_count": len(gt_positive_names), "label": is_correct,
            "oc_log_val": oc_log_val, "oc_bucket_1000": oc_bucket_1000, "oc_last_digit": oc_last_digit,
            "name_word_len": len(diag_name.split()),
            "name_is_syndrome": 1 if "syndrome" in diag_name.lower() else 0,
            "pat_age": pat_age, "pat_is_male": pat_is_male,
            "input_word_len": input_word_len, "input_sent_count": input_sent_count,
            "negation_count": neg_count, "case_vote_entropy": case_entropy,
            "weighted_score": w_score, "weighted_mean_conf": mean_w_conf, "rank_std": rank_std,
            "appear_count_elite": len(ranks),
            "agreement_ratio": len(ranks) / len(target_models),
            "is_unique_candidate": 1 if len(ranks) == 1 else 0,
            "kings_consensus": 1 if kings_votes == len(KINGS_LIST) else 0,
            "sem_sim_name": cand.get("sem_sim_name", 0.0),
            "sem_sim_reasoning_max": max(reasoning_sims) if reasoning_sims else 0.0,
            "sem_sim_reasoning_mean": float(np.mean(reasoning_sims)) if reasoning_sims else 0.0,
            "is_king_top1": kings_hit_top1,
            "mean_reasoning_len": float(np.mean(txt_lens)) if txt_lens else 0.0,
            "mean_certainty_score": float(np.mean(txt_certainties)) if txt_certainties else 0.0,
            **ont_feats,
        }

        for mn in target_models:
            if mn in cand["per_model"]:
                info = cand["per_model"][mn]
                row[f"rank__{mn}"] = info["rank"]
                row[f"conf__{mn}"] = info["confidence"]
                row[f"hit__{mn}"] = 1
                row[f"z_conf__{mn}"] = (info["confidence"] - mean_conf_case) / std_conf_case
                row[f"r_sim__{mn}"] = info.get("sem_sim_reasoning", 0.0)
            else:
                row[f"rank__{mn}"] = 999
                row[f"conf__{mn}"] = 0.0
                row[f"hit__{mn}"] = 0
                row[f"z_conf__{mn}"] = -3.0
                row[f"r_sim__{mn}"] = 0.0

        valid_cands.append(row)

    if not valid_cands:
        return [], []

    group_rows = [{
        "split": split_name, "case_id": case_id,
        "candidate_count": len(valid_cands),
        "has_positive_label": 1 if any(r["label"] == 1 for r in valid_cands) else 0,
        "case_entropy": case_entropy,
    }]
    return valid_cands, group_rows


# ── CSV headers ──────────────────────────────────────────────────────────────

BASE_HEADERS = [
    "split", "case_id", "orphacode", "diagnosis_name",
    "gt_positive_count", "label",
    "oc_log_val", "oc_bucket_1000", "oc_last_digit",
    "name_word_len", "name_is_syndrome",
    "pat_age", "pat_is_male", "input_word_len", "input_sent_count", "negation_count",
    "case_vote_entropy",
    "weighted_score", "weighted_mean_conf", "rank_std",
    "appear_count_elite", "agreement_ratio", "is_unique_candidate", "kings_consensus",
    "sem_sim_name", "sem_sim_reasoning_max", "sem_sim_reasoning_mean",
    "is_king_top1", "mean_reasoning_len", "mean_certainty_score",
    "ont_depth", "ont_is_leaf", "ont_ancestor_match", "ont_num_parents",
]

GROUP_HEADERS = ["split", "case_id", "candidate_count", "has_positive_label", "case_entropy"]


def build_model_headers(model_names: List[str]) -> List[str]:
    return [
        f"{sfx}{mn}"
        for mn in model_names
        for sfx in ("rank__", "conf__", "hit__", "z_conf__", "r_sim__")
    ]


def discover_active_models(
    models_root: Path,
    score_root: Path,
    case_ids: List[str],
    primary_fname: str,
    followup_fname: str,
    score_fname: str,
    requested_models: Optional[List[str]] = None,
) -> List[str]:
    """Models with both-stage predictions and at least one follow-up score."""
    active: List[str] = []
    if not models_root.is_dir():
        return active
    requested = set(requested_models or [])
    for model_dir in sorted((p for p in models_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        name = model_dir.name
        if requested and name not in requested:
            continue
        if name not in ELITE_MODELS_CONFIG:
            continue
        has_prediction = any(
            (model_dir / cid / primary_fname).is_file()
            and (model_dir / cid / followup_fname).is_file()
            for cid in case_ids
        )
        has_score = any((score_root / name / cid / score_fname).is_file() for cid in case_ids)
        if has_prediction and has_score:
            active.append(name)
    return active


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build diagnosis ranking features (follow-up stage).")
    ap.add_argument("--query_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--score_root", required=True,
                    help="Root containing per-model LLM-judge score files")
    ap.add_argument("--primary_models_root", required=True)
    ap.add_argument("--semantic_model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs for semantic model (0 = CPU only)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel worker processes (each loads a model copy; "
                         "keep <= 2 * num_gpus to avoid GPU OOM)")
    ap.add_argument("--ontology_path", default="rare_diagnosis/training/orphanet_hierarchy.json")
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--test_ids", required=True)
    ap.add_argument("--primary_fname", default="most_likely_diagnosis_orphacode.json")
    ap.add_argument("--followup_fname", default="followup_diagnosis_orphacode.json")
    ap.add_argument("--score_fname", default="followup_diagnosis_score.json")
    ap.add_argument("--models", default="",
                    help="Comma-separated model tags to consider (default: discover from roots)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def load_ids(p):
        with open(p) as f:
            return [str(x) for x in json.load(f)]

    train_ids = load_ids(args.train_ids)
    test_ids = load_ids(args.test_ids)
    all_ids = sorted(set(train_ids + test_ids))
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    avail_models = discover_active_models(
        Path(args.primary_models_root), Path(args.score_root), all_ids,
        args.primary_fname, args.followup_fname, args.score_fname, requested_models,
    )
    if not avail_models:
        raise SystemExit("No active follow-up diagnosis models with predictions and judge scores.")
    all_headers = BASE_HEADERS + build_model_headers(avail_models)
    logger.info("Active models (%d): %s", len(avail_models), avail_models)
    args_dict = vars(args)

    def run_split(split, ids):
        if not ids:
            return
        f_path = Path(args.out_dir) / f"features.{split}.csv"
        g_path = Path(args.out_dir) / f"groups.{split}.csv"

        f_csv = open(f_path, "w", newline="", encoding="utf-8")
        g_csv = open(g_path, "w", newline="", encoding="utf-8")
        w_f = csv.DictWriter(f_csv, fieldnames=all_headers, extrasaction="ignore")
        w_g = csv.DictWriter(g_csv, fieldnames=GROUP_HEADERS, extrasaction="ignore")
        w_f.writeheader()
        w_g.writeheader()

        func = partial(process_single_case, split_name=split, args_dict=args_dict, available_model_dirs=avail_models)
        total_rows = total_pos = 0

        if args.workers and args.workers > 0:
            # Use 'spawn' to avoid CUDA-in-fork deadlocks on Linux. Create the
            # shared counter from the SAME context — Python 3.12 forbids sharing
            # a fork-context SemLock with a spawn-context process.
            mp_ctx = mp.get_context("spawn")
            counter = mp_ctx.Value("i", 0)

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp_ctx,
                initializer=_pool_initializer,
                initargs=(counter, args.semantic_model, args.num_gpus),
            ) as executor:
                futures = {executor.submit(func, cid): cid for cid in ids}
                it = (
                    tqdm(concurrent.futures.as_completed(futures), total=len(ids), desc=split, ncols=100)
                    if tqdm else concurrent.futures.as_completed(futures)
                )
                for fut in it:
                    try:
                        rows, grps = fut.result()
                        for x in rows:
                            w_f.writerow(x)
                            total_rows += 1
                            if x["label"] == 1:
                                total_pos += 1
                        for x in grps:
                            w_g.writerow(x)
                    except Exception as e:
                        msg = f"Error processing case {futures[fut]}: {e}"
                        (tqdm.write(msg) if tqdm else logger.error(msg))
        else:
            # Serial, in-process (no subprocess pool). Use --workers 0 on hosts
            # where spawned workers crash loading the model; the model is loaded
            # once in this process, which is known to work where spawn does not.
            global _worker_counter
            _worker_counter = mp.Value("i", 0)
            init_worker(args.semantic_model, args.num_gpus)
            it = tqdm(ids, total=len(ids), desc=split, ncols=100) if tqdm else ids
            for cid in it:
                try:
                    rows, grps = func(cid)
                    for x in rows:
                        w_f.writerow(x)
                        total_rows += 1
                        if x["label"] == 1:
                            total_pos += 1
                    for x in grps:
                        w_g.writerow(x)
                except Exception as e:
                    msg = f"Error processing case {cid}: {e}"
                    (tqdm.write(msg) if tqdm else logger.error(msg))

        f_csv.close()
        g_csv.close()

        if total_rows:
            pct = total_pos / total_rows * 100
            logger.info("[%s] rows=%d positives=%d (%.4f%%)\n  -> %s", split.upper(), total_rows, total_pos, pct, f_path)
        else:
            logger.warning("[%s] no rows generated", split.upper())

    logger.info("Building FOLLOW-UP features (V4.4 Relaxed Recall + Greedy Label) on %d GPUs...", args.num_gpus)
    run_split("train", train_ids)
    run_split("test", test_ids)
    logger.info("Finished. Features saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
