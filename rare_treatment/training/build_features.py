#!/usr/bin/env python3
"""
Step 2: Build Learning-to-Rank features for treatment reranking.

Pools candidates from multi-model LLM outputs, computes 50+ features per
candidate (consensus, semantic similarity, NLI entailment, rationale text,
Stage-3 evaluation auxiliaries), and writes per-split feature CSVs.

Multi-GPU support: each GPU shard loads its own embedding + NLI model.

Usage:
    python build_features.py \\
        --case_root /data/cases \\
        --llm_root /data/llm_outputs \\
        --score_root /data/judge_scores \\
        --train_ids dataset/train_cases.json \\
        --test_ids dataset/test_cases.json \\
        --out_dir /data/features \\
        --num_gpus 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from collections import Counter
import multiprocessing as mp

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
# Ensure we import data_io from this package, not a same-named sibling module
if "data_io" in sys.modules and not sys.modules["data_io"].__file__.startswith(str(THIS_DIR)):
    del sys.modules["data_io"]

try:
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    HAS_ST = True
except Exception:
    HAS_ST = False

from data_io import (
    RE_DIGIT, RE_ALPHANUM, YES_SET,
    read_json, normalize_text, safe_int, safe_float, extract_idx, to_yes,
    clip_score_1_to_5,
)

DEFAULT_WEIGHT_STRATEGY = "coverage"
DEFAULT_TOPK_MODELS = 3
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"


def resolve_local_model_reference(model_name: str) -> str:
    """Resolve a Hub model ID to its cached snapshot when offline mode is active."""
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate.resolve())
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    if not offline:
        return model_name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Local Hugging Face snapshot is unavailable for {model_name!r}. "
            "Pre-cache the model or disable offline mode."
        ) from exc


def list_subdirs(path: Path) -> List[str]:
    if not path.is_dir():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


def entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    ent = 0.0
    for v in counts.values():
        p = v / total
        ent -= p * math.log(p + 1e-12)
    return ent


def shard_list(xs: List[str], n: int, idx: int) -> List[str]:
    return xs[idx::n]


# ---------------------------------------------------------------------------
# Model weight computation
# ---------------------------------------------------------------------------

def get_active_models(out_root: Path, score_root: Path) -> List[str]:
    models_out = set(list_subdirs(out_root))
    models_score = set(list_subdirs(score_root))
    return sorted(models_out & models_score)


def compute_model_weights(
    active_models: List[str],
    case_ids: List[str],
    out_root: Path,
    output_fname: str,
    strategy: str,
) -> Dict[str, float]:
    if not active_models:
        return {}
    if strategy == "uniform":
        return {m: 1.0 for m in active_models}

    counts = {m: 0 for m in active_models}
    total_cases = len(case_ids)
    for case_id in case_ids:
        for m in active_models:
            fp = out_root / m / case_id / output_fname
            if fp.exists():
                counts[m] += 1
    weights = {m: (counts[m] / max(1, total_cases)) for m in active_models}
    return weights


def pick_top_models(model_weights: Dict[str, float], k: int) -> List[str]:
    if not model_weights:
        return []
    items = [(m, w) for m, w in model_weights.items() if w > 0]
    if not items:
        return []
    items.sort(key=lambda x: (-x[1], x[0]))
    return [m for m, _ in items[:max(1, k)]]


# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------

def build_headers(model_names: List[str]) -> List[str]:
    base_headers = [
        "case_id", "candidate_key", "treatment_type", "specific_treatment",
        "dosage_or_details", "treatment_rationale", "anticipated_treatment_response", "safety_considerations",
        "label",
        "case_type_entropy", "pos_count_union",
        "weighted_rank_score", "rank_std", "appear_count", "agreement_ratio",
        "kings_consensus", "is_king_top1",
        "mean_importance", "weighted_mean_importance",
        "model_support_weight_sum", "model_support_weight_ratio",
        "model_support_weight_mean", "model_support_weight_max",
        "topk_support_count", "topk_support_ratio",
        "rank_min", "rank_max", "rank_mean", "rank_median",
        "rank_top1_count", "rank_top3_count", "rank_top5_count",
        "imp_min", "imp_max", "imp_std",
        "w_mean_rank",
        "sem_sim_candidate",
        "rationale_len_mean", "rationale_certainty_mean",
        "feat_nli_entailment",
        "feat_soft_consensus",
        # Stage-3 structured evaluation features (candidate-level aggregates)
        "feat_eval_scored_models",
        "feat_eval_coverage_ratio",
        "feat_eval_appropriate_ratio",
        "feat_eval_performed_ratio",
        "feat_eval_performed_any",
        "feat_eval_completeness_mean",
        "feat_eval_helpfulness_mean",
        "feat_eval_safety_mean",
        "feat_eval_quality_mean",
        "feat_eval_quality_std",
    ]
    model_headers = []
    for mn in model_names:
        model_headers.extend([f"rank__{mn}", f"imp__{mn}", f"hit__{mn}"])
    return base_headers + model_headers


# ---------------------------------------------------------------------------
# Query text construction
# ---------------------------------------------------------------------------

def build_query_text(plan_obj: Dict[str, Any]) -> str:
    mh = plan_obj.get("medical_history", {}) if isinstance(plan_obj, dict) else {}
    cc = str(mh.get("chief_complaint", "")).strip()
    hpi = str(mh.get("history_of_present_illness", "")).strip()

    dx = plan_obj.get("diagnosis", {}) if isinstance(plan_obj, dict) else {}
    final_dx = str(dx.get("final_diagnosis", "")).strip()
    reasoning = str(dx.get("diagnostic_reasoning", "")).strip()

    text = f"Diagnosis: {final_dx}. Reasoning: {reasoning}. Chief complaint: {cc}. HPI: {hpi}"
    return text.strip()


# ---------------------------------------------------------------------------
# Treatment plan output parsing
# ---------------------------------------------------------------------------

def parse_treatment_plan_output(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    recs = obj.get("treatment_recommendations", {})
    if not isinstance(recs, dict):
        return []

    out: List[Dict[str, Any]] = []
    keys = sorted(recs.keys(), key=lambda k: extract_idx(k))
    for k in keys:
        v = recs.get(k, {})
        if not isinstance(v, dict):
            continue
        spec = str(v.get("specific_treatment", "")).strip()
        ttype = str(v.get("treatment_type", "")).strip()
        if len(spec) < 2:
            continue
        out.append({
            "old_key": str(k),
            "rank": extract_idx(k),
            "treatment_type": ttype,
            "specific_treatment": spec,
            "dosage_or_details": str(v.get("dosage_or_details", "")).strip(),
            "treatment_rationale": str(v.get("treatment_rationale", "")).strip(),
            "importance_score": safe_int(v.get("importance_score", 0), default=0),
            "anticipated_treatment_response": str(v.get("anticipated_treatment_response", "")).strip(),
            "safety_considerations": str(v.get("safety_considerations", "")).strip(),
        })
    return out


# ---------------------------------------------------------------------------
# Stage-3 evaluation score parsing
# ---------------------------------------------------------------------------

def parse_treatment_score(score_obj: Any) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(score_obj, dict):
        return out
    root = score_obj.get("suggested_treatment_score", {})
    if not isinstance(root, dict):
        return out
    for tk, tv in root.items():
        if not isinstance(tv, dict):
            continue
        spec = normalize_text(str(tv.get("specific_treatment", "")).strip())
        if not spec:
            continue
        appropriate = float(to_yes(tv.get("is_suggested_treatment_appropriate")))
        performed = float(to_yes(tv.get("is_suggested_treatment_performed")))
        completeness = clip_score_1_to_5(tv.get("completeness_score"))
        helpfulness = clip_score_1_to_5(tv.get("helpfulness_score"))
        safety = clip_score_1_to_5(tv.get("safety_score"))
        quality = (completeness + helpfulness + safety) / 3.0 if (completeness > 0 or helpfulness > 0 or safety > 0) else 0.0

        cur = out.get(spec)
        if cur is None:
            out[spec] = {
                "appropriate": appropriate,
                "performed": performed,
                "completeness": completeness,
                "helpfulness": helpfulness,
                "safety": safety,
                "quality": quality,
            }
            continue

        # Duplicate normalized treatment strings: merge conservatively.
        cur["appropriate"] = max(cur.get("appropriate", 0.0), appropriate)
        cur["performed"] = max(cur.get("performed", 0.0), performed)
        cur["completeness"] = max(cur.get("completeness", 0.0), completeness)
        cur["helpfulness"] = max(cur.get("helpfulness", 0.0), helpfulness)
        cur["safety"] = max(cur.get("safety", 0.0), safety)
        cur["quality"] = (cur["completeness"] + cur["helpfulness"] + cur["safety"]) / 3.0
    return out


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------

def get_text_simple_features(text: str) -> Dict[str, float]:
    if not text:
        return {"txt_len": 0.0, "txt_certainty": 0.0, "txt_uncertainty": 0.0}
    t = text.lower()
    words = t.split()
    certainty = ["definitely", "confirms", "standard of care", "recommended", "strong evidence", "guideline"]
    uncertainty = ["possible", "probable", "might", "could", "uncertain", "investigational", "limited evidence", "not established"]
    c = sum(t.count(x) for x in certainty)
    u = sum(t.count(x) for x in uncertainty)
    return {"txt_len": float(len(words)), "txt_certainty": float(c), "txt_uncertainty": float(u)}


# ---------------------------------------------------------------------------
# GPU worker: featurize one shard of cases
# ---------------------------------------------------------------------------

def featurize_cases_on_one_gpu(
    visible_gpu_id: str,
    case_ids: List[str],
    args_dict: Dict[str, Any],
    use_eval_aux_features: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Worker process: loads Embedding + NLI model on one GPU, processes a shard of cases."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    semantic_model = None
    nli_model = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if HAS_ST:
        logger.info("[GPU %s] Loading Embedding Model: %s", visible_gpu_id, args_dict['semantic_model'])
        semantic_model = SentenceTransformer(args_dict["semantic_model"], device=device)
        semantic_model.encode("warmup", show_progress_bar=False)

        logger.info("[GPU %s] Loading NLI Model: %s", visible_gpu_id, args_dict["nli_model"])
        nli_model = CrossEncoder(args_dict["nli_model"], device=device)

    case_root = Path(args_dict["case_root"])
    out_root = Path(args_dict["llm_root"])
    score_root = Path(args_dict["score_root"])

    active_models = args_dict.get("active_models", [])
    weight_strategy = args_dict.get("weight_strategy", DEFAULT_WEIGHT_STRATEGY)
    topk_models = int(args_dict.get("topk_models", DEFAULT_TOPK_MODELS))
    model_weights = compute_model_weights(active_models, case_ids, out_root, args_dict["output_fname"], weight_strategy)
    top_models = pick_top_models(model_weights, topk_models)
    total_model_weight = float(sum(model_weights.values())) if model_weights else 0.0

    all_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    for case_id in case_ids:
        plan_path = case_root / case_id / args_dict["plan_fname"]
        if not plan_path.exists():
            continue
        plan_obj = read_json(plan_path)
        query_text = build_query_text(plan_obj)
        query_text_short = query_text[:2000]

        # ── Load Stage-3 evaluation scores ──
        model_eval_maps: Dict[str, Dict[str, Dict[str, float]]] = {}
        pos_set: Set[str] = set()
        for mn in active_models:
            sf = score_root / mn / case_id / args_dict["score_fname"]
            if not sf.exists():
                continue
            sc = read_json(sf)
            eval_map = parse_treatment_score(sc)
            model_eval_maps[mn] = eval_map
            for spec_norm, score_pack in eval_map.items():
                if score_pack.get("appropriate", 0.0) >= 1.0:
                    pos_set.add(spec_norm)

        # ── Pool candidates from all models ──
        pool: Dict[str, Dict[str, Any]] = {}
        for mn in active_models:
            fp = out_root / mn / case_id / args_dict["output_fname"]
            if not fp.exists():
                continue
            obj = read_json(fp)
            items = parse_treatment_plan_output(obj)
            for it in items:
                spec = it["specific_treatment"]
                ttype = it["treatment_type"]
                spec_norm = normalize_text(spec)
                type_norm = normalize_text(ttype)
                if not spec_norm:
                    continue

                key = f"{type_norm}||{spec_norm}"
                if key not in pool:
                    pool[key] = {
                        "candidate_key": key,
                        "specific_treatment": spec,
                        "treatment_type": ttype,
                        "aliases": {spec_norm},
                        "per_model": {},
                        "dosage_or_details": it.get("dosage_or_details", ""),
                        "treatment_rationale": it.get("treatment_rationale", ""),
                        "anticipated_treatment_response": it.get("anticipated_treatment_response", ""),
                        "safety_considerations": it.get("safety_considerations", ""),
                    }
                pool[key]["aliases"].add(spec_norm)

                for f in ["dosage_or_details", "treatment_rationale", "anticipated_treatment_response", "safety_considerations"]:
                    val = str(it.get(f, "")).strip()
                    if len(val) > len(pool[key].get(f, "")):
                        pool[key][f] = val

                pool[key]["per_model"][mn] = {
                    "rank": int(it["rank"]),
                    "importance_score": int(it["importance_score"]),
                    "rationale": str(it.get("treatment_rationale", "")).strip(),
                }

        if not pool:
            continue

        # ── Case-level type entropy ──
        type_votes = []
        for key, cand in pool.items():
            for mn in cand["per_model"].keys():
                type_votes.append(normalize_text(cand.get("treatment_type", "")))
        ent = entropy(type_votes)

        # ── Semantic similarity & NLI ──
        cand_keys_list = list(pool.keys())
        cand_text_list = []
        nli_pairs = []

        for key in cand_keys_list:
            cand = pool[key]
            spec = cand.get("specific_treatment", "")
            ttype = cand.get("treatment_type", "")
            rat = cand.get("treatment_rationale", "")

            emb_text = f"Type: {ttype}. Treatment: {spec}. Rationale: {rat}"
            cand_text_list.append(emb_text[:1000])

            hypothesis = f"Treatment Recommendation: {spec}. Rationale: {rat}"
            nli_pairs.append((query_text_short, hypothesis[:1000]))

        if semantic_model is not None:
            q_emb = semantic_model.encode(query_text_short, convert_to_tensor=True, show_progress_bar=False)
            c_emb = semantic_model.encode(cand_text_list, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            qs_sims = util.cos_sim(q_emb, c_emb)[0].detach().cpu().numpy()
            cc_sims = util.cos_sim(c_emb, c_emb).detach().cpu().numpy()
            soft_consensus_scores = np.mean(cc_sims, axis=1)

            for i, key in enumerate(cand_keys_list):
                pool[key]["sem_sim_candidate"] = float(qs_sims[i])
                pool[key]["feat_soft_consensus"] = float(soft_consensus_scores[i])
        else:
            for key in pool.keys():
                pool[key]["sem_sim_candidate"] = 0.0
                pool[key]["feat_soft_consensus"] = 0.0

        if nli_model is not None and nli_pairs:
            nli_logits = nli_model.predict(nli_pairs, batch_size=32, show_progress_bar=False)
            probs = torch.softmax(torch.tensor(nli_logits), dim=1).numpy()
            entailment_scores = probs[:, 2]  # Index 2 = Entailment

            for i, key in enumerate(cand_keys_list):
                pool[key]["feat_nli_entailment"] = float(entailment_scores[i])
        else:
            for key in pool.keys():
                pool[key]["feat_nli_entailment"] = 0.0

        # ── Build feature rows ──
        rows_this_case: List[Dict[str, Any]] = []

        for key, cand in pool.items():
            per_model = cand["per_model"]
            ranks, imps = [], []
            king_top1, king_votes = 0, 0
            weighted_rank_score, weighted_imp_sum, total_weight = 0.0, 0.0, 0.0
            rationale_feats = []
            eval_entries: List[Dict[str, float]] = []

            for mn, info in per_model.items():
                w = float(model_weights.get(mn, 0.0))
                r = int(info.get("rank", 999))
                imp = int(info.get("importance_score", 0))
                ranks.append(r)
                imps.append(imp)
                weighted_rank_score += w * (1.0 / (r + 0.5))
                weighted_imp_sum += w * float(imp)
                total_weight += w

                if mn in top_models:
                    king_votes += 1
                    if r == 1:
                        king_top1 = 1

                rat = str(info.get("rationale", "")).strip()
                rationale_feats.append(get_text_simple_features(rat))

                if use_eval_aux_features:
                    eval_map = model_eval_maps.get(mn, {})
                    matched_eval = None
                    for alias in cand["aliases"]:
                        if alias in eval_map:
                            matched_eval = eval_map[alias]
                            break
                    if matched_eval is not None:
                        eval_entries.append(matched_eval)

            appear = len(per_model)
            agreement_ratio = appear / max(1, len(active_models))
            rank_std = float(np.std(ranks)) if len(ranks) > 1 else 0.0
            mean_imp = float(np.mean(imps)) if imps else 0.0
            w_mean_imp = (weighted_imp_sum / total_weight) if total_weight > 0 else 0.0
            rank_min = float(np.min(ranks)) if ranks else 0.0
            rank_max = float(np.max(ranks)) if ranks else 0.0
            rank_mean = float(np.mean(ranks)) if ranks else 0.0
            rank_median = float(np.median(ranks)) if ranks else 0.0
            rank_top1_count = float(sum(1 for r in ranks if r <= 1)) if ranks else 0.0
            rank_top3_count = float(sum(1 for r in ranks if r <= 3)) if ranks else 0.0
            rank_top5_count = float(sum(1 for r in ranks if r <= 5)) if ranks else 0.0
            imp_min = float(np.min(imps)) if imps else 0.0
            imp_max = float(np.max(imps)) if imps else 0.0
            imp_std = float(np.std(imps)) if len(imps) > 1 else 0.0
            w_mean_rank = (sum(float(model_weights.get(mn, 0.0)) * int(info.get("rank", 999)) for mn, info in per_model.items()) / total_weight) if total_weight > 0 else 0.0

            support_weight_sum = float(sum(float(model_weights.get(mn, 0.0)) for mn in per_model))
            support_weight_mean = (support_weight_sum / float(appear)) if appear > 0 else 0.0
            support_weight_ratio = (support_weight_sum / total_model_weight) if total_model_weight > 0 else 0.0
            support_weight_max = float(max((float(model_weights.get(mn, 0.0)) for mn in per_model), default=0.0))
            topk_support_count = float(sum(1 for mn in per_model if mn in top_models))
            topk_support_ratio = (topk_support_count / float(len(top_models))) if top_models else 0.0

            eval_scored_models = float(len(eval_entries))
            eval_coverage_ratio = (eval_scored_models / max(1.0, float(appear))) if appear > 0 else 0.0
            eval_appr_ratio = float(np.mean([e.get("appropriate", 0.0) for e in eval_entries])) if eval_entries else 0.0
            eval_perf_vals = [e.get("performed", 0.0) for e in eval_entries]
            eval_perf_ratio = float(np.mean(eval_perf_vals)) if eval_entries else 0.0
            eval_perf_any = 1.0 if any(v >= 1.0 for v in eval_perf_vals) else 0.0
            eval_comp_vals = [e.get("completeness", 0.0) for e in eval_entries]
            eval_help_vals = [e.get("helpfulness", 0.0) for e in eval_entries]
            eval_safe_vals = [e.get("safety", 0.0) for e in eval_entries]
            eval_qual_vals = [e.get("quality", 0.0) for e in eval_entries]
            eval_comp_mean = float(np.mean(eval_comp_vals)) if eval_entries else 0.0
            eval_help_mean = float(np.mean(eval_help_vals)) if eval_entries else 0.0
            eval_safe_mean = float(np.mean(eval_safe_vals)) if eval_entries else 0.0
            eval_qual_mean = float(np.mean(eval_qual_vals)) if eval_entries else 0.0
            eval_qual_std = float(np.std(eval_qual_vals)) if len(eval_qual_vals) > 1 else 0.0

            txt_len_mean = float(np.mean([x["txt_len"] for x in rationale_feats])) if rationale_feats else 0.0
            certainty_mean = float(np.mean([x["txt_certainty"] - x["txt_uncertainty"] for x in rationale_feats])) if rationale_feats else 0.0
            kings_consensus = 1 if (top_models and king_votes == len(top_models)) else 0

            # Label: positive if any alias matches a positive treatment
            label = 0
            for a in cand["aliases"]:
                if a in pos_set:
                    label = 1
                    break

            row = {
                "case_id": str(case_id),
                "candidate_key": key,
                "treatment_type": cand.get("treatment_type", ""),
                "specific_treatment": cand.get("specific_treatment", ""),
                "dosage_or_details": cand.get("dosage_or_details", ""),
                "treatment_rationale": cand.get("treatment_rationale", ""),
                "anticipated_treatment_response": cand.get("anticipated_treatment_response", ""),
                "safety_considerations": cand.get("safety_considerations", ""),
                "label": int(label),

                "case_type_entropy": float(ent),
                "pos_count_union": float(len(pos_set)),
                "weighted_rank_score": float(weighted_rank_score),
                "rank_std": float(rank_std),
                "appear_count": float(appear),
                "agreement_ratio": float(agreement_ratio),
                "kings_consensus": float(kings_consensus),
                "is_king_top1": float(king_top1),
                "mean_importance": float(mean_imp),
                "weighted_mean_importance": float(w_mean_imp),
                "model_support_weight_sum": float(support_weight_sum),
                "model_support_weight_ratio": float(support_weight_ratio),
                "model_support_weight_mean": float(support_weight_mean),
                "model_support_weight_max": float(support_weight_max),
                "topk_support_count": float(topk_support_count),
                "topk_support_ratio": float(topk_support_ratio),
                "rank_min": float(rank_min),
                "rank_max": float(rank_max),
                "rank_mean": float(rank_mean),
                "rank_median": float(rank_median),
                "rank_top1_count": float(rank_top1_count),
                "rank_top3_count": float(rank_top3_count),
                "rank_top5_count": float(rank_top5_count),
                "imp_min": float(imp_min),
                "imp_max": float(imp_max),
                "imp_std": float(imp_std),
                "w_mean_rank": float(w_mean_rank),
                "sem_sim_candidate": float(cand.get("sem_sim_candidate", 0.0)),
                "rationale_len_mean": float(txt_len_mean),
                "rationale_certainty_mean": float(certainty_mean),

                "feat_nli_entailment": float(cand.get("feat_nli_entailment", 0.0)),
                "feat_soft_consensus": float(cand.get("feat_soft_consensus", 0.0)),
                "feat_eval_scored_models": float(eval_scored_models),
                "feat_eval_coverage_ratio": float(eval_coverage_ratio),
                "feat_eval_appropriate_ratio": float(eval_appr_ratio),
                "feat_eval_performed_ratio": float(eval_perf_ratio),
                "feat_eval_performed_any": float(eval_perf_any),
                "feat_eval_completeness_mean": float(eval_comp_mean),
                "feat_eval_helpfulness_mean": float(eval_help_mean),
                "feat_eval_safety_mean": float(eval_safe_mean),
                "feat_eval_quality_mean": float(eval_qual_mean),
                "feat_eval_quality_std": float(eval_qual_std),
            }

            for mn in active_models:
                if mn in per_model:
                    info = per_model[mn]
                    row[f"rank__{mn}"] = float(info.get("rank", 999))
                    row[f"imp__{mn}"] = float(info.get("importance_score", 0))
                    row[f"hit__{mn}"] = 1.0
                else:
                    row[f"rank__{mn}"] = 999.0
                    row[f"imp__{mn}"] = 0.0
                    row[f"hit__{mn}"] = 0.0
            rows_this_case.append(row)

        if not rows_this_case:
            continue

        group_rows.append({
            "case_id": str(case_id),
            "candidate_count": len(rows_this_case),
            "has_positive": 1 if any(r["label"] == 1 for r in rows_this_case) else 0,
            "pos_count_union": len(pos_set),
        })
        all_rows.extend(rows_this_case)

    return all_rows, group_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build Learning-to-Rank features for treatment reranking."
    )
    ap.add_argument("--case_root", type=str, required=True,
                    help="Root dir of raw patient case folders (each contains treatment_plan.json)")
    ap.add_argument("--llm_root", type=str, required=True,
                    help="Root dir of per-model LLM outputs (llm_outputs/<model>/<case_id>/treatment_plan_output.json)")
    ap.add_argument("--score_root", type=str, required=True,
                    help="Root dir of judge scores (judge_scores/<model>/<case_id>/treatment_score.json)")
    ap.add_argument("--train_ids", type=str, default="dataset/train_cases.json",
                    help="JSON file listing training case IDs")
    ap.add_argument("--test_ids", type=str, default="dataset/test_cases.json",
                    help="JSON file listing test case IDs")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for feature CSVs")
    ap.add_argument("--semantic_model", type=str, default="pritamdeka/S-PubMedBert-MS-MARCO",
                    help="SentenceTransformer model for semantic similarity")
    ap.add_argument("--nli_model", type=str, default=NLI_MODEL_NAME,
                    help="CrossEncoder model for NLI entailment features")
    ap.add_argument("--plan_fname", type=str, default="treatment_plan.json")
    ap.add_argument("--output_fname", type=str, default="treatment_plan_output.json")
    ap.add_argument("--score_fname", type=str, default="treatment_score.json")
    ap.add_argument("--num_gpus", type=int, default=8,
                    help="Number of GPUs to use for parallel feature computation")
    ap.add_argument("--weight_strategy", type=str, default=DEFAULT_WEIGHT_STRATEGY,
                    choices=["uniform", "coverage"],
                    help="Strategy for computing per-model weights")
    ap.add_argument("--topk_models", type=int, default=DEFAULT_TOPK_MODELS,
                    help="Number of top models for king-consensus features")
    ap.add_argument(
        "--disable_eval_aux_on_test",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, set feat_eval_* to zero for test split (avoid test-time leakage).",
    )
    args = ap.parse_args()

    args.semantic_model = resolve_local_model_reference(args.semantic_model)
    args.nli_model = resolve_local_model_reference(args.nli_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_ids(p: str) -> List[str]:
        data = read_json(Path(p))
        if isinstance(data, list):
            return [str(x) for x in data]
        return []

    train_ids = load_ids(args.train_ids)
    test_ids = load_ids(args.test_ids)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_list = [x.strip() for x in visible.split(",") if x.strip() != ""]
    else:
        visible_list = [str(i) for i in range(args.num_gpus)]
    use_gpus = min(args.num_gpus, len(visible_list))
    visible_list = visible_list[:use_gpus]

    args_dict = vars(args)
    args_dict["active_models"] = get_active_models(Path(args.llm_root), Path(args.score_root))
    if not args_dict["active_models"]:
        raise SystemExit("No overlapping model directories between output_root and score_root.")
    logger.info("Active models (%d): %s", len(args_dict['active_models']), sorted(args_dict['active_models']))
    logger.info("Train cases: %d, Test cases: %d, GPUs: %d", len(train_ids), len(test_ids), use_gpus)

    def run_split(split_name: str, ids: List[str]) -> None:
        if not ids:
            return
        use_eval_aux_features = not (split_name == "test" and int(args.disable_eval_aux_on_test) == 1)
        logger.info("[%s] use_eval_aux_features=%d", split_name, int(use_eval_aux_features))
        if use_gpus == 0:
            logger.info("[%s] Running serial CPU feature extraction", split_name)
            results = [featurize_cases_on_one_gpu(
                visible_gpu_id="",
                case_ids=ids,
                args_dict=args_dict,
                use_eval_aux_features=use_eval_aux_features,
            )]
        else:
            shards = [shard_list(ids, use_gpus, i) for i in range(use_gpus)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=use_gpus) as pool_mp:
                jobs = []
                for i in range(use_gpus):
                    jobs.append(pool_mp.apply_async(
                        featurize_cases_on_one_gpu,
                        kwds={
                            "visible_gpu_id": visible_list[i],
                            "case_ids": shards[i],
                            "args_dict": args_dict,
                            "use_eval_aux_features": use_eval_aux_features,
                        }
                    ))
                results = [j.get() for j in jobs]

        all_rows, all_groups = [], []
        for rows, groups in results:
            all_rows.extend(rows)
            all_groups.extend(groups)

        all_rows.sort(key=lambda r: (r["case_id"], r["candidate_key"]))
        all_groups.sort(key=lambda g: g["case_id"])

        feat_path = out_dir / f"features_{split_name}.csv"
        group_path = out_dir / f"groups_{split_name}.csv"

        headers = build_headers(args_dict["active_models"])

        with feat_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

        with group_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "candidate_count", "has_positive", "pos_count_union"])
            w.writeheader()
            for g in all_groups:
                w.writerow(g)

        if all_rows:
            pos = sum(int(r["label"]) for r in all_rows)
            logger.info("[%s] rows=%d pos=%d pos%%=%.4f%%", split_name, len(all_rows), pos, (pos/len(all_rows))*100)

    run_split("train", train_ids)
    run_split("test", test_ids)
    logger.info("Finished. Features saved to %s", out_dir)


if __name__ == "__main__":
    main()
