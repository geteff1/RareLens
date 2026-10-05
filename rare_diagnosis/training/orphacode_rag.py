"""Resolve diagnosis names to OrphaCode using local embeddings + optional cache."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# -- Type alias for LLM client (openai.OpenAI or any object with .chat.completions.create) --
LLMClient = Any  # accepts openai.OpenAI or compatible

# -- Visit type helpers (replaces schema.visit_type) --
VisitType = str  # "primary" | "followup"


def normalize_visit_type(vt: Any) -> str:
    s = str(vt or "primary").strip().lower().replace("-", "").replace("_", "")
    return "followup" if s in ("followup", "follow") else "primary"


# -- Inline prompt (replaces core_tool.prompt.templates.diagnosis_orphacode_rag) --
_ORPHACODE_RAG_PROMPT = """
You are an Orphanet medical coding expert.
You will be given a disease name and semantically retrieved candidates (from a knowledge-base Excel file).

Select the OrphaCode that best matches the disease from the candidates.
If no match is found, output Not Found.

You may only judge based on the provided candidates; do not guess.

Output strict JSON only, for example:
{{
  "predicted_orphacode": "558"
}}

Disease name:
{diagnosis_name}

Candidates (sorted by semantic similarity):
{rag_context}
""".strip()


def build_diagnosis_orphacode_rag_prompt(
    *, diagnosis_name: str, candidates: List[Dict[str, Any]]
) -> str:
    lines = []
    for idx, cand in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. Disease: {cand.get('disease_name', '')} | "
            f"OrphaCode: {cand.get('orphacode', '')} | "
            f"Similarity: {float(cand.get('score', 0.0)):.4f}"
        )
    rag_context = "\n".join(lines) if lines else "(no candidates)"
    return _ORPHACODE_RAG_PROMPT.format(
        diagnosis_name=str(diagnosis_name or "").strip(),
        rag_context=rag_context,
    )


_RE_DIGIT = re.compile(r"\d+")
_RE_WORD = re.compile(r"[a-z0-9]+")

DEFAULT_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_CACHE_DIR = (Path(__file__).resolve().parent / "orphacode_rag_cache").resolve()
# Threshold semantics:
# - retrieval floor: scores below this are discarded as weak retrievals
# - llm trigger floor: among kept candidates, scores below this do not trigger LLM disambiguation
# - auto accept: scores above this are accepted directly without LLM
_DEFAULT_AUTO_ACCEPT_SCORE = 0.92
_DEFAULT_RETRIEVAL_FLOOR_SCORE = 0.72

_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}
_ENCODER_CACHE_LOCK = threading.RLock()

_RERANKER_CACHE: Dict[str, CrossEncoder] = {}
_RERANKER_CACHE_LOCK = threading.RLock()

_KB_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_KB_CACHE_LOCK = threading.RLock()
_KB_INFLIGHT: Dict[Tuple[str, str], threading.Event] = {}

_GLOBAL_NAME_CODE_CACHE: Dict[str, Optional[int]] = {}
_GLOBAL_NAME_CODE_CACHE_LOCK = threading.RLock()


def _get_encoder(model_name: str = "") -> Optional[SentenceTransformer]:
    resolved_name = str(model_name or "").strip() or DEFAULT_ENCODER_MODEL
    with _ENCODER_CACHE_LOCK:
        if resolved_name in _ENCODER_CACHE:
            return _ENCODER_CACHE[resolved_name]
        try:
            encoder = SentenceTransformer(resolved_name, device="cpu")
            _ENCODER_CACHE[resolved_name] = encoder
            return encoder
        except Exception as e:
            logger.warning(
                "[OrphaRAG] Failed to load encoder '%s': %s — falling back to lexical matching.",
                resolved_name, e,
            )
            _ENCODER_CACHE[resolved_name] = None
            return None


def _get_reranker(model_name: str = "") -> CrossEncoder:
    resolved_name = str(model_name or "").strip() or DEFAULT_RERANKER_MODEL
    with _RERANKER_CACHE_LOCK:
        reranker = _RERANKER_CACHE.get(resolved_name)
        if reranker is None:
            reranker = CrossEncoder(resolved_name)
            _RERANKER_CACHE[resolved_name] = reranker
        return reranker


def _clean_orphacode(value: Any) -> Optional[int]:
    if value is None:
        return None
    nums = _RE_DIGIT.findall(str(value).strip())
    if not nums:
        return None
    try:
        return int(nums[0])
    except Exception:
        logger.exception("[OrphaRAG] failed to parse orphacode from value=%r nums=%r", value, nums)
        return None


def _tokenize(text: str) -> List[str]:
    return _RE_WORD.findall(str(text or "").lower())


def _lexical_score(query: str, candidate: str) -> float:
    q = str(query or "").strip().lower()
    c = str(candidate or "").strip().lower()
    if not q or not c:
        return 0.0
    q_tokens = set(_tokenize(q))
    c_tokens = set(_tokenize(c))
    if not q_tokens or not c_tokens:
        jacc = 0.0
    else:
        jacc = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    seq = SequenceMatcher(None, q, c).ratio()
    return 0.6 * float(jacc) + 0.4 * float(seq)


def _normalize_entries_obj(data: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        items = data.items()
        for oc_raw, info in items:
            if not isinstance(info, dict):
                continue
            oc = _clean_orphacode(oc_raw)
            name = str(info.get("name") or info.get("disease_name") or "").strip()
            if oc is None or not name:
                continue
            out.append({"orphacode": oc, "disease_name": name})
        return out
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            oc = _clean_orphacode(row.get("orphacode"))
            name = str(row.get("disease_name") or row.get("name") or "").strip()
            if oc is None or not name:
                continue
            out.append({"orphacode": oc, "disease_name": name})
    return out


def _load_entries(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_entries_obj(data)


def _encode_texts(
    texts: List[str],
    batch_size: int = 128,
    embedding_model_name: str = "",
) -> Optional[np.ndarray]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    encoder = _get_encoder(embedding_model_name)
    if encoder is None:
        return None
    vecs = encoder.encode(
        texts,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def _build_index(
    entries: List[Dict[str, Any]],
    batch_size: int = 128,
    embedding_model_name: str = "",
) -> Tuple[Any, np.ndarray]:
    texts = [e["disease_name"] for e in entries]
    emb = _encode_texts(
        texts,
        batch_size=batch_size,
        embedding_model_name=embedding_model_name,
    )
    index: Any = faiss.IndexFlatIP(int(emb.shape[1]))
    index.add(emb)
    return index, emb


def _load_vector_cache_dir(cache_dir: str | Path, expected_model: str = "") -> Optional[Dict[str, Any]]:
    root = Path(cache_dir).resolve()
    entries_path = root / "entries.json"
    emb_path = root / "kb_embeddings.npy"
    index_path = root / "faiss_index.bin"
    meta_path = root / "metadata.json"
    if not (entries_path.exists() and emb_path.exists() and index_path.exists()):
        return None
    try:
        entries = _load_entries(entries_path)
        if not entries:
            return None
        embeddings = np.asarray(np.load(str(emb_path)), dtype=np.float32)
        index = faiss.read_index(str(index_path))
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta_obj, dict):
                metadata = meta_obj
        built_model = str(metadata.get("embedding_model") or "").strip()
        if expected_model and built_model and expected_model != built_model:
            logger.warning(
                "[OrphaRAG] cache embedding model mismatch: requested=%r cache=%r dir=%s",
                expected_model,
                built_model,
                str(root),
            )
        return {
            "entries": entries,
            "index": index,
            "embeddings": embeddings,
            "cache_dir": str(root),
            "metadata": metadata,
        }
    except Exception:
        logger.exception("[OrphaRAG] failed to load vector cache from %s", str(root))
        return None


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    reranker_model_name: str = "",
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    ranked = [dict(c) for c in candidates]
    pairs = [(query, c["disease_name"]) for c in ranked]
    reranker = _get_reranker(reranker_model_name)
    scores = reranker.predict(pairs)
    for cand, score in zip(ranked, scores):
        cand["rerank_score"] = float(score)
    # For OrphaCode mapping, vector `score` is the primary signal (stable, [0..1] cosine/IP),
    # while reranker logits can be uncalibrated and sometimes over-prefer longer/more-specific names.
    # So we sort by vector score first, then rerank_score as a tie-breaker.
    ranked.sort(key=lambda x: (float(x.get("score", 0.0)), float(x.get("rerank_score", 0.0))), reverse=True)
    return ranked[: max(1, int(top_k))]


def _knowledge_base(
    *,
    ontology_path: str | Path,
    embedding_model_name: str = "",
    vector_cache_dir: str | Path | None = None,
    debug: bool = False,
    kb_embedding_batch: int = 128,
) -> Dict[str, Any]:
    cache_dir = str(vector_cache_dir or "").strip()
    cache_key = (
        str(Path(ontology_path).resolve()),
        str(Path(cache_dir).resolve()) if cache_dir else "",
    )
    while True:
        with _KB_CACHE_LOCK:
            cached = _KB_CACHE.get(cache_key)
            if cached is not None:
                return cached
            ev = _KB_INFLIGHT.get(cache_key)
            if ev is None:
                ev = threading.Event()
                _KB_INFLIGHT[cache_key] = ev
                is_builder = True
            else:
                is_builder = False
        if is_builder:
            break
        ev.wait()

    try:
        requested_model = str(embedding_model_name or "").strip() or DEFAULT_ENCODER_MODEL
        if cache_dir:
            cached_kb = _load_vector_cache_dir(cache_dir, expected_model=requested_model)
            if cached_kb is not None:
                kb = {
                    "entries": cached_kb["entries"],
                    "index": cached_kb["index"],
                    "vector_cache_dir": cached_kb.get("cache_dir", ""),
                    "metadata": cached_kb.get("metadata", {}),
                }
                with _KB_CACHE_LOCK:
                    _KB_CACHE[cache_key] = kb
                if debug:
                    print(
                        f"[OrphaRAG] KB: loaded cache dir={kb['vector_cache_dir']} entries={len(kb['entries'])}",
                        flush=True,
                    )
                return kb

        entries = _load_entries(ontology_path)
        index, _ = _build_index(
            entries,
            batch_size=kb_embedding_batch,
            embedding_model_name=requested_model,
        )
        kb = {
            "entries": entries,
            "index": index,
            "vector_cache_dir": "",
            "metadata": {"embedding_model": requested_model},
        }
        with _KB_CACHE_LOCK:
            _KB_CACHE[cache_key] = kb
        if debug:
            print(f"[OrphaRAG] KB: built in-memory index entries={len(entries)}", flush=True)
        return kb
    finally:
        with _KB_CACHE_LOCK:
            ev = _KB_INFLIGHT.pop(cache_key, None)
            if ev is not None:
                ev.set()


def _search_index(
    *,
    query_embeddings: np.ndarray,
    entries: List[Dict[str, Any]],
    index: Any,
    top_k: int,
    known_orphacodes: Optional[set],
    retrieval_score_floor: float,
) -> List[Dict[str, Any]]:
    fetch_k = max(1, int(top_k)) if not known_orphacodes else max(1, int(top_k) * 4)
    scores, indices = index.search(query_embeddings, fetch_k)
    results: List[Dict[str, Any]] = []
    for idx, score in zip(indices[0], scores[0]):
        if int(idx) < 0 or int(idx) >= len(entries):
            continue
        if float(score) < float(retrieval_score_floor):
            break
        entry = entries[int(idx)]
        if known_orphacodes is not None and entry["orphacode"] not in known_orphacodes:
            continue
        results.append(
            {
                "orphacode": entry["orphacode"],
                "disease_name": entry["disease_name"],
                "score": float(score),
            }
        )
        if len(results) >= max(1, int(top_k)):
            break
    return results


def _retrieve_candidates(
    *,
    diagnosis_name: str,
    ontology_path: str | Path,
    embedding_model_name: str,
    llm_client: Optional[LLMClient],
    top_k: int,
    vector_cache_dir: str | Path | None = None,
    debug: bool = False,
    kb_embedding_batch: int = 128,
    known_orphacodes: Optional[set] = None,
    retrieval_score_floor: float = _DEFAULT_RETRIEVAL_FLOOR_SCORE,
    auto_accept_score: float = _DEFAULT_AUTO_ACCEPT_SCORE,
    use_rerank: bool = True,
    rerank_k: int = 5,
) -> Tuple[List[Dict[str, Any]], bool]:
    del llm_client
    kb = _knowledge_base(
        ontology_path=ontology_path,
        embedding_model_name=embedding_model_name,
        vector_cache_dir=vector_cache_dir,
        debug=debug,
        kb_embedding_batch=kb_embedding_batch,
    )
    entries = kb.get("entries") or []
    index = kb.get("index")
    if entries and index is not None:
        query_embeddings = _encode_texts(
            [str(diagnosis_name or "").strip()],
            embedding_model_name=embedding_model_name,
        )
        if query_embeddings is None:
            raise RuntimeError(
                f"[OrphaRAG] Encoder '{embedding_model_name or DEFAULT_ENCODER_MODEL}' failed to load; "
                "cannot perform vector retrieval. Set HF_HOME to your local model cache."
            )
        candidates = _search_index(
            query_embeddings=query_embeddings,
            entries=entries,
            index=index,
            top_k=top_k,
            known_orphacodes=known_orphacodes,
            retrieval_score_floor=retrieval_score_floor,
        )
        if use_rerank and candidates:
            candidates = rerank(diagnosis_name, candidates, top_k=rerank_k)
        skip_llm = bool(candidates and float(candidates[0].get("score", 0.0)) >= float(auto_accept_score))
        return candidates, skip_llm

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        if known_orphacodes is not None and entry.get("orphacode") not in known_orphacodes:
            continue
        score = _lexical_score(str(diagnosis_name or ""), str(entry.get("disease_name") or ""))
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [
        {
            "orphacode": entry["orphacode"],
            "disease_name": entry["disease_name"],
            "score": float(score),
        }
        for score, entry in scored[: max(1, int(top_k))]
    ]
    if use_rerank and out:
        out = rerank(diagnosis_name, out, top_k=rerank_k)
    return out, bool(out and float(out[0].get("score", 0.0)) >= 0.95)


def _retrieve_candidates_batch(
    *,
    diagnosis_names: List[str],
    ontology_path: str | Path,
    embedding_model_name: str,
    llm_client: Optional[LLMClient],
    top_k: int,
    vector_cache_dir: str | Path | None = None,
    debug: bool = False,
    kb_embedding_batch: int = 128,
    known_orphacodes: Optional[set] = None,
    retrieval_score_floor: float = _DEFAULT_RETRIEVAL_FLOOR_SCORE,
    auto_accept_score: float = _DEFAULT_AUTO_ACCEPT_SCORE,
    use_rerank: bool = True,
    rerank_k: int = 5,
) -> Dict[str, Tuple[List[Dict[str, Any]], bool]]:
    del llm_client
    names = [str(n or "").strip() for n in diagnosis_names if str(n or "").strip()]
    if not names:
        return {}
    kb = _knowledge_base(
        ontology_path=ontology_path,
        embedding_model_name=embedding_model_name,
        vector_cache_dir=vector_cache_dir,
        debug=debug,
        kb_embedding_batch=kb_embedding_batch,
    )
    entries = kb.get("entries") or []
    index = kb.get("index")
    if entries and index is not None:
        fetch_k = max(1, int(top_k)) if not known_orphacodes else max(1, int(top_k) * 4)
        query_embeddings = _encode_texts(
            names,
            embedding_model_name=embedding_model_name,
        )
        if query_embeddings is None:
            raise RuntimeError(
                f"[OrphaRAG] Encoder '{embedding_model_name or DEFAULT_ENCODER_MODEL}' failed to load; "
                "cannot perform vector retrieval. Set HF_HOME to your local model cache."
            )
        scores, indices = index.search(query_embeddings, fetch_k)
        out: Dict[str, Tuple[List[Dict[str, Any]], bool]] = {}
        for i, name in enumerate(names):
            candidates: List[Dict[str, Any]] = []
            for idx, score in zip(indices[i], scores[i]):
                if int(idx) < 0 or int(idx) >= len(entries):
                    continue
                if float(score) < float(retrieval_score_floor):
                    break
                entry = entries[int(idx)]
                if known_orphacodes is not None and entry["orphacode"] not in known_orphacodes:
                    continue
                candidates.append(
                    {
                        "orphacode": entry["orphacode"],
                        "disease_name": entry["disease_name"],
                        "score": float(score),
                    }
                )
                if len(candidates) >= max(1, int(top_k)):
                    break
            if use_rerank and candidates:
                candidates = rerank(name, candidates, top_k=rerank_k)
            out[name] = (
                candidates,
                bool(candidates and float(candidates[0].get("score", 0.0)) >= float(auto_accept_score)),
            )
        return out

    out2: Dict[str, Tuple[List[Dict[str, Any]], bool]] = {}
    for name in names:
        out2[name] = _retrieve_candidates(
            diagnosis_name=name,
            ontology_path=ontology_path,
            embedding_model_name=embedding_model_name,
            llm_client=None,
            top_k=top_k,
            vector_cache_dir=vector_cache_dir,
            debug=debug,
            kb_embedding_batch=kb_embedding_batch,
            known_orphacodes=known_orphacodes,
            retrieval_score_floor=retrieval_score_floor,
            auto_accept_score=auto_accept_score,
            use_rerank=use_rerank,
            rerank_k=rerank_k,
        )
    return out2


def _call_llm_json(
    *,
    prompt: str,
    llm_client: Optional[Any],
) -> Optional[str]:
    """Call LLM via openai.OpenAI-compatible client. Returns raw text or None."""
    if llm_client is None:
        return None
    try:
        # llm_client is an openai.OpenAI instance (or compatible)
        response = llm_client.chat.completions.create(
            model=getattr(llm_client, "_rag_model", None) or "gpt-5-nano",
            messages=[
                {"role": "system", "content": "You must output JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
    except Exception as e:
        logger.warning("[OrphaRAG] LLM call failed: %s", e)
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def _llm_pick_orphacode(
    *,
    diagnosis_name: str,
    candidates: List[Dict[str, Any]],
    llm_client: Optional[LLMClient],
    debug: bool = False,
) -> Optional[int]:
    del debug
    if not candidates:
        return None
    prompt = build_diagnosis_orphacode_rag_prompt(
        diagnosis_name=diagnosis_name,
        candidates=candidates,
    )
    resp = _call_llm_json(prompt=prompt, llm_client=llm_client)
    if resp is None:
        return None
    match = re.search(r'"predicted_orphacode"\s*:\s*"([^"]+)"', resp)
    if not match:
        return None
    value = match.group(1).strip()
    if value.lower() == "not found":
        return None
    picked = _clean_orphacode(value)
    if picked is None:
        return None
    allowed = {int(_clean_orphacode(c.get("orphacode")) or -1) for c in (candidates or [])}
    allowed.discard(-1)
    return picked if picked in allowed else None


class OrphaRAG:
    def __init__(
        self,
        ontology_path: str | Path,
        use_rerank: bool = True,
        use_llm_fallback: bool = True,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 50,
        rerank_k: int = 5,
        retrieval_score_floor: float = _DEFAULT_RETRIEVAL_FLOOR_SCORE,
        auto_accept_score: float = _DEFAULT_AUTO_ACCEPT_SCORE,
        llm_trigger_score: Optional[float] = None,
        high_conf_threshold: Optional[float] = None,
        llm_threshold: Optional[float] = None,
        vector_cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        embedding_model_name: str = DEFAULT_ENCODER_MODEL,
    ):
        self.ontology_path = str(ontology_path)
        self.vector_cache_dir = str(vector_cache_dir or "").strip() or None
        self.embedding_model_name = str(embedding_model_name or "").strip() or DEFAULT_ENCODER_MODEL
        self.use_rerank = use_rerank
        self.use_llm_fallback = use_llm_fallback
        self.llm = llm_client
        self.top_k = int(top_k)
        self.rerank_k = int(rerank_k)
        if high_conf_threshold is not None:
            auto_accept_score = float(high_conf_threshold)
        if llm_threshold is not None:
            llm_trigger_score = float(llm_threshold)
        self.retrieval_score_floor = float(retrieval_score_floor)
        self.auto_accept_score = float(auto_accept_score)
        self.llm_trigger_score = float(llm_trigger_score) if llm_trigger_score is not None else self.retrieval_score_floor
        if self.llm_trigger_score < self.retrieval_score_floor:
            logger.warning(
                "[OrphaRAG] llm_trigger_score=%s below retrieval_score_floor=%s; clamping to retrieval floor",
                self.llm_trigger_score,
                self.retrieval_score_floor,
            )
            self.llm_trigger_score = self.retrieval_score_floor
        if self.auto_accept_score < self.llm_trigger_score:
            logger.warning(
                "[OrphaRAG] auto_accept_score=%s below llm_trigger_score=%s; direct accept path may be unreachable",
                self.auto_accept_score,
                self.llm_trigger_score,
            )

        kb = _knowledge_base(
            ontology_path=self.ontology_path,
            embedding_model_name=self.embedding_model_name,
            vector_cache_dir=self.vector_cache_dir,
            kb_embedding_batch=128,
        )
        self.entries = kb.get("entries") or []
        self.index = kb.get("index")

    def resolve(self, name: str) -> Tuple[Optional[int], List[Dict[str, Any]]]:
        candidates, skip_llm = _retrieve_candidates(
            diagnosis_name=name,
            ontology_path=self.ontology_path,
            embedding_model_name=self.embedding_model_name,
            llm_client=self.llm,
            top_k=self.top_k,
            vector_cache_dir=self.vector_cache_dir,
            retrieval_score_floor=self.retrieval_score_floor,
            auto_accept_score=self.auto_accept_score,
            use_rerank=self.use_rerank,
            rerank_k=self.rerank_k,
        )
        if not candidates:
            return None, []
        # Important: candidates may be reordered by rerank_score, while `score` remains the
        # original vector similarity. Threshold decisions should be based on vector scores,
        # not the reranker ordering.
        best_by_score = max(candidates, key=lambda x: float(x.get("score", 0.0)))
        max_score = float(best_by_score.get("score", 0.0))

        # Auto-accept if ANY candidate is above auto_accept_score.
        if max_score >= float(self.auto_accept_score):
            return best_by_score.get("orphacode"), candidates

        # Below LLM trigger floor: strategy C
        # - If LLM fallback is enabled and configured, expand candidate pool and ask LLM to choose.
        # - If LLM fails / not configured, do not pick a code.
        if max_score < float(self.llm_trigger_score):
            if self.use_llm_fallback and self.llm is not None:
                expanded_top_k = max(int(self.top_k), 50)
                expanded_candidates, _ = _retrieve_candidates(
                    diagnosis_name=name,
                    ontology_path=self.ontology_path,
                    embedding_model_name=self.embedding_model_name,
                    llm_client=self.llm,
                    top_k=expanded_top_k,
                    vector_cache_dir=self.vector_cache_dir,
                    retrieval_score_floor=self.retrieval_score_floor,
                    auto_accept_score=self.auto_accept_score,
                    use_rerank=False,  # keep expansion cheap; LLM will decide
                    rerank_k=self.rerank_k,
                )
                picked = _llm_pick_orphacode(
                    diagnosis_name=name,
                    candidates=expanded_candidates,
                    llm_client=self.llm,
                )
                return picked, expanded_candidates
            return None, candidates

        # If retrieve() already decided we can skip LLM, accept the best-by-score candidate.
        if skip_llm:
            return best_by_score.get("orphacode"), candidates
        if self.use_llm_fallback:
            return _llm_pick_orphacode(
                diagnosis_name=name,
                candidates=candidates,
                llm_client=self.llm,
            ), candidates
        # Non-LLM mode: once above llm_trigger_score, pick best-by-vector-score directly.
        return best_by_score.get("orphacode"), candidates


def resolve_batch(rag: OrphaRAG, names: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        code, candidates = rag.resolve(name)
        out[name] = {"orphacode": code, "candidates": candidates}
    return out


def enrich_diagnosis_dict_with_orphacode(
    *,
    diagnosis_items: Dict[str, Any],
    ontology_path: str | Path,
    embedding_model_name: str,
    llm_client: Optional[LLMClient] = None,
    embedding_base_url: str = "",
    embedding_api_key: str = "",
    vector_cache_dir: str | Path | None = None,
    debug: bool = False,
    rag_model_label: str = "",
    kb_embedding_batch: int = 128,
    retrieve_top_k: int = 5,
    ignore_existing_orphacode: bool = False,
    known_orphacodes: Optional[set] = None,
    name_to_code_cache: Optional[Dict[str, Optional[int]]] = None,
    retrieval_score_floor: Optional[float] = None,
    llm_trigger_score: Optional[float] = None,
    auto_accept_score: Optional[float] = None,
    min_retrieval_score: float = _DEFAULT_RETRIEVAL_FLOOR_SCORE,
    high_confidence_score: float = _DEFAULT_AUTO_ACCEPT_SCORE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    del embedding_base_url, embedding_api_key, rag_model_label
    if not isinstance(diagnosis_items, dict):
        return {}, {"resolved_items": 0, "total_items": 0}
    resolved_retrieval_score_floor = (
        float(retrieval_score_floor) if retrieval_score_floor is not None else float(min_retrieval_score)
    )
    resolved_llm_trigger_score = (
        float(llm_trigger_score) if llm_trigger_score is not None else float(resolved_retrieval_score_floor)
    )
    resolved_auto_accept_score = (
        float(auto_accept_score) if auto_accept_score is not None else float(high_confidence_score)
    )

    enriched: Dict[str, Any] = {}
    meta_items: Dict[str, Any] = {}
    if name_to_code_cache is None:
        name_to_code_cache = {}

    to_retrieve: List[str] = []
    for item in diagnosis_items.values():
        if not isinstance(item, dict):
            continue
        existing = _clean_orphacode(item.get("orphacode") or item.get("ORPHAcode") or item.get("orpha"))
        if existing is not None and not bool(ignore_existing_orphacode):
            continue
        name = str(item.get("diagnosis_name") or item.get("disease_name") or "").strip()
        if not name or name in name_to_code_cache:
            continue
        with _GLOBAL_NAME_CODE_CACHE_LOCK:
            if name in _GLOBAL_NAME_CODE_CACHE:
                continue
        to_retrieve.append(name)

    if to_retrieve:
        to_retrieve = list(dict.fromkeys(to_retrieve))
    batch_candidates = (
        _retrieve_candidates_batch(
            diagnosis_names=to_retrieve,
            ontology_path=ontology_path,
            embedding_model_name=embedding_model_name,
            llm_client=llm_client,
            top_k=retrieve_top_k,
            vector_cache_dir=vector_cache_dir,
            debug=debug,
            kb_embedding_batch=kb_embedding_batch,
            known_orphacodes=known_orphacodes,
            retrieval_score_floor=resolved_retrieval_score_floor,
            auto_accept_score=resolved_auto_accept_score,
            use_rerank=True,
            rerank_k=min(10, max(1, retrieve_top_k)),
        )
        if to_retrieve
        else {}
    )

    if to_retrieve:
        names_need_llm: List[str] = []
        candidates_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for name in to_retrieve:
            with _GLOBAL_NAME_CODE_CACHE_LOCK:
                if name in _GLOBAL_NAME_CODE_CACHE:
                    name_to_code_cache[name] = _GLOBAL_NAME_CODE_CACHE[name]
                    continue
            candidates, skip_llm = batch_candidates.get(name, ([], False))
            # Filter candidates by known_orphacodes before any decision so LLM
            # only sees valid codes and the threshold check uses filtered scores.
            if known_orphacodes is not None:
                candidates = [c for c in candidates if c.get("orphacode") in known_orphacodes]
            candidates_by_name[name] = candidates
            if skip_llm and candidates:
                name_to_code_cache[name] = candidates[0].get("orphacode")
            elif candidates:
                # Align with single-item path: only trigger LLM when best vector
                # score meets llm_trigger_score; otherwise skip to avoid wasted calls.
                top_score = float(max(c.get("score", 0.0) for c in candidates))
                if top_score >= resolved_llm_trigger_score:
                    names_need_llm.append(name)
                else:
                    name_to_code_cache[name] = None
            else:
                name_to_code_cache[name] = None

        if names_need_llm and llm_client is not None:
            max_workers = min(8, max(1, len(names_need_llm)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _llm_pick_orphacode,
                        diagnosis_name=name,
                        # candidates already filtered by known_orphacodes above
                        candidates=candidates_by_name.get(name, []),
                        llm_client=llm_client,
                        debug=debug,
                    ): name
                    for name in names_need_llm
                }
                for fut in concurrent.futures.as_completed(futs):
                    name = futs[fut]
                    try:
                        name_to_code_cache[name] = fut.result()
                    except Exception:
                        logger.exception("[OrphaRAG] async llm pick failed for diagnosis_name=%r", name)
                        name_to_code_cache[name] = None

        for name in to_retrieve:
            resolved_n = name_to_code_cache.get(name)
            # Safety net: candidates are already filtered above, but guard against
            # any edge-case where a code outside known_orphacodes slipped through.
            if resolved_n is not None and known_orphacodes is not None and int(resolved_n) not in known_orphacodes:
                resolved_n = None
                name_to_code_cache[name] = None
            with _GLOBAL_NAME_CODE_CACHE_LOCK:
                _GLOBAL_NAME_CODE_CACHE[name] = resolved_n

    for key, item in diagnosis_items.items():
        if not isinstance(item, dict):
            continue
        row = dict(item)
        existing = _clean_orphacode(row.get("orphacode") or row.get("ORPHAcode") or row.get("orpha"))
        if bool(ignore_existing_orphacode):
            existing = None
        diagnosis_name = str(row.get("diagnosis_name") or row.get("disease_name") or "").strip()
        candidates: List[Dict[str, Any]] = []
        resolved = existing

        if resolved is None and diagnosis_name:
            if diagnosis_name in name_to_code_cache:
                resolved = name_to_code_cache[diagnosis_name]
                candidates = batch_candidates.get(diagnosis_name, ([], False))[0]
            else:
                with _GLOBAL_NAME_CODE_CACHE_LOCK:
                    if diagnosis_name in _GLOBAL_NAME_CODE_CACHE:
                        resolved = _GLOBAL_NAME_CODE_CACHE[diagnosis_name]
                if resolved is None:
                    candidates, skip_llm = _retrieve_candidates(
                        diagnosis_name=diagnosis_name,
                        ontology_path=ontology_path,
                        embedding_model_name=embedding_model_name,
                        llm_client=llm_client,
                        top_k=retrieve_top_k,
                        vector_cache_dir=vector_cache_dir,
                        debug=debug,
                        kb_embedding_batch=kb_embedding_batch,
                        known_orphacodes=known_orphacodes,
                        retrieval_score_floor=min_retrieval_score,
                        auto_accept_score=high_confidence_score,
                        use_rerank=True,
                        rerank_k=min(10, max(1, retrieve_top_k)),
                    )
                    # Filter candidates by known_orphacodes before threshold check
                    # so scores and LLM input are both constrained to valid codes.
                    if known_orphacodes is not None:
                        candidates = [c for c in candidates if c.get("orphacode") in known_orphacodes]
                    top_score = float(candidates[0].get("score", 0.0)) if candidates else 0.0
                    if skip_llm and candidates:
                        resolved = candidates[0]["orphacode"]
                    elif candidates and top_score >= resolved_llm_trigger_score:
                        resolved = _llm_pick_orphacode(
                            diagnosis_name=diagnosis_name,
                            candidates=candidates,
                            llm_client=llm_client,
                            debug=debug,
                        )
                    name_to_code_cache[diagnosis_name] = resolved
                    with _GLOBAL_NAME_CODE_CACHE_LOCK:
                        _GLOBAL_NAME_CODE_CACHE[diagnosis_name] = resolved

        if resolved is not None and known_orphacodes is not None and int(resolved) not in known_orphacodes:
            resolved = None

        if resolved is not None:
            row["orphacode"] = int(resolved)
        elif bool(ignore_existing_orphacode):
            # When explicitly ignoring existing codes, clear old/hallucinated value
            # so callers see None instead of a stale code the RAG could not validate.
            row["orphacode"] = None
        enriched[str(key)] = row
        meta_items[str(key)] = {
            "diagnosis_name": diagnosis_name,
            "resolved_orphacode": resolved,
            "used_existing_orphacode": existing is not None,
            "retrieved_candidates": candidates,
        }

    resolved_count = sum(
        1 for item in enriched.values() if isinstance(item, dict) and _clean_orphacode(item.get("orphacode")) is not None
    )
    return enriched, {
        "resolved_items": resolved_count,
        "total_items": len(enriched),
        "items": meta_items,
    }


def enrich_case_bundle_llm_outputs(
    *,
    case_bundle: Dict[str, Any],
    ontology_path: str | Path,
    embedding_model_name: str,
    llm_client: Optional[LLMClient] = None,
    embedding_base_url: str = "",
    embedding_api_key: str = "",
    vector_cache_dir: str | Path | None = None,
    debug: bool = False,
    kb_embedding_batch: int = 128,
    retrieve_top_k: int = 5,
    visit_type: VisitType = "primary",
    ignore_existing_orphacode: bool = False,
    known_orphacodes: Optional[set] = None,
    retrieval_score_floor: Optional[float] = None,
    llm_trigger_score: Optional[float] = None,
    auto_accept_score: Optional[float] = None,
    min_retrieval_score: float = _DEFAULT_RETRIEVAL_FLOOR_SCORE,
    high_confidence_score: float = _DEFAULT_AUTO_ACCEPT_SCORE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = dict(case_bundle or {})
    visit_type = normalize_visit_type(visit_type)
    llm_outputs = bundle.get("llm_outputs")
    if not isinstance(llm_outputs, dict) or not llm_outputs:
        return bundle, {"models_total": 0, "models_enriched": 0}

    if vector_cache_dir is None:
        vector_cache_dir = str(DEFAULT_CACHE_DIR)

    name_to_code_cache: Dict[str, Optional[int]] = {}
    updated_outputs: Dict[str, Any] = {}
    per_model_meta: Dict[str, Any] = {}
    models_enriched = 0
    n_models = len(llm_outputs)

    for mi, (model_name, payload) in enumerate(llm_outputs.items(), start=1):
        if debug:
            print(f"[OrphaRAG] ========== model {mi}/{n_models}: {model_name} ==========", flush=True)
        label = f"{mi}/{n_models} {model_name}"
        if visit_type == "followup":
            if isinstance(payload, dict) and isinstance(payload.get("most_likely_diagnosis"), dict):
                inner = payload.get("most_likely_diagnosis")
            else:
                inner = payload if isinstance(payload, dict) else {}
            enriched_inner, meta = enrich_diagnosis_dict_with_orphacode(
                diagnosis_items=inner if isinstance(inner, dict) else {},
                ontology_path=ontology_path,
                embedding_model_name=embedding_model_name,
                llm_client=llm_client,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                vector_cache_dir=vector_cache_dir,
                debug=debug,
                rag_model_label=label,
                kb_embedding_batch=kb_embedding_batch,
                retrieve_top_k=retrieve_top_k,
                ignore_existing_orphacode=ignore_existing_orphacode,
                known_orphacodes=known_orphacodes,
                retrieval_score_floor=retrieval_score_floor,
                llm_trigger_score=llm_trigger_score,
                auto_accept_score=auto_accept_score,
                name_to_code_cache=name_to_code_cache,
                min_retrieval_score=min_retrieval_score,
                high_confidence_score=high_confidence_score,
            )
            updated_outputs[str(model_name)] = {"most_likely_diagnosis": enriched_inner}
        else:
            # Primary: payload = {"most_likely_diagnosis": {...}, "further_diagnostic_test": {...}}
            # Extract the diagnosis dict, enrich it, then put it back.
            if isinstance(payload, dict) and isinstance(payload.get("most_likely_diagnosis"), dict):
                primary_items = payload["most_likely_diagnosis"]
            else:
                primary_items = payload if isinstance(payload, dict) else {}
            enriched_inner, meta = enrich_diagnosis_dict_with_orphacode(
                diagnosis_items=primary_items,
                ontology_path=ontology_path,
                embedding_model_name=embedding_model_name,
                llm_client=llm_client,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                vector_cache_dir=vector_cache_dir,
                debug=debug,
                rag_model_label=label,
                kb_embedding_batch=kb_embedding_batch,
                retrieve_top_k=retrieve_top_k,
                ignore_existing_orphacode=ignore_existing_orphacode,
                known_orphacodes=known_orphacodes,
                retrieval_score_floor=retrieval_score_floor,
                llm_trigger_score=llm_trigger_score,
                auto_accept_score=auto_accept_score,
                name_to_code_cache=name_to_code_cache,
                min_retrieval_score=min_retrieval_score,
                high_confidence_score=high_confidence_score,
            )
            # Reconstruct full payload with enriched most_likely_diagnosis
            if isinstance(payload, dict) and "most_likely_diagnosis" in payload:
                updated_outputs[str(model_name)] = {**payload, "most_likely_diagnosis": enriched_inner}
            else:
                updated_outputs[str(model_name)] = enriched_inner

        per_model_meta[str(model_name)] = meta
        if int(meta.get("resolved_items", 0)) > 0:
            models_enriched += 1

    bundle["llm_outputs"] = updated_outputs
    rag_meta = {
        "enabled": True,
        "ontology_path": str(Path(ontology_path).resolve()),
        "embedding_model": str(embedding_model_name or "").strip() or DEFAULT_ENCODER_MODEL,
        "embedding_runtime_model": str(embedding_model_name or "").strip() or DEFAULT_ENCODER_MODEL,
        "vector_cache_dir": str(vector_cache_dir or "").strip(),
        "kb_embedding_batch": int(kb_embedding_batch),
        "retrieve_top_k": int(retrieve_top_k),
        "models_total": len(updated_outputs),
        "models_enriched": models_enriched,
        "unique_names_resolved": len(name_to_code_cache),
        "per_model": per_model_meta,
    }
    bundle["orphacode_rag_meta"] = rag_meta
    return bundle, rag_meta


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_ENCODER_MODEL",
    "OrphaRAG",
    "enrich_case_bundle_llm_outputs",
    "enrich_diagnosis_dict_with_orphacode",
    "resolve_batch",
    "rerank",
]
