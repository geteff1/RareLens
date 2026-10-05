#!/usr/bin/env python3
"""
Batch-generate diagnosis outputs from LLMs for training/reproduce.

Walks input_folder for case directories containing primary_consultation.json,
calls one LLM model via the OpenAI-compatible API, parses the JSON response,
and saves outputs under output_folder/<model>/<case_id>/.

Supports two modes:
  - Direct API: pass --base-url and --api-key on the command line.
  - Config file: pass --config pointing to a JSON list of model entries
    (each with "model", "base_url", "api_key", optional "tags").

Per model/case output files:
  - primary: primary_consultation_output.json and
    most_likely_diagnosis_orphacode.json
  - followup: followup_consultation_output.json and
    followup_diagnosis_orphacode.json

Usage:
    # Config-file mode on selected demo cases
    python -m rare_diagnosis.training.generate_llm_outputs \\
        data_500 outputs/diagnosis_demo_primary/llm_outputs \\
        --model GPT-5 \\
        --config llm_config.json \\
        --case-ids outputs/diagnosis_demo_primary/splits/all.json

    # Follow-up stage
    python -m rare_diagnosis.training.generate_llm_outputs \\
        data_500 outputs/diagnosis_demo_followup/llm_outputs \\
        --model GPT-5 --config llm_config.json \\
        --case-ids outputs/diagnosis_demo_followup/splits/all.json \\
        --visit-type followup
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Prompts ─────────────────────────────────────────────────────────────────

DIAGNOSIS_PRIMARY_PROMPT = """
You are an expert clinician.

Your task is to:

1. Read the patient's information carefully.

2. Formulate your own diagnostic hypotheses through robust medical reasoning.

3. Determine the 5 most likely diagnoses:
   3.1 Follow your own reasoning path to identify the five most likely diagnoses
   3.2 Provide brief diagnostic reasoning for each diagnosis
   3.3 Assign a confidence score for each diagnosis (0-10 scale, where 0 = no confidence and 10 = absolute confidence)
   3.4 Sort the diagnoses from highest to lowest confidence score

4. Determine the top 5 crucial diagnostic tests required to reach a final diagnosis:
   4.1 Recommend only tests that are crucial and confirmatory for reaching a final diagnosis
   4.2 Provide brief rationale for why this test is essential
   4.2 Assign a necessity score (0-10) for each diagnostic test based on its importance for confirming the diagnosis(0-10 scale, where 0 = not necessary at all and 10 = absolutely necessary)


5.Output your analysis in the following JSON format:

Output in the following json format:
{{
  "most_likely_diagnosis": {{
    "diagnosis1": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis2": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis3": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis4": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis5": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }}
  }},
  "further_diagnostic_test": {{
    "test1": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test2": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test3": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test4": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test5": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }}
  }}
}}

Here is the patient's information:

{content}
""".strip()

DIAGNOSIS_FOLLOWUP_PROMPT = """
You are an expert clinician.

Your task is to:

1. Read the patient's information carefully.

2. Formulate your own diagnostic hypotheses through robust medical reasoning.

3. Determine the 5 most likely diagnoses:
   3.1 Follow your own reasoning path to identify the five most likely diagnoses
   3.2 Provide brief diagnostic reasoning for each diagnosis
   3.3 Assign a confidence score for each diagnosis (0-10 scale, where 0 = no confidence and 10 = absolute confidence)
   3.4 Sort the diagnoses from highest to lowest confidence score

4.Output your analysis in the following JSON format:

Output in the following json format:
{{
  "most_likely_diagnosis": {{
    "diagnosis1": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis2": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis3": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis4": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis5": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }}
  }}
}}


Here is the patient's information:
{content}
""".strip()

# ── Models that need special handling ──────────────────────────────────────

THINKING_MODELS = {"qwq-32b", "deepseek-r1"}
QWEN3_PREFIXES = ("qwen3-",)


# ── Config loading ─────────────────────────────────────────────────────────

def load_model_config(config_path: str, model_tag: str) -> dict:
    """Load model config (base_url, api_key, model) from a JSON config list."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_list = json.load(f)

    selected = next(
        (cfg for cfg in config_list if model_tag in cfg.get("tags", [])),
        None,
    )
    if selected is None:
        selected = next(
            (cfg for cfg in config_list if cfg.get("model") == model_tag),
            None,
        )
    if selected is not None:
        selected = dict(selected)
        credential_tag = selected.get("credentials_from")
        if credential_tag:
            source = next(
                (
                    cfg for cfg in config_list
                    if credential_tag == cfg.get("model")
                    or credential_tag in cfg.get("tags", [])
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    f"Credential source '{credential_tag}' not found in {config_path}."
                )
            selected.setdefault("base_url", source.get("base_url"))
            selected.setdefault("api_key", source.get("api_key"))
        return selected

    raise ValueError(f"Model '{model_tag}' not found in {config_path}.")


# ── LLM call ──────────────────────────────────────────────────────────────

def get_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 10,
    delay: float = 2.0,
    stream: bool = False,
) -> str | None:
    """Call the LLM API with retry logic."""
    model_lower = model.lower()
    is_thinking = model_lower in THINKING_MODELS
    is_qwen3 = any(model_lower.startswith(p) for p in QWEN3_PREFIXES)

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output the final answer in JSON."},
                {"role": "user", "content": prompt},
            ]

            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Qwen3 models need enable_thinking=False (unless they are thinking models)
            if is_qwen3 and not is_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}

            if stream:
                kwargs["stream"] = True
                response_stream = client.chat.completions.create(**kwargs)
                full_content = ""
                for chunk in response_stream:
                    if chunk.choices and hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
            else:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

        except Exception as e:
            logger.warning("Error: %s. Retry %d/%d", e, attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error("Max retries reached.")
                return None


# ── JSON parsing ──────────────────────────────────────────────────────────

def parse_json(text: str) -> dict:
    """Parse JSON from LLM text output, handling markdown code blocks and bare JSON."""
    # Try: ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try: ``` ... ```
    m = re.search(r"```\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try: bare { ... }
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0).strip())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Unable to parse JSON from the response.", text, 0)


# ── Single case processing ────────────────────────────────────────────────

def process_case(
    case_path: str,
    input_base: str,
    output_base: str,
    client: OpenAI,
    model: str,
    *,
    visit_type: str = "primary",
    temperature: float = 0.0,
    max_retries: int = 10,
    stream: bool = False,
    dry_run: bool = False,
    enable_orphacode_rag: bool = False,
    rag_client: OpenAI | None = None,
    rag_ontology_path: str = "",
    rag_embedding_model: str = "",
    rag_vector_cache_dir: str = "",
    rag_top_k: int = 5,
) -> bool:
    relative_path = os.path.relpath(case_path, input_base)
    output_case_path = os.path.join(output_base, relative_path)
    os.makedirs(output_case_path, exist_ok=True)

    raw_fname, diag_fname = output_filenames(visit_type)
    raw_output_path = os.path.join(output_case_path, raw_fname)
    diag_output_path = os.path.join(output_case_path, diag_fname)

    # Skip if both outputs already exist
    if os.path.exists(raw_output_path) and os.path.exists(diag_output_path):
        logger.info("Output already exists for case '%s'. Skipping.", relative_path)
        return True

    # Read input based on visit type
    if visit_type == "followup":
        input_file = os.path.join(case_path, "follow_up_consultation.json")
    else:
        input_file = os.path.join(case_path, "primary_consultation.json")

    if not os.path.exists(input_file):
        logger.warning("Input '%s' not found in '%s'. Skipping.", os.path.basename(input_file), relative_path)
        return False

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s. Skipping.", input_file)
        return False

    aggregated_content = json.dumps(data, ensure_ascii=False)

    # Select prompt based on visit type
    prompt_template = DIAGNOSIS_FOLLOWUP_PROMPT if visit_type == "followup" else DIAGNOSIS_PRIMARY_PROMPT
    full_prompt = prompt_template.format(content=aggregated_content)

    if dry_run:
        logger.info("[dry-run] case '%s': input keys=%s, prompt_len=%d",
                     relative_path, list(data.keys()), len(full_prompt))
        return True

    response = get_completion(
        client, full_prompt, model,
        temperature=temperature,
        max_retries=max_retries,
        stream=stream,
    )
    if response is None:
        logger.error("Failed to get response for case '%s'.", relative_path)
        return False

    try:
        parsed = parse_json(response)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON for case '%s': %s", relative_path, e)
        return False

    # OrphaCode RAG enrichment (resolve diagnosis names to OrphaCode)
    if enable_orphacode_rag:
        from rare_diagnosis.training.orphacode_rag import enrich_diagnosis_dict_with_orphacode

        mld = parsed.get("most_likely_diagnosis")
        if isinstance(mld, dict) and mld:
            enriched, _meta = enrich_diagnosis_dict_with_orphacode(
                diagnosis_items=mld,
                ontology_path=rag_ontology_path,
                embedding_model_name=rag_embedding_model,
                llm_client=rag_client,
                vector_cache_dir=rag_vector_cache_dir,
                retrieve_top_k=rag_top_k,
            )
            parsed["most_likely_diagnosis"] = enriched

    # Save raw full payload
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=4)

    # Save diagnosis section for feature building
    diagnosis_section = parsed.get("most_likely_diagnosis", {})
    if not isinstance(diagnosis_section, dict):
        diagnosis_section = {}
    with open(diag_output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis_section, f, ensure_ascii=False, indent=4)

    logger.info("Successfully processed case '%s'.", relative_path)
    return True


def output_filenames(visit_type: str) -> tuple[str, str]:
    """Return stage-specific output names so primary and follow-up never collide."""
    if visit_type == "followup":
        return "followup_consultation_output.json", "followup_diagnosis_orphacode.json"
    return "primary_consultation_output.json", "most_likely_diagnosis_orphacode.json"


def collect_pending_cases(all_cases, input_folder, output_folder, visit_type):
    """Return case paths still missing output files."""
    pending = []
    for case_path in all_cases:
        relative_path = os.path.relpath(case_path, input_folder)
        output_case_path = os.path.join(output_folder, relative_path)
        raw_fname, diag_fname = output_filenames(visit_type)
        raw_path = os.path.join(output_case_path, raw_fname)
        diag_path = os.path.join(output_case_path, diag_fname)
        if not (os.path.exists(raw_path) and os.path.exists(diag_path)):
            pending.append(case_path)
    return pending


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate diagnosis outputs from LLMs."
    )
    parser.add_argument("input_folder", help="Input folder with case dirs containing primary_consultation.json")
    parser.add_argument("output_folder", help="Output folder where results will be saved")
    parser.add_argument("--model", required=True,
                        help="Model name/tag (for example GPT-5 or Qwen3-32B)")
    parser.add_argument("--visit-type", choices=("primary", "followup"), default="primary",
                        help="Visit stage: primary (default) or followup")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL (direct mode)")
    parser.add_argument("--api-key", default=None,
                        help="API key (direct mode)")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config list (config mode). "
                             "Each entry: {model, base_url, api_key, tags}")
    parser.add_argument("--case-ids", default=None,
                        help="Optional JSON array of case IDs to process")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of concurrent threads (1 = sequential)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Max retry iterations for unfinished cases")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per LLM call")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming API (needed for some models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse all input JSONs and build prompts without calling the LLM")
    # OrphaCode RAG options
    parser.add_argument("--enable-orphacode-rag", action="store_true",
                        help="Enrich diagnosis with OrphaCode via RAG after LLM generation")
    parser.add_argument("--rag-ontology-path", default="rare_diagnosis/training/orphanet_hierarchy.json",
                        help="Path to Orphanet hierarchy JSON for RAG")
    parser.add_argument("--rag-model", default="gpt-5-nano",
                        help="LLM model for RAG disambiguation")
    parser.add_argument("--rag-embedding-model", default="BAAI/bge-base-en-v1.5",
                        help="Embedding model for RAG retrieval")
    parser.add_argument("--rag-vector-cache-dir", default="rare_diagnosis/orphacode_rag_cache",
                        help="Prebuilt OrphaCode vector-cache directory")
    parser.add_argument("--rag-top-k", type=int, default=5,
                        help="Top-K candidates for RAG retrieval")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        logger.error("Input folder '%s' does not exist.", args.input_folder)
        sys.exit(1)

    # Initialize OpenAI client (not needed for dry-run)
    client = None
    model_name = args.model
    if not args.dry_run:
        if args.config:
            model_cfg = load_model_config(args.config, args.model)
            client = OpenAI(
                base_url=model_cfg.get("base_url"),
                api_key=model_cfg.get("api_key"),
            )
            model_name = model_cfg.get("model", args.model)
        elif args.base_url and args.api_key:
            client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        else:
            logger.error("Provide either --config or both --base-url and --api-key.")
            sys.exit(1)

    # Resolve the RAG model independently. This avoids sending gpt-5-nano to a
    # non-OpenAI generation endpoint when another provider supplies candidates.
    rag_client = None
    if args.enable_orphacode_rag and not args.dry_run:
        if args.config:
            rag_cfg = load_model_config(args.config, args.rag_model)
            rag_base_url = rag_cfg.get("base_url")
            rag_api_key = rag_cfg.get("api_key")
            rag_model_name = rag_cfg.get("model", args.rag_model)
        else:
            rag_base_url = args.base_url
            rag_api_key = args.api_key
            rag_model_name = args.rag_model
        rag_client = OpenAI(
            base_url=rag_base_url,
            api_key=rag_api_key,
        )
        rag_client._rag_model = rag_model_name  # type: ignore[attr-defined]

    output_folder = os.path.join(args.output_folder, args.model)
    os.makedirs(output_folder, exist_ok=True)

    selected_case_ids = None
    if args.case_ids:
        with open(args.case_ids, "r", encoding="utf-8") as f:
            selected_case_ids = {str(x) for x in json.load(f)}

    # Gather only cases eligible for the requested stage.
    all_cases = []
    for root, dirs, files in os.walk(args.input_folder):
        required_input = (
            "follow_up_consultation.json" if args.visit_type == "followup"
            else "primary_consultation.json"
        )
        case_id = os.path.basename(root)
        if required_input in files and (selected_case_ids is None or case_id in selected_case_ids):
            all_cases.append(root)
    all_cases.sort()

    if not all_cases:
        logger.warning("No cases found under %s", args.input_folder)
        return

    worker_count = max(1, args.num_workers)

    for iteration in range(1, args.max_iterations + 1):
        cases_to_process = collect_pending_cases(
            all_cases, args.input_folder, output_folder, args.visit_type
        )
        total_cases = len(cases_to_process)

        if total_cases == 0:
            logger.info("All cases already have outputs. Exiting.")
            return

        logger.info("Iteration %d/%d: %d cases pending.", iteration, args.max_iterations, total_cases)

        with tqdm(total=total_cases, desc=f"Processing Cases (iter {iteration})", unit="case") as pbar:
            if worker_count == 1:
                for case_path in cases_to_process:
                    process_case(
                        case_path, args.input_folder, output_folder,
                        client, model_name,
                        visit_type=args.visit_type,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        stream=args.stream,
                        dry_run=args.dry_run,
                        enable_orphacode_rag=args.enable_orphacode_rag,
                        rag_client=rag_client,
                        rag_ontology_path=args.rag_ontology_path,
                        rag_embedding_model=args.rag_embedding_model,
                        rag_vector_cache_dir=args.rag_vector_cache_dir,
                        rag_top_k=args.rag_top_k,
                    )
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=min(worker_count, total_cases)) as executor:
                    future_to_case = {}
                    for case_path in cases_to_process:
                        future = executor.submit(
                            process_case,
                            case_path, args.input_folder, output_folder,
                            client, model_name,
                            visit_type=args.visit_type,
                            temperature=args.temperature,
                            max_retries=args.max_retries,
                            stream=args.stream,
                            dry_run=args.dry_run,
                            enable_orphacode_rag=args.enable_orphacode_rag,
                            rag_client=rag_client,
                            rag_ontology_path=args.rag_ontology_path,
                            rag_embedding_model=args.rag_embedding_model,
                            rag_vector_cache_dir=args.rag_vector_cache_dir,
                            rag_top_k=args.rag_top_k,
                        )
                        future_to_case[future] = case_path

                    for future in as_completed(future_to_case):
                        case_path = future_to_case[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error("Error processing case '%s': %s",
                                         os.path.relpath(case_path, args.input_folder), e)
                        pbar.update(1)

        if args.dry_run:
            logger.info("Dry run complete; no output files were written.")
            return

        remaining = collect_pending_cases(
            all_cases, args.input_folder, output_folder, args.visit_type
        )
        if not remaining:
            logger.info("All cases have been processed.")
            return
        if iteration == args.max_iterations:
            logger.warning("Reached max iterations (%d). %d cases still missing.",
                           args.max_iterations, len(remaining))


if __name__ == "__main__":
    main()
